import re
from collections import OrderedDict
from typing import Callable, Sequence, Type, Union

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock
from .dynamic_mlp import FCNet,get_dynamic_mlp

class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    """
    Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    This network is non-determistic When `spatial_dims` is 3 and CUDA is enabled. Please check the link below
    for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    #("flatten", nn.Flatten(1)),   #
                    #("out", nn.Linear(in_channels, out_channels)),  # gai forward
                ]
            )
        )

        #mlp
        # self.fn=get_act_layer("GELU")
        # self.linear1=nn.Linear(7,64)
        # self.linear2=nn.Linear(64,16)
        # self.drop1=nn.Dropout(0.1)
        # self.drop2=nn.Dropout(0.1)

        #dynamic mlp "Dynamic MLP for Fine-Grained Image Classification by Leveraging Geographical and Temporal Information"
        #dynamic mlp-a
        #dynamic mlp-b
        #dynamic mlp-c



        num_classes=2
        mlp_cin=7
        mlp_d1=256
        mlp_h1=64
        mlp_n1=2

        inplanes1=1024

        self.fc=nn.Linear(512*2,num_classes)

        self.clinic_net=FCNet(num_inputs=mlp_cin,num_classes=mlp_d1,num_filts=256)   #get the clinic feature?
        self.loc_att=get_dynamic_mlp(inplanes=inplanes1,mlp_d=mlp_d1,mlp_h=mlp_h1,mlp_n=mlp_n1,mlp_type='a')  #convoluted image feature with the kernel generated by clinic feature?





        #se block for feature fusion
        # self.linear_f1=nn.Linear(1024+16,256)
        # self.linear_f2=nn.Linear(256,1024+16)
        # self.sigmoid=nn.Sigmoid()
        #
        #
        # #classification
        # in_channels1=int(1024+16)
        self.flatten=nn.Flatten(1)
        # self.classify=nn.Linear(in_channels1,out_channels)

        # for m in self.modules():
        #     if isinstance(m, conv_type):
        #         nn.init.kaiming_normal_(torch.as_tensor(m.weight))
        #     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        #         nn.init.constant_(torch.as_tensor(m.weight), 1)
        #         nn.init.constant_(torch.as_tensor(m.bias), 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor,x_cli:torch.Tensor) -> torch.Tensor:
        # x = self.features(x) #[2,1024,15,11,3]
        # x = self.class_layers(x) #[2,2] probability
        # return x
        x=self.features(x) #[10,1024,3,3,3]
        x=self.class_layers(x) #[10,1024,1,1,1]
        x=self.flatten(x) #[batch,1024]  #image feature

        clinic=self.clinic_net(x_cli.float()) #[batch,256]convolated clinic feature
        dy_x=self.loc_att(x,clinic)  #enhanced image feature by dynamic kernel from clinic feature
        predict=self.fc(dy_x) #prediction

        # #mlp
        # x_cli=x_cli.float()
        # x_cli_m=self.fn(self.linear1(x_cli))
        # x_cli_m=self.drop1(x_cli_m)
        # x_cli_m=self.linear2(x_cli_m)
        # x_cli_m=self.drop2(x_cli_m)
        #
        # x_fus=torch.cat([x,x_cli_m],dim=1) #[batch,1024+16]

        #eca
        # input of conv1d [batch,1,1024+16]

        #se
        # x_fus1=self.linear_f1(x_fus)
        # x_fus2=self.linear_f2(x_fus1)
        # x_fus2=self.sigmoid(x_fus2)
        # x_fus_fin=x_fus*x_fus2



        #transformer-fusion
        #self-attention
        # similarity=torch.mm(x_fus,x_fus)
        # similarity=similarity.softmax(dim=-1).log()
        # x_fus1 = x_fus * similarity
        #errors in loss backward



        # #classification
        # x_fianl=x_fus_fin #[batch,1024+16]
        # #x_fianl=x_fianl.float()

        # predict=self.classify(x_fianl) #[batch,2]

        return dy_x,predict