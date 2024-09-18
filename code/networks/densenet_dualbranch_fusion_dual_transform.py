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
from .ccnet_3d import CrissCrossAttention

class _DenseLayer1(nn.Module):
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

        self.layers1 = nn.Sequential()

        self.layers1.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers1.add_module("relu1", get_act_layer(name=act))
        self.layers1.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers1.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers1.add_module("relu2", get_act_layer(name=act))
        self.layers1.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers1.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features1 = self.layers1(x)
        return torch.cat([x, new_features1], 1)


class _DenseBlock1(nn.Sequential):
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
            layer = _DenseLayer1(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition1(nn.Sequential):
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


#####
class _DenseLayer2(nn.Module):
    def __init__(
        self,
        spatial_dims2: int,
        in_channels2: int,
        growth_rate2: int,
        bn_size2: int,
        dropout_prob2: float,
        act2: Union[str, tuple] = ("relu", {"inplace": True}),
        norm2: Union[str, tuple] = "batch",
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

        out_channels = bn_size2 * growth_rate2
        conv_type: Callable = Conv[Conv.CONV, spatial_dims2]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims2]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm2, spatial_dims=spatial_dims2, channels=in_channels2))
        self.layers.add_module("relu1", get_act_layer(name=act2))
        self.layers.add_module("conv1", conv_type(in_channels2, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=norm2, spatial_dims=spatial_dims2, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act2))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate2, kernel_size=3, padding=1, bias=False))

        if dropout_prob2 > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock2(nn.Sequential):
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
            layer = _DenseLayer2(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act, norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition2(nn.Sequential):
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

class CCAModel(nn.Module):
    def __init__(self,inchannels_cc):
        super(CCAModel, self).__init__()
        self.cca=CrissCrossAttention(inchannels_cc)
    def forward(self,x,y,recurrence):
        for i in range(recurrence):
            out=self.cca(x,y)
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x,y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width,dimension = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = y.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width,dimension)

        out = self.gamma*out + x
        return out

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
        # T1 image feature
        self.features_t1 = nn.Sequential(
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
            block_t1 = _DenseBlock1(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )

            self.features_t1.add_module(f"denseblock{i + 1}", block_t1)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features_t1.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans_t1 = _Transition1(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features_t1.add_module(f"transition{i + 1}", trans_t1)
                in_channels = _out_channels

        # T2 image feature
        in_channels2 = 1
        self.features_t2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        conv_type(in_channels2, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels2 = init_features
        for i, num_layers in enumerate(block_config):
            block_t2 = _DenseBlock2(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels2,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )

            self.features_t2.add_module(f"denseblock{i + 1}", block_t2)
            in_channels2 += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features_t2.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels2)
                )
            else:
                _out_channels2 = in_channels2 // 2
                trans_t2 = _Transition2(
                    spatial_dims, in_channels=in_channels2, out_channels=_out_channels2, act=act, norm=norm
                )
                self.features_t2.add_module(f"transition{i + 1}", trans_t2)
                in_channels2 = _out_channels2



###========transform for finding complementary features
        self.cc_t1_1=CCAModel(inchannels_cc=256)
        self.cc_t1_2 = CCAModel(inchannels_cc=512)
        self.cc_t1_3 = CCAModel(inchannels_cc=1024)
        self.cc_t1_4 = CCAModel(inchannels_cc=1024)

        # self.cam_t1_1=CAM_Module(in_dim=256)
        # self.cam_t1_2 = CAM_Module(in_dim=512)
        # self.cam_t1_3 = CAM_Module(in_dim=1024)
        # self.cam_t1_4 = CAM_Module(in_dim=1024)

        # self.t1_1c=nn.Conv3d(in_channels=256*2, out_channels=256, kernel_size=1)
        # self.t1_2c = nn.Conv3d(in_channels=512 * 2, out_channels=512, kernel_size=1)
        # self.t1_3c = nn.Conv3d(in_channels=1024 * 2, out_channels=1024, kernel_size=1)
        # self.t1_4c = nn.Conv3d(in_channels=1024 * 2, out_channels=1024, kernel_size=1)

        self.cc_t2_1 = CCAModel(inchannels_cc=256)
        self.cc_t2_2 = CCAModel(inchannels_cc=512)
        self.cc_t2_3 = CCAModel(inchannels_cc=1024)
        self.cc_t2_4 = CCAModel(inchannels_cc=1024)

        # self.cam_t2_1 = CAM_Module(in_dim=256)
        # self.cam_t2_2 = CAM_Module(in_dim=512)
        # self.cam_t2_3 = CAM_Module(in_dim=1024)
        # self.cam_t2_4 = CAM_Module(in_dim=1024)

        # self.t2_1c = nn.Conv3d(in_channels=256 * 2, out_channels=256, kernel_size=1)
        # self.t2_2c = nn.Conv3d(in_channels=512 * 2, out_channels=512, kernel_size=1)
        # self.t2_3c = nn.Conv3d(in_channels=1024 * 2, out_channels=1024, kernel_size=1)
        # self.t2_4c = nn.Conv3d(in_channels=1024 * 2, out_channels=1024, kernel_size=1)

        # pooling and classification
        self.class_layers_t1 = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    #("flatten", nn.Flatten(1)),   #
                    #("out", nn.Linear(in_channels, out_channels)),  # gai forward
                ]
            )
        )

        self.class_layers_t2 = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    # ("flatten", nn.Flatten(1)),   #
                    # ("out", nn.Linear(in_channels, out_channels)),  # gai forward
                ]
            )
        )


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

    # dymlp & fc classify for T1
        self.flatten_t1 = nn.Flatten(1)
        self.clinic_net_t1=FCNet(num_inputs=mlp_cin,num_classes=mlp_d1,num_filts=256)   #get the clinic feature?
        self.loc_att_t1=get_dynamic_mlp(inplanes=inplanes1,mlp_d=mlp_d1,mlp_h=mlp_h1,mlp_n=mlp_n1,mlp_type='a')  #convoluted image feature with the kernel generated by clinic feature?

    # dymlp & fc classify for T2
        self.flatten_t2 = nn.Flatten(1)
        self.clinic_net_t2 = FCNet(num_inputs=mlp_cin, num_classes=mlp_d1, num_filts=256)  # get the clinic feature?
        self.loc_att_t2 = get_dynamic_mlp(inplanes=inplanes1, mlp_d=mlp_d1, mlp_h=mlp_h1, mlp_n=mlp_n1, mlp_type='a')

    # classify
        self.fc = nn.Linear(512 * 4, num_classes)

        # for m in self.modules():
        #     if isinstance(m, conv_type):
        #         nn.init.kaiming_normal_(torch.as_tensor(m.weight))
        #     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        #         nn.init.constant_(torch.as_tensor(m.weight), 1)
        #         nn.init.constant_(torch.as_tensor(m.bias), 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x_t1: torch.Tensor, x_t2:torch.Tensor, x_cli:torch.Tensor) -> torch.Tensor:
        #T1 dymlp guided image feature extraction
        x_t1=self.features_t1.conv0(x_t1) #[10,64,48,48,48]
        x_t1 = self.features_t1.norm0(x_t1)  # [10,64,48,48,48]
        x_t1 = self.features_t1.relu0(x_t1)  # [10,64,48,48,48]
        x_t1 = self.features_t1.pool0(x_t1)  # [10,64,24,24,24]
        x_t1 = self.features_t1.denseblock1(x_t1)  # [10,256,24,24,24]   out for transform
        # t2
        x_t2 = self.features_t2.conv0(x_t2)  # [10,64,48,48,48]
        x_t2 = self.features_t2.norm0(x_t2)  # [10,64,48,48,48]
        x_t2 = self.features_t2.relu0(x_t2)  # [10,64,48,48,48]
        x_t2 = self.features_t2.pool0(x_t2)  # [10,64,24,24,24]
        x_t2 = self.features_t2.denseblock1(x_t2)  # [10,256,24,24,24]   out for transform
        #complementary1
        xx_t1 = self.cc_t1_1(x_t1,x_t2,2) # [10,256,24,24,24]  cc spatial attention
        xx_t2 = self.cc_t2_1(x_t2,x_t1,2) # [10,256,24,24,24]
        # xx_t1_c=self.cam_t1_1(x_t1,x_t2) #channel attention
        # xx_t2_c = self.cam_t2_1(x_t2, x_t1)
        # x_t1=xx_t1+xx_t1_c
        # x_t2=xx_t2+xx_t2_c
        x_t1 = xx_t1
        x_t2 = xx_t2
        # x_t1 = torch.cat([x_t1, xx_t1], dim=1)
        # x_t2 = torch.cat([x_t2, xx_t2], dim=1)
        # x_t1=self.t1_1c(x_t1)
        # x_t2=self.t2_1c(x_t2)

        #t1
        x_t1 = self.features_t1.transition1(x_t1)  # [10,128,12,12,12]
        x_t1 = self.features_t1.denseblock2(x_t1)  # [10,512,12,12,12]   out for transform
        #t2
        x_t2 = self.features_t2.transition1(x_t2)  # [10,128,12,12,12]
        x_t2 = self.features_t2.denseblock2(x_t2)  # [10,512,12,12,12]   out for transform
        #complementary2
        xx_t1 = self.cc_t1_2(x_t1, x_t2, 2)  # [10,512,12,12,12]
        xx_t2 = self.cc_t2_2(x_t2, x_t1, 2)  # [10,512,12,12,12]
        # xx_t1_c = self.cam_t1_2(x_t1, x_t2)
        # xx_t2_c = self.cam_t2_2(x_t2, x_t1)
        # x_t1 = xx_t1 + xx_t1_c
        # x_t2 = xx_t2 + xx_t2_c
        x_t1 = xx_t1
        x_t2 = xx_t2
        # x_t1 = torch.cat([x_t1,xx_t1],dim=1)
        # x_t2 = torch.cat([x_t2,xx_t2],dim=1)
        # x_t1 = self.t1_2c(x_t1)
        # x_t2 = self.t2_2c(x_t2)

        #t1
        x_t1 = self.features_t1.transition2(x_t1)  # [10,256,6,6,6]
        x_t1 = self.features_t1.denseblock3(x_t1)  # [10,1024,6,6,6]     out for transform
        #t2
        x_t2 = self.features_t2.transition2(x_t2)  # [10,256,6,6,6]
        x_t2 = self.features_t2.denseblock3(x_t2)  # [10,1024,6,6,6]     out for transform
        #complementary3
        xx_t1 = self.cc_t1_3(x_t1, x_t2, 2)  # [10,256,24,24,24]
        xx_t2 = self.cc_t2_3(x_t2, x_t1, 2)  # [10,256,24,24,24]
        # xx_t1_c = self.cam_t1_3(x_t1, x_t2)
        # xx_t2_c = self.cam_t2_3(x_t2, x_t1)
        # x_t1 = xx_t1 + xx_t1_c
        # x_t2 = xx_t2 + xx_t2_c
        x_t1 = xx_t1
        x_t2 = xx_t2
        # x_t1 = torch.cat([x_t1, xx_t1], dim=1)
        # x_t2 = torch.cat([x_t2, xx_t2], dim=1)
        # x_t1 = self.t1_3c(x_t1)
        # x_t2 = self.t2_3c(x_t2)

        #t1
        x_t1 = self.features_t1.transition3(x_t1)  # [10,512,3,3,3]
        x_t1 = self.features_t1.denseblock4(x_t1)  # [10,1024,3,3,3]     out for transform
        # t2
        x_t2 = self.features_t2.transition3(x_t2)  # [10,512,3,3,3]
        x_t2 = self.features_t2.denseblock4(x_t2)  # [10,1024,3,3,3]     out for transform
        # complementary4
        xx_t1 = self.cc_t1_4(x_t1, x_t2, 2)  # [10,256,24,24,24]
        xx_t2 = self.cc_t2_4(x_t2, x_t1, 2)  # [10,256,24,24,24]
        # xx_t1_c = self.cam_t1_4(x_t1, x_t2)
        # xx_t2_c = self.cam_t2_4(x_t2, x_t1)
        # x_t1 = xx_t1 + xx_t1_c
        # x_t2 = xx_t2 + xx_t2_c
        x_t1 = xx_t1
        x_t2 = xx_t2
        # x_t1 = torch.cat([x_t1, xx_t1], dim=1)
        # x_t2 = torch.cat([x_t2, xx_t2], dim=1)
        # x_t1 = self.t1_4c(x_t1)
        # x_t2 = self.t2_4c(x_t2)

        x_t1 = self.features_t1.norm5(x_t1)  # [10,1024,3,3,3]
        x_t2 = self.features_t2.norm5(x_t2)  # [10,1024,3,3,3]

        # T1 dymlp guided image feature extraction
        x_t1=self.class_layers_t1(x_t1) #[10,1024,1,1,1]
        x_t1=self.flatten_t1(x_t1) #[batch,1024]  #image feature
        clinic1=self.clinic_net_t1(x_cli.float()) #[batch,256]convolated clinic feature
        dy_x_t1=self.loc_att_t1(x_t1,clinic1)  #enhanced image feature by dynamic kernel from clinic feature

        # T2 dymlp guided image feature extraction
        x_t2 = self.class_layers_t2(x_t2)  # [10,1024,1,1,1]
        x_t2 = self.flatten_t2(x_t2)  # [batch,1024]  #image feature
        clinic2 = self.clinic_net_t2(x_cli.float())  # [batch,256]convolated clinic feature
        dy_x_t2 = self.loc_att_t2(x_t2, clinic2)  # enhanced image feature by dynamic kernel from clinic feature

        ########==== fc for classifying =====
        dy_x=torch.cat([dy_x_t1,dy_x_t2],dim=1)   #mlp 2048->2
        predict=self.fc(dy_x) #prediction
        return predict