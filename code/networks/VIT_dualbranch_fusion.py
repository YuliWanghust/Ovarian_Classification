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

from typing import Tuple, Union
import torch  # torch==1.9.1
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import ViT  # monai 封装好的包


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            in_channels: int,
            img_size: Tuple[int, int, int],
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
            dropout_rate: float = 0.0,
            act: Union[str, tuple] = ("relu", {"inplace": True}),
            spatial_dims1: int,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()
        avg_pool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[Pool.ADAPTIVEAVG, spatial_dims1]

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,  # feature_size=16
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        self.class_layers_t1 = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    # ("flatten", nn.Flatten(1)),   #
                    # ("out", nn.Linear(in_channels, out_channels)),  # gai forward
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

        # dynamic mlp "Dynamic MLP for Fine-Grained Image Classification by Leveraging Geographical and Temporal Information"
        # dynamic mlp-a
        # dynamic mlp-b
        # dynamic mlp-c

        num_classes = 2
        mlp_cin = 7
        mlp_d1 = 256
        mlp_h1 = 64
        mlp_n1 = 2  # recurrence number
        inplanes1 = 1024

        # dymlp & fc classify for T1
        self.flatten_t1 = nn.Flatten(1)
        self.clinic_net_t1 = FCNet(num_inputs=mlp_cin, num_classes=mlp_d1, num_filts=256)  # get the clinic feature?
        self.loc_att_t1 = get_dynamic_mlp(inplanes=inplanes1, mlp_d=mlp_d1, mlp_h=mlp_h1, mlp_n=mlp_n1,
                                          mlp_type='a')  # convoluted image feature with the kernel generated by clinic feature?

        # dymlp & fc classify for T2
        self.flatten_t2 = nn.Flatten(1)
        self.clinic_net_t2 = FCNet(num_inputs=mlp_cin, num_classes=mlp_d1, num_filts=256)  # get the clinic feature?
        self.loc_att_t2 = get_dynamic_mlp(inplanes=inplanes1, mlp_d=mlp_d1, mlp_h=mlp_h1, mlp_n=mlp_n1, mlp_type='a')

        # classify
        self.fc = nn.Linear(512 * 4, num_classes)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights['state_dict']:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights['state_dict']['module.transformer.patch_embedding.position_embeddings_3d'])
            self.vit.patch_embedding.cls_token.copy_(
                weights['state_dict']['module.transformer.patch_embedding.cls_token'])
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.weight'])
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights['state_dict']['module.transformer.patch_embedding.patch_embeddings.1.bias'])

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights['state_dict']['module.transformer.norm.weight'])
            self.vit.norm.bias.copy_(weights['state_dict']['module.transformer.norm.bias'])



    def forward(self, x_t1: torch.Tensor, x_t2:torch.Tensor, x_cli:torch.Tensor) -> torch.Tensor:
        # x = self.features(x) #[2,1024,x,y,z]
        # x = self.class_layers(x) #[2,2] probability
        # return x

        #T1 dymlp guided image feature extraction
        x, hidden_states_out = self.vit(x_t1)  # x[4,216,768]  hidden_states_out 是list 0-11个tensor
        enc1 = self.encoder1(x_t1)  # [4,16,96,96,96]
        x2 = hidden_states_out[3]  # [4,216,768]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))  # [4,32,48,48,48]
        x3 = hidden_states_out[6]  # [4,216,768]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))  # [4,64,24,24,24]
        x4 = hidden_states_out[9]  # [4,216,768]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))  # [4,128,12,12,12]


        x_t1=self.class_layers_t1(enc4) #[10,1024,1,1,1]
        x_t1=self.flatten_t1(x_t1) #[batch,1024]  #image feature
        clinic1=self.clinic_net_t1(x_cli.float()) #[batch,256]convolated clinic feature
        dy_x_t1=self.loc_att_t1(x_t1,clinic1)  #enhanced image feature by dynamic kernel from clinic feature

        # T2 dymlp guided image feature extraction
        x_t2 = self.features_t2(x_t2)  # [10,1024,3,3,3]
        x_t2 = self.class_layers_t2(x_t2)  # [10,1024,1,1,1]
        x_t2 = self.flatten_t2(x_t2)  # [batch,1024]  #image feature
        clinic2 = self.clinic_net_t2(x_cli.float())  # [batch,256]convolated clinic feature
        dy_x_t2 = self.loc_att_t2(x_t2, clinic2)  # enhanced image feature by dynamic kernel from clinic feature

        ########====dual1/2=====
        dy_x=torch.cat([dy_x_t1,dy_x_t2],dim=1)   #mlp 2048->2

        ########====dual3=======




        predict=self.fc(dy_x) #prediction


        return predict