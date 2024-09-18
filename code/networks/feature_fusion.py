import torch.nn as nn
import torch
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock

class FF(nn.Module):
    # multi-modal feature fusion method
    def __init__(self,):
        super(FF,self).__init__()
        #se block for feature fusion
        self.linear_f1=nn.Linear(1024+16,256)
        self.linear_f2=nn.Linear(256,1024+16)
        self.sigmoid=nn.Sigmoid()


        #multi-head attention/transformer
        # hidden_size1=1024+16
        # mlp_dim1=256
        # num_heads1=1
        # self.mlp=MLPBlock(hidden_size=hidden_size1,mlp_dim=mlp_dim1,dropout_rate=0.1)
        # self.norm1=nn.LayerNorm(hidden_size1)
        # self.attn=SABlock(hidden_size=hidden_size1,num_heads=num_heads1,dropout_rate=0.1)
        # self.norm2=nn.LayerNorm(hidden_size1)

        #position embeding
        

        #classification
        in_channels1=int(1024+16)
        out_channels=2
        self.flatten=nn.Flatten(1)
        self.classify=nn.Linear(in_channels1,out_channels)

    def forward(self, x_concat: torch.Tensor) -> torch.Tensor:
        # se
        x_concat=x_concat.float()
        x_fus1 = self.linear_f1(x_concat)
        x_fus2 = self.linear_f2(x_fus1)
        x_fus2 = self.sigmoid(x_fus2)
        x_fus_fin = x_concat * x_fus2

        #multi-head attention/transformer
        # x_in=x_concat.float()
        # x_in=torch.unsqueeze(x_in,dim=1)
        # x1=x_in+self.attn(self.norm1(x_in))
        # x2=x1+self.mlp(self.norm2(x1))
        # x_fus_fin=torch.squeeze(x2,dim=1)

        ##classification
        x_fianl=x_fus_fin #[batch,1024+16]
        predict=self.classify(x_fianl) #[batch,2]

        return x_fianl,predict