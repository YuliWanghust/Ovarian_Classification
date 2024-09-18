'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W,D):
     #return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W*D,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x,y):


        m_batchsize, _, height, width,dimension = x.size()  # x[2,64,5,6,7]
        proj_query = self.query_conv(x)  # [2,8,5,6,7] #batch,channel,x,y
        ##0-batch; 1-channel; 2-x-height; 3-y-width  4-dimension in query, key, value
        proj_query_H = proj_query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width*dimension, -1, height).permute(0, 2, 1)  # [84,5,8]
        proj_query_W = proj_query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height*dimension, -1, width).permute(0, 2, 1)  # [70,6,8]
        proj_query_D = proj_query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height*width, -1,dimension).permute(0, 2, 1)   # [60,7,8]
        proj_key = self.key_conv(y)  # [2,8,5,6,7]
        proj_key_H = proj_key.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width*dimension, -1, height)  # [84,8,5]
        proj_key_W = proj_key.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height*dimension, -1, width) # [70,8,6]
        proj_key_D = proj_key.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1,dimension) # [60,8,7]
        proj_value = self.value_conv(x)  # [2,64,5,6,7]
        proj_value_H = proj_value.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize * width*dimension, -1, height)  # [84,64,5]
        proj_value_W = proj_value.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize * height*dimension, -1, width)  # [70,64,6]
        proj_value_D = proj_value.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize * height * width, -1, dimension)  # [60,64,7]
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width,dimension)).view(m_batchsize, width, height,dimension,height).permute(0,2,1,3,4) # [2,5,6,7,5]
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, dimension, width)  # [2,5,6,7,6]
        energy_D=torch.bmm(proj_query_D,proj_key_D).view(m_batchsize,height,width,dimension,dimension) #[2, 5, 6, 7, 7]
        concate = self.softmax(torch.cat([energy_H, energy_W,energy_D], 4))  # [2,5,6,7,18]

        att_H = concate[:, :,:, :, 0:height].permute(0, 2, 1, 3, 4).contiguous().view(m_batchsize * width*dimension, height, height)  # [84,5,5]
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, :,height:height + width].contiguous().view(m_batchsize * height*dimension, width, width)  # [70,6,6]
        att_D=concate[:,:,:,:,height+width:height+width+dimension].contiguous().view(m_batchsize*height*width,dimension,dimension) #[60,7,7]
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height,dimension).permute(0, 2, 3,1,4)  # [2,64,5,6,7]
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width,dimension).permute(0, 2, 1,3,4)  # [2,64,5,6,7]
        out_D= torch.bmm(proj_value_D, att_D.permute(0,2,1)).view(m_batchsize,height, -1, width,dimension).permute(0, 2, 1,3,4) # [2,64,5,6,7]
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W + out_D) + x  # [2,64,5,6,7]



if __name__ == '__main__':
    model = CrissCrossAttention(64)
    x = torch.randn(2, 64, 5, 6, 7)
    y = torch.randn(2, 64, 5, 6, 7)
    out = model(x,y)
    print(out.shape)