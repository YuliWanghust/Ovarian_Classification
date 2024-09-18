import torch

class PAM_Module(torch.nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = torch.nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = torch.nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = torch.nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W X D)
            returns :
                out : attention value + input feature
                attention: B X (HxWXD) X (HxWXD)
        """
        m_batchsize, C, height, width,dimension = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*dimension).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*dimension)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*dimension)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width,dimension)

        out = self.gamma*out + x
        return out


class CAM_Module(torch.nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax  = torch.nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W X D)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, dimension = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out