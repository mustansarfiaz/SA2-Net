# -*- coding: utf-8 -*-
# @Time    : 2022/12/19 8:59 
# @Author  : Mustansar Fiaz
# @File    : gaganet.py

import torch.nn as nn
import torch
import torch.nn.functional as F
from .CTrans import ChannelTransformer

    
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

################# Dynamic kernel for each stage ##################
class AG_new(nn.Module):
    def __init__(self, F_g, F_l,kernel_size=11):
        super(AG_new,self).__init__()
        self.dwconv = nn.Sequential(                        
            LayerNorm(F_g, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(F_g, F_g, kernel_size=1,stride=1,padding=0,bias=True),
            nn.GELU(),
            nn.Conv2d(F_g, F_g, kernel_size=kernel_size,stride=1,padding=kernel_size//2 ,bias=True,groups=F_g),
            )
        
        self.conv = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=5,stride=1,padding=2,bias=True,groups=F_l),            
        )
        self.proj = nn.Conv2d(F_l, F_l, kernel_size=3,stride=1,padding=1,bias=True)

        #self.g_enhance = nn.Conv2d(F_l, F_l, 3, padding=1, groups=F_l)
        
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Sequential(
            nn.Conv2d(F_l * 2, F_l, 1),
            nn.GELU(),
            nn.Conv2d(F_l, F_l, 3, 1, 1),
            nn.Sigmoid()
        )

        #self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(2*F_l, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self,g, x):
        #print(g.shape)
        #print(x.shape)
        #print('*********************8')
        g1 = self.dwconv(g)
        
        x1 = self.conv(x)  
        cga = torch.cat([g1,x1],dim=1)
        #cga = g1+x1
        #w = self.weight(torch.cat([x1, g1], dim=1))
        psi = self.act(cga)
        psi = self.psi(psi)*x1
        
        out = self.proj(psi) + x
        #out = self.psa(out)+out
        return out
    
class UpBlock_attention(nn.Module):
    def __init__(self, F_g, F_l,  out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        #self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.AG = AG_new(F_g=F_g, F_l=F_l)
        self.nConvs = _make_nConv(F_l, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.AG(g=up, x=skip_x)
        #x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(skip_x_att)

# class MLP(nn.Module):
#     """
#     Multilayer Perceptron (MLP)
#     """

#     def __init__(self, channel, bias=True):
#         super().__init__()
#         self.w_1 = nn.Conv2d(channel, channel, bias=bias, kernel_size=1)
#         self.w_2 = nn.Conv2d(channel, channel, bias=bias, kernel_size=1)

#     def forward(self, x):
#         return self.w_2(F.tanh(self.w_1(x)))


# """ The proposed blocks
# """


class PSCA(nn.Module):
    """ Progressive Spectral Channel Attention (PSCA) 
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Conv2d(d_model, d_ff, 1, bias=False)
        self.w_2 = nn.Conv2d(d_ff, d_model, 1, bias=False)
        self.w_3 = nn.Conv2d(d_model, d_model, 1, bias=False)

        nn.init.zeros_(self.w_3.weight)

    def forward(self, x):
        x = self.w_3(x) * x + x
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.w_2(x)
        return x


class ASC(nn.Module):
    """ Attentive Skip Connection
    """

    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        w = self.weight(torch.cat([x, y], dim=1))
        out = (1 - w) * x + w * y
        return out



class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        in_x = x.clone()
        x = self.norm(x)
        x = x + self.act(self.pos(x))
        
        x = self.fc1(x)
        x = self.act(x)
        
        x = self.fc2(x)

        return x+in_x
    


##  Mixed-Scale Feed-forward Network (MSFN)
class Mixed_Scal_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super(Mixed_Scal_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        #x1 = torch.cat([x1_3, x1_5], dim=1)
        #x2 = torch.cat([x2_3, x2_5], dim=1)
        
        x1 = x1_3 * self.sigmoid(x1_5)
        x2 = x2_3 * self.sigmoid(x2_5)

        #x1 = self.relu3_1(self.dwconv3x3_1(x1))
        #x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

   
class BottleNeck_Block(nn.Module):
    def __init__(self, channels,dims, multi_head=True, ffn=False):
        super(BottleNeck_Block, self).__init__()

        self.conv = nn.Conv2d(channels*4, channels+4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(channels*4)

        self.qconv = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, bias=True,groups = channels)
        
        self.conv1 = nn.Conv2d(channels, dims[0], kernel_size=3, stride=1, padding=1, bias=True)        
        self.conv2 = nn.Conv2d(channels, dims[1], kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(channels, dims[2], kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(channels, dims[3], kernel_size=3, stride=1, padding=1, bias=True)
        
        self.gcn1 = Mixed_Scal_FeedForward(dim=channels)
        self.gcn2 = Mixed_Scal_FeedForward(dim=channels)
        self.gcn3 = Mixed_Scal_FeedForward(dim=channels)
        self.gcn4 = Mixed_Scal_FeedForward(dim=channels)
        #self.dfconv = DeformableConv2d(channels,channels)
        
        self.mlp1 = MLP(channels)
        self.mlp2 = MLP(channels)
        self.mlp3 = MLP(channels)
        self.mlp4 = MLP(channels)      
        
        
        self.act = nn.GELU()
        
    def forward(self, input1, input2, input3, input4):
        B, C, H, W = input3.shape

        if input1.size()[2:] != input3.size()[2:]:
            input11 = F.interpolate(input1, size=input3.size()[2:], mode='bilinear')
        if input2.size()[2:] != input3.size()[2:]:
            input22 = F.interpolate(input2, size=input3.size()[2:], mode='bilinear')
        if input3.size()[2:] != input3.size()[2:]:
            input33 = F.interpolate(input3, size=input3.size()[2:], mode='bilinear')
        if input4.size()[2:] != input3.size()[2:]:
            input44 = F.interpolate(input4, size=input3.size()[2:], mode='bilinear')

        input11_a = self.gcn1(input11)
        input22_a = self.gcn2(input22)
        input33_a = self.gcn3(input3)
        input44_a = self.gcn4(input44)
        fuse = torch.cat((input11_a, input22_a, input33_a, input44_a), 1)
        fuse = self.act(self.conv(self.bn(fuse)))
        
        q, gates = torch.split(fuse, (C, 4), 1)
        q = self.act(self.qconv(q))
                
        input11 =   input11_a*(gates[:,0,:,:].unsqueeze(1))*q
        input22 =   input22_a*(gates[:,1,:,:].unsqueeze(1))*q
        input33 =   input33_a*(gates[:,2,:,:].unsqueeze(1))*q
        input44 =   input44_a*(gates[:,3,:,:].unsqueeze(1))*q
        
        input11 = F.interpolate(input11, size=input1.size()[2:], mode='bilinear') + input1
        input22 = F.interpolate(input22, size=input2.size()[2:], mode='bilinear') + input2
        input33 = F.interpolate(input33, size=input3.size()[2:], mode='bilinear') + input3
        input44 = F.interpolate(input44, size=input4.size()[2:], mode='bilinear') + input4
            
        
        input11 = self.mlp1(input11)
        input22 = self.mlp2(input22)
        input33 = self.mlp3(input33)
        input44 = self.mlp4(input44)
        
        input11 = self.conv1(input11)
        input22 = self.conv2(input22)
        input33 = self.conv3(input33)
        input44 = self.conv4(input44)
        
        return input11,input22,input33, input44


class Multi_scale_Fuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layernorm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(2*out_channels, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm2 = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
           

    def forward(self, x, scale_img):
        residual = x        
        x1 = self.layernorm(x)        
        x1 = torch.cat((F.relu(self.conv1(scale_img)), x1), axis=1)
        x1 = self.layernorm2(F.relu(self.conv2(x1)))
        x1 = F.relu(self.conv3(x1))
        out = F.dropout(x1, 0.3)
        #out = F.max_pool2d(x1, (2,2))
            # with skip
        return out + residual
    
class MyNet(nn.Module):
    def __init__(self, n_channels=3, base_channel=64,n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = base_channel
        
        dims = [in_channels, in_channels*2, in_channels*4, in_channels*8]
        
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        
        self.bottleneck = BottleNeck_Block(in_channels,dims)
        self.side_conv2 = nn.Conv2d(dims[1], in_channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv2d(dims[2], in_channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(dims[3], in_channels, kernel_size=3, stride=1, padding=1)
        self.side_conv5 = nn.Conv2d(dims[3], dims[3], kernel_size=3, stride=1, padding=1)
     
        self.up4 = UpBlock_attention(dims[3], dims[3], in_channels*4, nb_Conv=2)
        self.up3 = UpBlock_attention(dims[2], dims[2], in_channels*2, nb_Conv=2)
        self.up2 = UpBlock_attention(dims[1], dims[1], in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(dims[0], dims[0], in_channels, nb_Conv=2)
        self.outc4 = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.outc3 = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.outc2 = nn.Conv2d(in_channels*2, n_classes, kernel_size=(1,1), stride=(1,1))
        self.outc1 = nn.Conv2d(in_channels*4, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss
        
        self.Multi_scale_Fuse1= Multi_scale_Fuse(n_channels, in_channels*2)
        self.Multi_scale_Fuse2= Multi_scale_Fuse(n_channels, in_channels*4)
        self.Multi_scale_Fuse3= Multi_scale_Fuse(n_channels, in_channels*8)
        #self.Multi_scale_Fuse4= Multi_scale_Fuse(n_channels, in_channels*8)
        
        self.scale_img = nn.AvgPool2d(2,2)   
        
    def forward(self, x):
        in_x = x
        x = x.float()
        # Multi-scale input
        #scale_img_2 = self.scale_img(x)
        #scale_img_3 = self.scale_img(scale_img_2)
        #scale_img_4 = self.scale_img(scale_img_3)  
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        #x2 = self.Multi_scale_Fuse1(x2, scale_img_2)
        x3 = self.down2(x2)
        #x3 = self.Multi_scale_Fuse2(x3, scale_img_3)
        x4 = self.down3(x3)
        #x4 = self.Multi_scale_Fuse3(x4, scale_img_4)
        x5 = self.down4(x4)
        x2, x3, x4, x5 = self.side_conv2(x2), self.side_conv3(x3), self.side_conv4(x4), self.side_conv5(x5)
        x1,x2,x3,x4 = self.bottleneck(x1,x2,x3,x4)
                
        x = self.up4(x5, x4)
        x_1 = x
        x = self.up3(x, x3)
        x_2 = x
        x = self.up2(x, x2)
        x_3 = x
        x = self.up1(x, x1)
        shape = x1.size()[2:]
        
        x_1 = F.interpolate(x_1, size=shape, mode='bilinear')
        x_2 = F.interpolate(x_2, size=shape, mode='bilinear')
        x_3 = F.interpolate(x_3, size=shape, mode='bilinear')
        
        
        if self.training:
            if self.n_classes ==1:                
                logits4 = self.last_activation(self.outc4(x))
                logits3 = self.last_activation(self.outc3(x_3))
                logits2 = self.last_activation(self.outc2(x_2))
                logits1 = self.last_activation(self.outc1(x_1))
            else:
                 logits4 = self.outc4(x)
                 logits3 = self.outc4(x_3)
                 logits2 = self.outc4(x_2)
                 logits1 = self.outc1(x_1)
            return logits4, logits3, logits2, logits1
        
        
        if self.n_classes ==1:
            logits = self.last_activation(self.outc4(x))
        else:
            logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1
            
        return logits
        




