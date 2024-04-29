#from lib.pvtv2 import pvt_v2_b2
from lib.P2P import p2t_base
#from lib.TransXNet import transxnet_s
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicConv2d(nn.Module):  # Conv3*3+BN+ReLU
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #x=self.relu(x)
        return x
class MSF(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MSF, self).__init__()
        self.relu = nn.ReLU(True)
        self.atrous_block1=nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

        self.atrous_block2=nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

        self.atrous_block3=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=4,dilation=4),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        self.conv=nn.Conv2d(ch_out*3, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x1=self.atrous_block1(x)
        #print('x1',x1.shape)
        x2=self.atrous_block2(x)
        #print('x2', x2.shape)
        x3=self.atrous_block3(x)
        #print('x3', x3.shape)
        x=torch.cat((x1, x2, x3), dim=1)
        x=self.relu(self.conv(x)+x)
        return x
class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            # self.sr = nn.Sequential(
            #     nn.Conv2d(dim, dim,
            #                kernel_size=sr_ratio + 3,
            #                stride=sr_ratio,
            #                padding=(sr_ratio + 3) // 2,
            #                groups=dim,
            #                bias=False
            #               ),
            #     nn.Conv2d(dim, dim,
            #                kernel_size=1,
            #                groups=dim,
            #                bias=False
            #
            #                 ), )
            self.sr = MSF(dim, dim)
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = max_out
#         return self.sigmoid(out)
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
# class CS(nn.Module):
#     def __init__(self, in_planes, ratio=16, kernel_size=7):
#         super(CS, self).__init__()
#         self.ca = ChannelAttention(in_planes, ratio)
#         self.sa = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         ca = self.ca(x)
#         sa = self.sa(x)
#         return ca * x + sa * x
class SA(nn.Module):

    def __init__(self, channel, groups=16):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Replace PaddlePaddle code with PyTorch code
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // groups, channel //  groups)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.view(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b*self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=2)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        #print('xn',xn.shape)
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=2)
        out = out.view(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
# class SA(nn.Module):
#
#     def __init__(self, channel, groups=16):
#         super().__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         # Replace PaddlePaddle code with PyTorch code
#         self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
#         self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
#
#         self.sigmoid = nn.Sigmoid()
#         self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
#
#     @staticmethod
#     def channel_shuffle(x, groups):
#         b, c, h, w = x.shape
#
#         x = x.view(b, groups, -1, h, w)
#         x = x.permute(0, 2, 1, 3, 4)
#
#         # flatten
#         x = x.reshape(b, -1, h, w)
#
#         return x
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         x = x.view(b * self.groups, -1, h, w)
#         x_0, x_1 = x.chunk(2, dim=1)
#
#         # channel attention
#         xn = self.avg_pool(x_0)
#         xn = self.cweight * xn + self.cbias
#         xn = x_0 * self.sigmoid(xn)
#
#         # spatial attention
#         xs = self.gn(x_1)
#         xs = self.sweight * xs + self.sbias
#         xs = x_1 * self.sigmoid(xs)
#
#         # concatenate along channel axis
#         out = torch.cat([xn, xs], dim=1)
#         out = out.view(b, -1, h, w)
#
#         out = self.channel_shuffle(out, 2)
#         return out
#
# class CFM(nn.Module):
#     def __init__(self, channel):
#         super(CFM, self).__init__()
#         self.relu = nn.ReLU(True)
#         #self.upsample = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample5 = BasicConv2d(2 * channel,  channel, 3, padding=1)
#         self.conv_concat2 = BasicConv2d( channel, 2 * channel, 3, padding=1)
#         self.conv_concat3 = BasicConv2d(channel, 3 * channel, 3, padding=1)
#         self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
#         #self.AFF=AFF(channel)
#         self.sa = SA(channel)
#         self.psi = nn.Sequential(
#             nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.conv1x1 = nn.Conv2d(2*channel, channel, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         #self.sigmoid = nn.Sigmoid()
#     def forward(self, x1, x2, x3):  # x4_t, x3_t, x2_t
#         x1_1 = x1
#         x2_1 = self.conv_upsample1(self.upsample(x1)) * x2  # a
#         #x2_1 = self.sigmoid(self.conv_upsample1(self.upsample(x1) + x2)) * x2
#
#         x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
#                * self.conv_upsample3(self.upsample(x2)) * x3  # b
#         #x3_1 = self.sigmoid(self.conv_upsample1(self.upsample(x2_1) + x3)) * x3
#         #x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
#         x2_22=self.conv_upsample4(self.upsample(x1_1))
#         # print('x2_22',x2_22.shape)
#         # print('x2_1', x2_1.shape)
#         #x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
#         x2_2=torch.add(x2_1,x2_22 )
#         x2_2 = self.sa(x2_2)
#         ps1 = self.relu(x2_2)
#         x2_2 = self.psi(ps1)
#         fea1 = x2_1 * x2_2
#         fea2 = x2_22 * x2_2
#         fea = torch.concat((fea1, fea2), axis=1)
#         x2_2 = self.conv1x1(fea)
#         #print('x2_2', x2_2.shape)
#         x2_2 = self.conv_concat2(x2_2)
#        # print('x3_1', x3_1.shape)
#         #a=self.conv_upsample5(self.upsample(x2_2))
#        # print('a',a.shape)
#         #x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
#         x3_2=torch.add(x3_1, self.conv_upsample5(self.upsample(x2_2)))
#         x3_2 = self.sa(x3_2)
#         ps2 = self.relu(x3_2)
#         x3_2 = self.psi(ps2)
#         fea11 = x3_1 * x3_2
#         fea22 = self.conv_upsample5(self.upsample(x2_2)) * x3_2
#         fea = torch.concat((fea11, fea22), axis=1)
#         x3_2 = self.conv1x1(fea)
#         x3_2 = self.conv_concat3(x3_2)
#
#         x1 = self.conv4(x3_2)
#
#         return x1
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channel, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channel // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)  # Y
        xg = self.global_att(xa)  # X
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel,  channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d( channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.AFF=AFF(channel)
    def forward(self, x1, x2, x3):  # x4_t, x3_t, x2_t
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2  # a
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3  # b
        #x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_22=self.conv_upsample4(self.upsample(x1_1))
        # print('x2_22',x2_22.shape)
        # print('x2_1', x2_1.shape)
        x2_2=self.AFF(x2_1,x2_22 )
        #print('x2_2', x2_2.shape)ssh -p 39019 root@region-9.autodl.pro
        x2_2 = self.conv_concat2(x2_2)
       # print('x3_1', x3_1.shape)
        a=self.conv_upsample5(self.upsample(x2_2))
       # print('a',a.shape)
        #x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2=self.AFF(x3_1, self.conv_upsample5(self.upsample(x2_2)))
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1
# class GCN(nn.Module):
#     def __init__(self, num_state, num_node, bias=False):
#         super(GCN, self).__init__()
#         self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
#         h = h - x
#         h = self.relu(self.conv2(h))
#         return h
#
#
# class SAM(nn.Module):
#     def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
#         super(SAM, self).__init__()
#
#         self.normalize = normalize
#         self.num_s = int(plane_mid)
#         self.num_n = (mids) * (mids)
#         self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
#
#         self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
#         self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
#         self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
#         self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)
#
#     def forward(self, x, edge):
#         edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))
#
#         n, c, h, w = x.size()
#         edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)
#
#         x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
#         x_proj = self.conv_proj(x)
#         x_mask = x_proj * edge
#
#         x_anchor1 = self.priors(x_mask)
#         x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
#         x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
#
#         x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
#         x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)
#
#         x_rproj_reshaped = x_proj_reshaped
#
#         x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
#         if self.normalize:
#             x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
#         x_n_rel = self.gcn(x_n_state)
#
#         x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
#         x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
#         out = x + (self.conv_extend(x_state))
#
#         return out

class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.num=num_in
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))#AP

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)  # 1*1卷积 W1
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)  # W2
        #self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)  # W4
        #self.conv_extend1=nn.Conv2d(num_in*num_in, num_in, kernel_size=1, bias=False)
        self.sa = SA(num_in)
        self.psi = nn.Sequential(
            nn.Conv2d(num_in, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(num_in,num_in, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.AFF = AFF(num_in)
       # self.con=nn.Conv2d(self.num_s,2 * self.num_s, kernel_size=1, stride=1, padding=0)
    def forward(self, x, edge):  # T1,T2
        #print('t2', edge.shape)
        edge1 = F.upsample(edge, (x.size()[-2], x.size()[-1]))  # W3  F.upsample 函数对张量 edge 进行了上采样，使其与另一个张量 x 具有相同的大小。
        KV=torch.add(x,edge1)
        qf=self.sa(KV)
       # print('qf', qf.shape)
        ps1 = self.relu(qf)
       # print('ps1', ps1.shape)
        a = self.psi(ps1)
        a1=x*a
        a2=edge1*a
        #out=torch.cat((a1,a2),axis=1)
        out = self.AFF(a1, a2)
        #print('out', out.shape)
        out=self.conv1x1(out)
       # print('out', out.shape)
        #out=torch.add(out,x)
        return out
class GateFusion(nn.Module):
    def __init__(self, in_planes):
        super(GateFusion, self).__init__()

        self.gate_1 = nn.Conv2d(in_planes * 2, in_planes, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 2, in_planes, kernel_size=1, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        cat_fea = torch.cat((x1, x2), dim=1)

        att_vec_1 = self.gate_1(cat_fea)
        att_vec_2 = self.gate_2(cat_fea)

        att_vec_cat = torch.cat((att_vec_1, att_vec_2), dim=1)
        att_vec_soft = self.softmax(att_vec_cat)

        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2

        return x_fusion

class FullyAttentionalBlock(nn.Module):
    def __init__(self, plane, norm_layer=nn.BatchNorm2d):
        super(FullyAttentionalBlock, self).__init__()
        #self.conv1 = nn.Linear(plane, plane)  # 全连接层
        self.conv1 = nn.Conv2d(plane, plane, kernel_size=1)
        #self.conv2 = nn.Linear(plane, plane)
        self.conv2 = nn.Conv2d(plane, plane, kernel_size=1)
        self.conv = nn.Sequential(nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),
                                  norm_layer(plane),
                                  nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        feat_h = x.permute(0, 3, 1, 2).contiguous().view( -1,batch_size * width,
                                                         height)  # x.permute(0, 3, 1, 2)：这一步是对输入张量 x 的维度进行重新排列。permute 函数根据给定的顺序重新排列张量的维度。这里，(0, 3, 1, 2) 表示对 x 张量的维度进行重新排序，第一个维度保持不变，第二个维度变为原来的第四个维度，第三个维度变为原来的第二个维度，第四个维度变为原来的第三个维度
        # contiguous()：这是为了保证张量在内存中是连续的
        #print('feat_h',feat_h.shape)
        feat_w = x.permute(0, 2, 1, 3).contiguous().view( -1,batch_size * height, width)
        #a=F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(1, 2, 0).contiguous()
        #print('a',a.shape)
        encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(1, 2, 0).contiguous())

        encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(1, 2, 0).contiguous())
        #print('1',encode_h.repeat(1, 1, width).shape)
        energy_h = torch.matmul(feat_h, encode_h.repeat( 1, 1, width))  # 执行了矩阵乘法操作。feat_h 和 encode_h.repeat(width, 1, 1) 是两个张量，torch.matmul 函数用于进行矩阵相乘。feat_h 的形状为 (batch_size * width, -1, height)，repeat(width, 1, 1) 意味着将 encode_h 沿着第一个维度（宽度维度）重复 width 次，energy_h 的形状为 (batch_size * width, -1, -1)
        #print(energy_w.shape)


        energy_w = torch.matmul(feat_w, encode_w.repeat(1, 1, height))
        full_relation_h = self.softmax(energy_h)  # [b*w, c, c]
        full_relation_w = self.softmax(energy_w)

        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3,
                                                                                                    1)  # torch.bmm(full_relation_h, feat_h)：这是两个张量之间的批量矩阵乘法操作。
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)
        out = self.gamma * (full_aug_h + full_aug_w) + x
        out = self.conv(out)
        return out
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output
class CFP(nn.Module):
    def __init__(self, nIn, d, KSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)

        self.dconv_4_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d+1 , d+1 ), groups=nIn // 16, bn_acti=True)

        self.dconv_4_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d+1 , d+1 ), groups=nIn // 16, bn_acti=True)

        self.dconv_4_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d+1 , d +1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_2_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4+1 ), int(d / 4+1 )), groups=nIn // 16, bn_acti=True)

        self.dconv_2_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4+1 ), int(d / 4 +1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4+1 ), int(d / 4 +1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2+1 ), int(d / 2+1 )), groups=nIn // 16, bn_acti=True)

        self.dconv_3_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2+1 ), int(d / 2+1 )), groups=nIn // 16, bn_acti=True)

        self.dconv_3_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 +1), int(d / 2+1 )), groups=nIn // 16, bn_acti=True)

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)

        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)

        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)

        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)
        output_4 = torch.cat([o4_1, o4_2, o4_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1, ad2, ad3, ad4], 1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output
class DRF_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DRF_block, self).__init__()
        #self.mrf1 = MRF_1(512, 512)
        self.cara1=CFP(512,8)
        self.cara2 = CFP(512,8)
        self.cara3 = CFP(512,8)
        #self.mrf2 = MRF_1(512, 512)
        #self.mrf3 = MRF_1(512, 512)
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        #x1 = self.mrf1(x)
        x1 = self.cara1(x)
        x2 = x + x1

        #x3 = self.mrf2(x2)
        x3 = self.cara2(x2)
        x4 = x2 + x3

        #x5 = self.mrf3(x4)
        x5 = self.cara1(x4)
        #x5 = self.cara3(x4)
        x6 = self.conv1(x)
        out = x4 + x5 + x6
        #out = x4 + x5 + x6
        return out
class PolypPVT(nn.Module):
    def __init__(self, channel=32, norm_layer=nn.BatchNorm2d):
        super(PolypPVT, self).__init__()
        # self.auto=Auto_Encoder_Model()
        #self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        self.backbone = p2t_base()
        # self.backbone = transxnet_b()
        path = '/root/autodl-tmp/pretrained_pth/p2t_base.pth'  # /root/autodl-tmp/pretrained_pth/p2t_base.pth  /home/130user4/polyp/polyp seg/pretrained_pth/sem_fpn_p2t_b_ade20k_80k.pth
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.layer0 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(64), nn.ReLU())
        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)
        # self.conv = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
        #                           # nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        #                           norm_layer(64),
        #                           nn.ReLU())
        #self.conv = BasicConv2d(64, 64, 3, stride=1, padding=1)
        self.OSAR=Attention(
         64, num_heads=2, sr_ratio=1)
        #self.conv_out = nn.Conv2d(64, 64, kernel_size=1)
        # self.layer_edge0 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #                                  nn.BatchNorm2d(64), nn.ReLU())
        # self.atten_edge_2 = ChannelAttention(64)
        # self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear',
        #                        align_corners=True)  # scale_factor=2 表示将特征图的大小沿着两个维度放大两倍
        self.CFM = CFM(channel)
        #self.AFF = AFF(channel)
        self.low_fusion = GateFusion(64)
        # self.ca = ChannelAttention(64)
        # self.sa = SpatialAttention()
        self.fully = FullyAttentionalBlock(64)
        #self.fully1 = FullyAttentionalBlock(32)
        self.SAM = SAM()
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

        self.drf = DRF_block(ch_in=512, ch_out=512)
        #self.drf1= DRF1_block(ch_in=320,ch_out=320)
    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        #print(x3.shape)
        x4 = pvt[3]
        x4 = self.drf(x4)
        # CIM
        x1_1 = self.layer0(x1)
        x2_1 = self.layer1(x2)
        low_feature = self.low_fusion(x1_1, x2_1)
        # low_feature = self.ca(low_feature) * low_feature  # channel attention
        # cim_feature = self.sa(low_feature) * low_feature  # spatial attention
        cim_feature = self.fully(low_feature)
        # cim_feature=torch.add(cim_feature,low_feature)
        # print('cim_feature',cim_feature.shape)
        # print('low_feature', low_feature.shape)
       # edge_out0 = self.layer_edge0(cim_feature)
       #  edge_out0 = self.layer_edge0(cim_feature)  # 64*88
       #  edge_out1 = self.layer_edge0(edge_out0)  # 64*176
       #  edge_out2 = self.layer_edge0(edge_out1) # 64*352
        # print('edge_out2',edge_out2.shape)
        #etten_edge_2 = self.atten_edge_2(edge_out2)
        a = self.OSAR(cim_feature)
        # print(a.shape)
        # a=self.conv_out(a)
        # print('a1',a.shape)
        # CFM
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)
        #print('cfm',cfm_feature.shape)
        # SAM
        #a = edge_out0.mul(etten_edge_2)  # 行的是按元素（element-wise）的乘法操作
        T2 = self.Translayer2_0(a)
        T2 = self.down05(T2)
       # print('T2', T2.shape)
        sam_feature = self.SAM(cfm_feature, T2)  # t1,t2
        #sam_feature=torch.add(cfm_feature,T2)#消融SA实验
        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)

        prediction1_8 = F.interpolate(prediction1, scale_factor=8,
                                      mode='bilinear')  # P1#双线性插值将这两个预测结果放大了8倍，得到 prediction1_8 和 prediction2_8。通常这样的放大操作是为了将低分辨率的预测结果提升到与原始图像相同的分辨率
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  # P2

        return prediction1_8, prediction2_8


if __name__ == '__main__':
    model = PolypPVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
