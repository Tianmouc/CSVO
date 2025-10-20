mycuda = 'cuda:0'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4#, SimpleViT
from .extractor3d import Encoder3D
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops
import random
def shuffle(x1, x2):
    # 获取输入的形状，假设输入的 x1 和 x2 形状相同
    batch_size, channels, height, width = x1.size()
    
    # 随机选择 n/2 个通道进行互换
    num_channels_to_swap = channels // 2
    indices_to_swap = random.sample(range(channels), num_channels_to_swap)
    
    # 创建交换后的新张量
    x1_shuffled = x1.clone()
    x2_shuffled = x2.clone()
    
    # 进行通道交换
    x1_shuffled[:, indices_to_swap, :, :] = x2[:, indices_to_swap, :, :]
    x2_shuffled[:, indices_to_swap, :, :] = x1[:, indices_to_swap, :, :]
    
    return x1_shuffled, x2_shuffled
def shift(x):
    # 获取输入的形状
    batch_size, channels, height, width = x.size()
    
    # 随机选择移动方向
    direction = random.choice(['up', 'down', 'left', 'right'])
    
    # 创建一个零填充的张量，保持与输入相同的形状
    shifted_x = torch.zeros_like(x)
    
    if direction == 'up':
        # 向上移动1格
        shifted_x[:, :, :-1, :] = x[:, :, 1:, :]
    elif direction == 'down':
        # 向下移动1格
        shifted_x[:, :, 1:, :] = x[:, :, :-1, :]
    elif direction == 'left':
        # 向左移动1格
        shifted_x[:, :, :, :-1] = x[:, :, :, 1:]
    elif direction == 'right':
        # 向右移动1格
        shifted_x[:, :, :, 1:] = x[:, :, :, :-1]
    
    return shifted_x
class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusion, self).__init__()
        # 定义卷积层，使用 1x1 卷积保持空间维度不变
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x1, x2):
        # 第一步：concat 操作，沿着通道维度（dim=1）拼接
        x = torch.cat((x1, x2), dim=1)
        
        # 第二步：1x1 卷积操作，保持空间维度不变
        x = self.conv(x)
        
        return x
        
class DualModalityFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualModalityFusion, self).__init__()
        # 1x1卷积层 + BatchNorm
        self.conv1x1_mod1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1_mod1 = nn.BatchNorm2d(out_channels)
        
        self.conv1x1_mod2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1_mod2 = nn.BatchNorm2d(out_channels)
        
        # 3x3卷积层 + BatchNorm
        self.conv3x3_mod1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2_mod1 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_mod2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2_mod2 = nn.BatchNorm2d(out_channels)
        
        # 1x1卷积层 + BatchNorm (第二次)
        self.conv1x1_mod1_second = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3_mod1 = nn.BatchNorm2d(out_channels)
        
        self.conv1x1_mod2_second = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3_mod2 = nn.BatchNorm2d(out_channels)
        self.featurefuser = FeatureFusion(out_channels, out_channels)

    def forward(self, x1, x2):
        # 1x1卷积 + BatchNorm
        out1 = self.bn1_mod1(self.conv1x1_mod1(x1))
        out2 = self.bn1_mod2(self.conv1x1_mod2(x2))
        
        # Shuffle
        out1, out2 = shuffle(out1, out2)
        
        res1 = out1
        res2 = out2
        # Shift
        out1 = shift(out1)
        out2 = shift(out2)
        
        # 3x3卷积 + BatchNorm
        res1 = self.bn2_mod1(self.conv3x3_mod1(res1))
        res2 = self.bn2_mod2(self.conv3x3_mod2(res2))
        # Add操作
        out1 = out2 + res1
        out2 = out1 + res2

        # Shuffle
        out1, out2 = shuffle(out1, out2)
        
        # 1x1卷积 + BatchNorm (第二次)
        out1 = self.bn3_mod1(self.conv1x1_mod1_second(out1))
        out2 = self.bn3_mod2(self.conv1x1_mod2_second(out2))
        # Residual加法
        out1 = out1 + x1
        out2 = out2 + x2
        return out1, out2
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        
        Channel_score = self.channel_attention(x)
        out = Channel_score * x
        Att_score = self.spatial_attention(out)
        out = Att_score * out
        return out
        
class DualModalityFusionStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.fuse1 = DualModalityFusion(in_channels,in_channels)
        self.fuse2 = DualModalityFusion(in_channels,in_channels)
        self.fuse3 = DualModalityFusion(in_channels,in_channels)
        self.output_enhance = CBAM(in_channels * 2)
        self.output_fuse = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self,x1,x2):

        x11,x21 = self.fuse1(x1,x2)
        x12,x22 = self.fuse2(x11,x21)
        x13,x23 = self.fuse3(x12,x22)
        x = torch.cat([x13,x23],dim=1)
        x_cbam = self.output_enhance(x)
        x_output = self.output_fuse(x_cbam)

        return x_output
# 定义随机目标函数，用于计算损失
def random_target_function(x):
    # 假设目标是输出的形状相同的矩阵，值随机
    return torch.randn_like(x)

def rgb_or_sd(events):
    res = events[0]
    assert res in ["RGB", "SD"]
    return res, events[1:]

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt


DIM = 384

class MultiHeadModule(nn.Module):
    def __init__(self):
        super(MultiHeadModule, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(128, 126),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 384),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (1, 10, 128, 120, 160)
        batch_size, channels, depth, height, width = x.shape
        
        # Reshape to (batch_size * channels * depth * height, width)
        x_reshaped = x.view(-1, width)
        
        # Apply MLPs
        out1 = self.mlp1(x_reshaped)
        out2 = self.mlp2(x_reshaped)
        
        # Reshape back to original dimensions with new channel sizes
        out1 = out1.view(batch_size, channels, depth, height, -1)
        out2 = out2.view(batch_size, channels, depth, height, -1)
        
        return out1, out2

class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)

from torchvision import transforms
from PIL import Image
import os

def save_normalized_tensor_as_image(directory, tensor):
    # 将torch矩阵归一化到0-255
    tensor = tensor.squeeze(0).squeeze(-1).squeeze(-1) 
    tensor = tensor - tensor.min()  # 将最小值归零
    tensor = tensor / tensor.max()  # 归一化到[0, 1]
    tensor = tensor * 255  # 转换到[0, 255]
    tensor = tensor.byte()  # 转换为整数类型
    
    # 转换为PIL图像
    image = transforms.ToPILImage()(tensor)

    # 获取目录下所有png图片
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]

    if not png_files:
        file_number = -1
    else:
    
    # 获取最大六位数的文件名并递增
        max_file_name = max(png_files, key=lambda x: int(x.split('.')[0]))
        file_number = int(max_file_name.split('.')[0])  # 解析六位数
    new_file_name = f"{file_number + 1:06d}.png"  # 保持六位数格式

    # 保存图片
    image.save(os.path.join(directory, new_file_name))
    print(f"Image saved as {new_file_name}")

class Patchifier(nn.Module):
    def __init__(self, model_path, patch_size=3, ablation='RGB', use_pretrained_sd=False):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(input_dim=3, output_dim=128, norm_fn='instance')
        
        if ablation != 'dpvo':
            if not 'td' in ablation:
                self.fnetSD = BasicEncoder4(input_dim=2, output_dim=128, norm_fn='instance')
                self.fnetSD2 = BasicEncoder4(input_dim=2, output_dim=128, norm_fn='instance')
            else:
                self.fnetSD = BasicEncoder4(input_dim=1, output_dim=128, norm_fn='instance')
                self.fnetSD2 = BasicEncoder4(input_dim=1, output_dim=128, norm_fn='instance')

        self.inet = BasicEncoder4(input_dim=3, output_dim=DIM, norm_fn='none')
        if ablation != 'dpvo':
            if 'td' not in ablation:
                self.inetSD = BasicEncoder4(input_dim=2, output_dim=DIM, norm_fn='none')
                self.inetSD2 = BasicEncoder4(input_dim=2, output_dim=DIM, norm_fn='none')
            else:
                self.inetSD = BasicEncoder4(input_dim=1, output_dim=DIM, norm_fn='none')
                self.inetSD2 = BasicEncoder4(input_dim=1, output_dim=DIM, norm_fn='none')

        self.ablation = ablation
        
        if ablation != 'dpvo':
            self.fuserf = DualModalityFusion(128,128)
            self.fuseri = DualModalityFusion(DIM, DIM)

        if ablation == 'dpvo':
            
            model_data = torch.load(model_path, map_location=torch.device('cpu'))
            filtered_state_dict = {k.replace("module.patchify.", ''): v for k, v in model_data.items() if 'patchify.fnet' in k}
            self.fnet.load_state_dict(filtered_state_dict, strict=False)
            filtered_state_dict = {k.replace("module.patchify.", ''): v for k, v in model_data.items() if 'patchify.inet' in k}
            self.inet.load_state_dict(filtered_state_dict, strict=False)

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g
        
    def __SD_sobel(self, sdx, sdy):
        dx = sdx[:,:,0,:-1,:-1]
        dy = sdy[:,:,0,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g
        
    def forward(self, images, patches_per_image=80, disps=None, gradient_bias=False, return_color=False, time_surface0=None, time_surface1=None, sdr=None, sdl = None, events=None, rank=None, isTianmouc=None, augmented=False):
        """ extract patches from input images """
        if rank!=None:
            mycuda = rank
        else:
            mycuda = 'cuda:0'
        #端到端训练时，如果不插帧，应该使用的ablation=rgb_taught
        n = images.shape[1]
        if 'async' in self.ablation.lower() or 'async_td' in self.ablation.lower() :
            zero_index = []
            for i in range(images.shape[1]):
                if (torch.all(images[0, i] == -0.5) and isTianmouc) or (random.randint(0,5) == 1 and not isTianmouc):
                    # print("SD Only")
                    zero_index.append(i)
            if len(zero_index) == n:
                if 'td' in self.ablation.lower():
                    sds = sdr
                else:
                    sds = torch.cat((sdr, sdl), dim=2)
                fmap = self.fnetSD2(sds) / 4.0
            elif len(zero_index) == 0:
                image_map = self.fnet(images) / 4.0
                if 'td' in self.ablation.lower():
                    sds = sdr
                else:
                    sds = torch.cat((sdr, sdl), dim=2)
                sdmap1 = self.fnetSD(sds) / 4.0
                fmap1, _ = self.fuserf(image_map[0], sdmap1[0])
                fmap = fmap1.unsqueeze(0)
            else:
                index_array = torch.tensor(zero_index, device=mycuda)
                images_nozero = torch.cat([images[0][i].unsqueeze(0) for i in range(n) if i not in index_array], dim=0).unsqueeze(0)
                image_map = self.fnet(images_nozero) / 4.0
                if 'td' in self.ablation.lower():
                    sds = sdr
                else:
                    sds = torch.cat((sdr, sdl), dim=2)
                sds_nozero = torch.cat([sds[0][i].unsqueeze(0) for i in range(n) if i not in index_array], dim=0).unsqueeze(0)
                sdmap1 = self.fnetSD(sds_nozero) / 4.0
                fmap1, _ = self.fuserf(image_map[0], sdmap1[0])
                fmap1 = fmap1.unsqueeze(0)
                mask = torch.tensor([i in index_array for i in range(n)])
                mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # shape变为(1, n, 1, 1, 1)

                # 使用布尔索引选择未被选中的向量
                sds_zero = torch.cat([sds[0][i].unsqueeze(0) for i in range(n) if i in index_array], dim=0).unsqueeze(0)
                fmap2 = self.fnetSD2(sds_zero) / 4.0

                fmap = torch.zeros_like(torch.cat((fmap1, fmap2), dim=1))
                count1 = 0
                count2 = 0
                for i in range(n):
                    if i in zero_index:
                        fmap[0][i] = fmap2[0][count2]
                        count2 += 1
                    else:
                        fmap[0][i] = fmap1[0][count1]
                        count1 +=1

            if len(zero_index) == n:
                if 'td' in self.ablation.lower():
                    sds = sdr
                else:
                    sds = torch.cat((sdr, sdl), dim=2)
                imap = self.inetSD2(sds) / 4.0
            elif len(zero_index) == 0:
                image_map = self.inet(images) / 4.0
                if 'td' in self.ablation.lower():
                    sds = sdr
                else:
                    sds = torch.cat((sdr, sdl), dim=2)
                sdmap1 = self.inetSD(sds) / 4.0
                imap1, _ = self.fuseri(image_map[0], sdmap1[0])
                imap = imap1.unsqueeze(0)
            else:
                index_array = torch.tensor(zero_index, device=mycuda)
                images_nozero = torch.cat([images[0][i].unsqueeze(0) for i in range(n) if i not in index_array], dim=0).unsqueeze(0)
                image_map = self.inet(images_nozero) / 4.0
                if 'td' in self.ablation.lower():
                    sds = sdr
                else:
                    sds = torch.cat((sdr, sdl), dim=2)
                sds_nozero = torch.cat([sds[0][i].unsqueeze(0) for i in range(n) if i not in index_array], dim=0).unsqueeze(0)
                sdmap1 = self.inetSD(sds_nozero) / 4.0
                imap1, _ = self.fuseri(image_map[0], sdmap1[0])
                imap1 = imap1.unsqueeze(0)
                mask = torch.tensor([i in index_array for i in range(n)])

                # 使用这个掩码的补集来选择那些未被选中的向量
                # 首先将掩码扩展到与input_tensor相同的维度
                mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # shape变为(1, n, 1, 1, 1)

                # 使用布尔索引选择未被选中的向量
                sds_zero = torch.cat([sds[0][i].unsqueeze(0) for i in range(n) if i in index_array], dim=0).unsqueeze(0)
                imap2 = self.inetSD2(sds_zero) / 4.0

                imap = torch.zeros_like(torch.cat((imap1, imap2), dim=1))
                count1 = 0
                count2 = 0
                for i in range(n):
                    if i in zero_index:
                        imap[0][i] = imap2[0][count2]
                        count2 += 1
                    else:
                        imap[0][i] = imap1[0][count1]
                        count1 +=1
            
        elif "sd_only" == self.ablation.lower():
            sds = torch.cat((sdr, sdl), dim=2) 
            fmap = self.fnetSD2(sds) / 4.0
            imap = self.inetSD2(sds) / 4.0 
        elif 'td_only' == self.ablation.lower():
            fmap = self.fnetSD2(sdr) / 4.0
            imap = self.inetSD2(sdr) / 4.0
        elif "dpvo" == self.ablation.lower():
            fmap = self.fnet(images) / 4.0
            imap = self.inet(images) / 4.0
        elif self.ablation.lower() == "sync":
            imagemap = self.fnet(images) / 4.0
            sds = torch.cat((sdr, sdl), dim=2)
            sdmap = self.fnetSD(sds) / 4.0
            fmap, _ = self.fuserf(imagemap[0], sdmap[0])
            fmap = fmap.unsqueeze(0)
            imagemap = self.inet(images) / 4.0
            sds = torch.cat((sdr, sdl), dim=2)
            sdmap = self.inetSD(sds) / 4.0
            imap, _ = self.fuseri(imagemap[0], sdmap[0])
            imap = imap.unsqueeze(0)

        else:
            raise ValueError("Ablation not permited")
        b, n, c, h, w = fmap.shape
        P = self.patch_size
        if self.ablation == 'async_td' or self.ablation == 'dpvo':
            gradient_bias = False
        if gradient_bias:
            if self.ablation == 'async' or self.ablation == 'sd_only' or self.ablation == 'async_td':
                g = self.__SD_sobel(sdr,sdl)
            elif self.ablation == 'dpvo':
                g = self.__image_gradient(images)
            else:
                raise ValueError("Grad Computation not permited")
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device=mycuda)
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device=mycuda)

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(net=g[0,:,None], coords=coords, radius=0, rank=rank).view(n, 3 * patches_per_image)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        else:
            x = torch.randint(1, w-1, size=[n, patches_per_image], device=mycuda)
            y = torch.randint(1, h-1, size=[n, patches_per_image], device=mycuda)
        coords = torch.stack([x, y], dim=-1).float().cuda()

        imap = altcorr.patchify(net=imap[0], coords=coords, radius=0, rank=rank).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(net=fmap[0], coords=coords, radius=P//2, rank=rank).view(b, -1, 128, P, P)
        if return_color:
            clr = altcorr.patchify(net=images[0], coords=4*(coords + 0.5), radius=0, rank=rank).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device=mycuda)
        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(net=grid[0], coords=coords, radius=P//2, rank=rank).view(b, -1, 3, P, P)
        index = torch.arange(n, device=mycuda).view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)
        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False, ablation='RGB', use_pretrained_sd=True, augmented=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P, ablation=ablation, use_pretrained_sd=use_pretrained_sd)
        self.update = Update(self.P)
        
        if use_pretrained_sd:
            model_data = torch.load('dpvo.pth', map_location=torch.device('cpu'))
            filtered_state_dict = {k.replace("module.update.", ''): v for k, v in model_data.items() if 'module.update' in k}
            self.update.load_state_dict(filtered_state_dict, strict=False)
            for param in self.update.parameters():
                param.requires_grad = False
            print("Use pretrained update module")
        self.DIM = DIM
        self.RES = 4
        self.augmented = augmented


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False, timeSurface0 = None, timeSurface1 = None, sdr=None, sdl=None, events=None, rank=None, isTianmouc=False):
        """ Estimates SE3 or Sim3 between pair of frames """
        if rank!=None:
            mycuda=rank
        else:
            mycuda='cuda:0'

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps, time_surface0 = timeSurface0, time_surface1 = timeSurface1, sdr=sdr, sdl=sdl, events=events, rank=rank, isTianmouc=isTianmouc, augmented=self.augmented, gradient_bias=True)
        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device=mycuda))
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device=mycuda, dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device=mycuda))
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device=mycuda))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device=mycuda)
                net = torch.cat([net1, net], dim=1)
                temp = np.random.rand()
                if temp < 0.1:
                    
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)
            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj

