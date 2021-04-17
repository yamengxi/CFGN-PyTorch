from math import gcd

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import common


def make_model(args, parent=False):
    return RFDN(args)


def generate_masks(num):
    masks = []
    for i in range(num):
        now = list(range(2 ** num))
        length = 2 ** (num - i)
        for j in range(2 ** i):
            tmp = now[j*length:j*length+length//2]
            now[j*length:j*length+length//2] = now[j*length+length//2:j*length+length]
            now[j*length+length//2:j*length+length] = tmp
        masks.append(now)
    return torch.tensor(masks)


class ButterflyConv_v1(nn.Module):
    def __init__(self, in_channels, act, out_channels, dilation=1):
        super(ButterflyConv_v1, self).__init__()

        min_channels = min(in_channels, out_channels)
        assert (min_channels & (min_channels - 1)) == 0 # Is min_channels = 2^n?

        if in_channels == out_channels:
            self.head = nn.Identity()
            self.tail = nn.Identity()
        elif in_channels > out_channels:
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=gcd(in_channels, out_channels)),
                act()
            )
            self.tail = nn.Identity()
        elif in_channels < out_channels:
            self.head = nn.Identity()
            self.tail = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=gcd(in_channels, out_channels)),
                act()
            )
        else:
            raise NotImplementedError("")

        self.num_butterflies = 0
        for i in range(10000):
            if 2 ** i == min_channels:
                self.num_butterflies = i
                break
        self.masks = generate_masks(self.num_butterflies)

        self.conv_acts = []
        for i in range(self.num_butterflies * 2):
            self.conv_acts.append(
                nn.Sequential(nn.Conv2d(min_channels, min_channels, 3, 1, dilation, dilation, groups=min_channels), act())
            )
        self.conv_acts = nn.Sequential(*self.conv_acts)

    def forward(self, x):
        self.masks = self.masks.to(x.device)
        x = self.head(x)

        now = x
        for i in range(self.num_butterflies):
            now = self.conv_acts[i*2](now) + self.conv_acts[i*2+1](torch.index_select(now, 1, self.masks[i]))
        now = now + x

        now = self.tail(now)
        return now


class SRB(nn.Module):
    def __init__(self, in_channels, act, *args):
        super(SRB, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.act = act()
    
    def forward(self, x):
        out = self.conv3x3(x) + x
        out = self.act(out)
        return out


class MainBlock(nn.Module):
    def __init__(self, in_channels, act, basic_module):
        super(MainBlock, self).__init__()
        self.steps = 3
        self.convs = []
        for i in range(self.steps):
            self.convs.append(nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0))
        self.convs = nn.Sequential(*self.convs)

        self.basic_modules = []
        for i in range(self.steps):
            self.basic_modules.append(basic_module(in_channels, act, in_channels))
        self.basic_modules = nn.Sequential(*self.basic_modules)

        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)

        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0)

        self.act = act()

    def forward(self, x):
        now = x
        features = []
        for i in range(self.steps):
            features.append(self.convs[i](now))
            now = self.basic_modules[i](now)
        now = self.conv3x3(now)
        features.append(now)
        features = torch.cat(features, 1)
        out = self.conv1x1(features)
        out = self.act(out)
        return out + x


class RFDN(nn.Module):
    """RFDN network structure.

    Args:
        args.scale (list[int]): Upsampling scale for the input image.
        args.n_colors (int): Channels of the input image.
        args.n_feats (int): Channels of the mid layer.
        args.n_resblocks (int): Number of main blocks.
        args.act (str): Activate function used in BFN. Default: nn.PReLU.
        args.rgb_range: .
        args.main_block_version:
        args.butterfly_conv_version:
        args.skip_connection (bool):.
    """
    def __init__(self, args):
        super(RFDN, self).__init__()
        assert len(args.scale) == 1
        scale = args.scale[0]
        n_colors = args.n_colors
        n_feats = args.n_feats
        n_resblocks = args.n_resblocks
        if args.act == 'relu':
            act = nn.ReLU
        elif args.act == 'lrelu':
            act = nn.LeakyReLU
        elif args.act == 'prelu':
            act = nn.PReLU
        else:
            raise NotImplementedError("")
        
        if args.basic_module_version == 'v1':
            basic_module = SRB
        elif args.basic_module_version == 'v2':
            basic_module = ButterflyConv_v1
        else:
            raise NotImplementedError("")

        rgb_range = args.rgb_range

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)

        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        self.main_blocks = []
        for i in range(n_resblocks):
            self.main_blocks.append(MainBlock(n_feats, act, basic_module))
        self.main_blocks = nn.Sequential(*self.main_blocks)

        self.features_fusion_module = nn.Sequential(
            nn.Conv2d(n_feats * n_resblocks, n_feats, 1, 1, 0),
            act()
        )

        self.final_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale * scale), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        now = x
        outs = []
        for main_block in self.main_blocks:
            now = main_block(now)
            outs.append(now)

        out = torch.cat(outs, 1)
        out = self.features_fusion_module(out)
        out = self.final_conv(out) + x

        out = self.upsampler(out)
        out = self.add_mean(out)

        return out


if __name__ == '__main__':
    # test network
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    import argparse
    args = argparse.Namespace()
    args.scale = [2]
    args.patch_size = 256
    args.n_colors = 3
    args.n_feats = 48
    args.n_resblocks = 6
    args.act = 'lrelu'
    args.rgb_range = 255
    args.basic_module_version = 'v1'

    # args.scale = [2]
    # args.patch_size = 256
    # args.n_colors = 3
    # args.n_feats = 64
    # args.n_resblocks = 6
    # args.act = 'lrelu'
    # args.rgb_range = 255
    # args.basic_module_version = 'v2'

    # import pdb
    # pdb.set_trace()
    model = RFDN(args)
    model.eval()

    from torchsummaryX import summary
    x = summary(model.cuda(), torch.zeros((1, 3, 720 // 4, 1280 // 4)).cuda())

    # from torchsummary import summary
    # summary(model.cuda(), input_size=(3, 720 // 4, 1280 // 4), batch_size=1)
