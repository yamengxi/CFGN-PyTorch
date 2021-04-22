from math import gcd

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import common


def make_model(args, parent=False):
    return CFGN(args)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'identity':
        return nn.Identity()
    elif act_type == 'relu':
        return nn.ReLU(inplace)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))


class SRB(nn.Module):
    def __init__(self, in_channels):
        super(SRB, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.act = activation('lrelu')

    def forward(self, x):
        out = self.conv3x3(x) + x
        out = self.act(out)
        return out


class CFGM(nn.Module):
    def __init__(self, in_channels):
        super(CFGM, self).__init__()

        self.num_conv = 0
        for i in range(10000):
            if 2 ** i >= in_channels:
                self.num_conv = i
                break

        self.conv_acts = []
        for i in range(self.num_conv * 2):
            if i % 2 == 0:
                self.conv_acts.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, 3, 1, 1, 1, groups=in_channels),
                        activation('prelu', n_prelu=in_channels)
                    )
                )
            else:
                self.conv_acts.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, 3, 1, 3, 3, groups=in_channels),
                        activation('prelu', n_prelu=in_channels)
                    )
                )
        self.conv_acts = nn.Sequential(*self.conv_acts)


    def forward(self, x):
        out = x
        for i in range(self.num_conv):
            out = self.conv_acts[i*2](out) + self.conv_acts[i*2+1](out)
        return out + x


def make_block(in_channels, block_type):
    block_type = block_type.lower()
    if block_type == 'base' or block_type == 'srb':
        return SRB(in_channels)
    elif block_type == 'cfgm':
        return CFGM(in_channels)
    else:
        raise NotImplementedError('block [{:s}] is not found'.format(block_type))


class MainBlock(nn.Module):
    def __init__(self, in_channels, act, block_type):
        super(MainBlock, self).__init__()

        self.num = 3

        self.blocks = [
            make_block(in_channels, block_type) for _ in range(self.num)
        ]
        self.blocks = nn.Sequential(*self.blocks)

        self.conv1x1s = [
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0),
                activation(act)
            ) for _ in range(self.num)
        ]
        self.conv1x1s = nn.Sequential(*self.conv1x1s)

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            activation(act)
        )

        self.conv1x1_act = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0),
            activation('lrelu')
        )
    
    def forward(self, x):
        now = x
        features = []
        for i in range(self.num):
            features.append(self.conv1x1s[i](now))
            now = self.blocks[i](now)
        features.append(self.conv3x3(now))
        features = torch.cat(features, 1)
        out = self.conv1x1_act(features)
        return out + x


class CFGN(nn.Module):
    """CFGN network structure.

    Args:
        args.scale (list[int]): Upsampling scale for the input image.
        args.n_colors (int): Channels of the input image.
        args.n_feats (int): Channels of the mid layer.
        args.n_resgroups (int): Number of context feature guided groups.
        args.act (str): Activate function used in network.
        args.rgb_range: 255.
        args.block_type: Block used in network, this option is used for ablation study.
    """
    def __init__(self, args):
        super(CFGN, self).__init__()
        assert len(args.scale) == 1
        scale = args.scale[0]
        n_colors = args.n_colors
        n_feats = args.n_feats
        n_resgroups = args.n_resgroups
        act = args.act
        rgb_range = args.rgb_range
        block_type = args.block_type

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)

        self.head_conv = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        self.head_act = activation('lrelu')

        self.body = []
        for i in range(n_resgroups):
            self.body.append(MainBlock(n_feats, act, block_type))
        self.body = nn.Sequential(*self.body)

        self.features_fusion_module = nn.Sequential(
            nn.Conv2d(n_feats * n_resgroups, n_feats, 1, 1, 0),
            activation('lrelu'),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            activation('lrelu')
        )

        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale * scale), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head_conv(x)

        now = self.head_act(x)
        outs = []
        for main_block in self.body:
            now = main_block(now)
            outs.append(now)

        outs = torch.cat(outs, 1)
        y = self.features_fusion_module(outs) + x

        y = self.upsampler(y)
        y = self.add_mean(y)

        return y


if __name__ == '__main__':
    # test network
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    import argparse
    args = argparse.Namespace()
    # args.scale = [2]
    # args.patch_size = 128
    # args.n_colors = 3
    # args.n_feats = 64
    # args.n_resgroups = 8
    # args.act = 'lrelu'
    # args.rgb_range = 255
    # args.block_type = 'cfgg'

    args.scale = [2]
    args.patch_size = 128
    args.n_colors = 3
    args.n_feats = 48
    args.n_resgroups = 6
    args.act = 'identity'
    args.rgb_range = 255
    args.block_type = 'base'

    # import pdb
    # pdb.set_trace()
    model = CFGN(args)
    model.eval()

    from torchsummaryX import summary
    x = summary(model.cuda(), torch.zeros((1, 3, 720 // args.scale[0], 1280 // args.scale[0])).cuda())

    # from torchsummary import summary
    # summary(model.cuda(), input_size=(3, 720 // 4, 1280 // 4), batch_size=1)
