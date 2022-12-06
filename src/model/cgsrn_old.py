import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import common

def make_model(args, parent=False):
    return CGSRN(args)

class GlobalContextExtractor(nn.Module):
    def __init__(self, channel, reduction=16, act=nn.ReLU):
        super(GlobalContextExtractor, self).__init__()
        assert reduction >= 1 and channel >= reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction), act(),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        num_batch, num_channel = x.size()[:2]
        y = self.avg_pool(x).view(num_batch, num_channel)
        y = self.fc(y).view(num_batch, num_channel, 1, 1)
        return x * y


class ContextGuidedModule(nn.Module):
    def __init__(self, in_channels=128, kernel_size=5, dilation=3, reduction=16, act=nn.ReLU, bn=nn.Identity):
        super(ContextGuidedModule, self).__init__()
        mid_channels = in_channels // 2
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, 1, 0), bn(mid_channels), act())
        self.f_loc = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, (kernel_size - 1) // 2)
        self.f_sur = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, (kernel_size - 1) // 2 * dilation, dilation)
        self.act = act()
        self.f_glo = GlobalContextExtractor(in_channels, reduction, act)
    
    def forward(self, x):
        x0 = self.conv1x1(x)
        loc = self.f_loc(x0)
        sur = self.f_sur(x0)
        joi_feat = torch.cat([loc, sur], 1)  # the joint feature
        joi_feat = self.act(joi_feat)
        out = self.f_glo(joi_feat)
        return out + x

class PatchBasedNonlocalModule(nn.Module):
    def __init__(self, num_convs=3, in_channels=128, kernel_size=3, act=nn.ReLU, bn=nn.Identity, num_patches=4):
        super(PatchBasedNonlocalModule, self).__init__()
        self.num_patches = num_patches

        mid_channels = in_channels // 2
        self.pre_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size, 1, (kernel_size - 1) // 2), bn(mid_channels), act())

        self.convs1 = []
        self.convs2 = []
        for i in range(num_convs):
            self.convs1.extend([nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, (kernel_size - 1) // 2), bn(mid_channels), act()])
            self.convs2.extend([nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, (kernel_size - 1) // 2), bn(mid_channels), act()])
        self.convs1 = nn.Sequential(*self.convs1)
        self.convs2 = nn.Sequential(*self.convs2)

        self.patch_based_convs1 = []
        self.patch_based_convs2 = []
        for i in range(num_patches*num_patches):
            self.patch_based_convs1.append(nn.Conv2d(mid_channels, mid_channels, 1, 1, 0))
            self.patch_based_convs2.append(nn.Conv2d(mid_channels, mid_channels, 1, 1, 0))
        self.patch_based_convs1 = nn.Sequential(*self.patch_based_convs1)
        self.patch_based_convs2 = nn.Sequential(*self.patch_based_convs2)

        self.final_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, 1, (kernel_size - 1) // 2), bn(mid_channels), act())

    def forward(self, x):
        h, w = x.shape[-2:]
        assert h % self.num_patches == 0 and w % self.num_patches == 0
        h_size = h // self.num_patches
        w_size = w // self.num_patches

        x0 = self.pre_conv(x)

        x1 = self.convs1(x0) + x0
        x2 = self.convs2(x0) + x0

        for i in range(self.num_patches):
            for j in range(self.num_patches):
                x1[:, :, h_size*i:h_size*(i+1), w_size*j:w_size*(j+1)] = self.patch_based_convs1[i*self.num_patches+j](x1[:, :, h_size*i:h_size*(i+1), w_size*j:w_size*(j+1)])
                x2[:, :, h_size*i:h_size*(i+1), w_size*j:w_size*(j+1)] = self.patch_based_convs2[i*self.num_patches+j](x2[:, :, h_size*i:h_size*(i+1), w_size*j:w_size*(j+1)])

        out = []
        for b in range(x0.size(0)):
            h_results = []
            for i in range(self.num_patches):
                w_results = []
                for j in range(self.num_patches):
                    w = []
                    for k in range(self.num_patches):
                        for l in range(self.num_patches):
                            if i == k and j == l:
                                w.append(torch.ones_like(x1[0, 0, 0, 0]) * 100000000)
                            else:
                                w.append((x1[b, :, h_size*i:h_size*(i+1), w_size*j:w_size*(j+1)] - x2[b, :, h_size*k:h_size*(k+1), w_size*l:w_size*(l+1)]).pow(2).sum())
                    w = torch.stack(w)
                    w = F.softmax(torch.exp(-w))
                    tmp = torch.zeros_like(x1[b, :, :h_size, :w_size])
                    for k in range(self.num_patches):
                        for l in range(self.num_patches):
                            tmp += w[k*self.num_patches+l] * x2[b, :, h_size*k:h_size*(k+1), w_size*l:w_size*(l+1)]
                    w_results.append(tmp)
                w_results = torch.cat(w_results, 2)
                h_results.append(w_results)
            h_results = torch.cat(h_results, 1)
            out.append(h_results)

        out = torch.stack(out)
        out = torch.cat((out, x0), dim=1)
        out = self.final_conv(out)

        return out + x


class JointModule(nn.Module):
    def __init__(self, in_channels=128, act=nn.ReLU):
        super(JointModule, self).__init__()
        self.context_guided_modules = []
        for i in range(4):
            self.context_guided_modules.append(ContextGuidedModule(in_channels=in_channels, act=act))
        self.context_guided_modules = nn.Sequential(*self.context_guided_modules)
        self.final_conv = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 3, 1, 1), act())

    def forward(self, x):
        out = self.context_guided_modules(x)
        out = torch.cat([x, out], 1)
        out = self.final_conv(out)
        return out


class MainBlock(nn.Module):
    def __init__(self, num_blocks=3, in_channels=128, up_scale=2, act=nn.ReLU):
        super(MainBlock, self).__init__()
        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(JointModule(in_channels, act))
        self.blocks.append(PatchBasedNonlocalModule(in_channels=in_channels, act=act))
        self.blocks = nn.Sequential(*self.blocks)
        self.final_conv = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 3, 1, 1), act())
        self.upsampler = nn.Identity() if up_scale <= 1 else common.Upsampler(common.default_conv, up_scale, in_channels)

    def forward(self, x):
        out = self.blocks(x)
        out = torch.cat([x, out], 1)
        out = self.final_conv(out) + x
        out = self.upsampler(out)
        return out


class HGDModule(nn.Module):
    def __init__(self, in_channels, center_channels, out_channels, norm_layer=None, act=nn.ReLU):
        super(HGDModule, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.out_channels = out_channels
        self.conv_cat= nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels, 1, bias=False),
            norm_layer(out_channels),
            act())
        self.conv_center= nn.Sequential(
            nn.Conv2d(in_channels*3, center_channels, 1, bias=False),
            #norm_layer(out_channels),
            #act(),
            #nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            act(),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels))
        self.norm_center= nn.Sequential(
            nn.Softmax(2))
        self.conv_affinity0 = nn.Sequential(
            nn.Conv2d(in_channels*3, out_channels , 1, bias=False),
            norm_layer(out_channels),
            act())
        self.conv_affinity1 = nn.Sequential(
            nn.Conv2d(out_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            act(),
            nn.Conv2d(center_channels, center_channels, 1, bias=False),
            norm_layer(center_channels),
            act())
        self.conv_up = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            act())
        self.avgpool0 = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        b, c, h_down, w_down = x[0].size()
        b, c, h_up, w_up = x[-1].size()

        x_down = []
        x_up = []
        for item in x:
            if item.shape[-2] == h_down and item.shape[-1] == w_down:
                x_down.append(item)
            else:
                x_down.append(F.interpolate(item, size=(h_down, w_down), mode='bilinear'))
            if item.shape[-2] == h_up and item.shape[-1] == w_up:
                x_up.append(item)
            else:
                x_up.append(F.interpolate(item, size=(h_up, w_up), mode='bilinear'))
        x_down = torch.cat(x_down, 1)
        x_up = torch.cat(x_up, 1)


        f_cat = self.conv_cat(x_down)
        f_cat = f_cat.view(b, self.out_channels, h_down*w_down)
        #f_x = x_cat.view(n, 2*c, h*w)
        f_center = self.conv_center(x_down)
        f_center_norm = f_center.view(b, self.center_channels, h_down*w_down)
        f_center_norm = self.norm_center(f_center_norm)
        #n x * in_channels x center_channels
        x_center = f_cat.bmm(f_center_norm.transpose(1, 2))
        

        ########################################
        f_cat = f_cat.view(b, self.out_channels, h_down, w_down)
        f_cat_avg = self.avgpool0(f_cat)
        value_avg = f_cat_avg.repeat(1, 1, h_up, w_up)

        ###################################
        #f_affinity = self.conv_affinity(guide_cat)
        guide_cat_conv = self.conv_affinity0(x_up)
        guide_cat_value_avg = guide_cat_conv + value_avg
        f_affinity = self.conv_affinity1(guide_cat_value_avg)
        n_aff, c_ff, h_aff, w_aff = f_affinity.size()
        f_affinity = f_affinity.view(n_aff, c_ff, h_aff * w_aff)
        norm_aff = ((self.center_channels) ** -.5)
        #x_up = norm_aff * x_center.bmm(f_affinity.transpose(1, 2))
        x_up = norm_aff * x_center.bmm(f_affinity)
        x_up = x_up.view(b, self.out_channels, h_aff, w_aff)
        x_up_cat = torch.cat([x_up, guide_cat_conv], 1)
        x_up_conv = self.conv_up(x_up_cat)
        return x_up_conv


class CGSRN(nn.Module):
    """CGSRN network structure.

    Args:
        args.scale (list[int]): Upsampling scale for the input image.
        args.n_colors (int): Channels of the input image.
        args.n_feats (int): Channels of the mid layer.
        args.n_resblocks (int): 
        act: Activate function used in CGSRN. Default: nn.PReLU.
    """
    def __init__(self, args, act=nn.PReLU):
        super(CGSRN, self).__init__()
        self.args = copy.deepcopy(args)
        assert len(self.args.scale) == 1
        self.args.scale = self.args.scale[0]

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        self.head = nn.Conv2d(args.n_colors, args.n_feats, 3, 1, 1)

        self.main_blocks = []
        if args.direct_up:
            self.main_blocks.append(MainBlock(num_blocks=args.n_resblocks, in_channels=args.n_feats, up_scale=1, act=act))
            self.main_blocks.append(MainBlock(num_blocks=args.n_resblocks, in_channels=args.n_feats, up_scale=4, act=act))
        else:
            for i in range(args.n_resgroups):
                self.main_blocks.append(MainBlock(num_blocks=args.n_resblocks, in_channels=args.n_feats, act=act))
        self.main_blocks = nn.Sequential(*self.main_blocks)

        self.HGDModule = HGDModule(args.n_feats, args.n_feats*4, args.n_feats, nn.Identity, act=act)

        self.tail = nn.Conv2d(args.n_feats, args.n_colors, 3, 1, 1)

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # # debug
        # conv = common.default_conv
        # kernel_size = 3
        # act = act(True)

        # m_body = [
        #     common.ResBlock(
        #         conv, args.n_feats, kernel_size, act=act, res_scale=args.res_scale
        #     ) for _ in range(args.n_resblocks)
        # ]
        # m_body.append(conv(args.n_feats, args.n_feats, kernel_size))
        # self.body = nn.Sequential(*m_body)

        # self.upsampler = common.Upsampler(conv, 4, args.n_feats, act=False)


        # self.ContextGuidedModule = ContextGuidedModule(args.n_feats, dilation=1)
        # self.JointModule = JointModule(args.n_feats)
        # self.MainBlock1 = MainBlock()
        # self.MainBlock2 = MainBlock()

    def forward(self, x):

        def _forward(x):
            x = self.sub_mean(x)
            x = self.head(x)

            outs = [x]
            for main_block in self.main_blocks:
                now = main_block(x)
                x = now + x if now.size() == x.size() else now # skip connection
                outs.append(x)
            # final_out = outs[-1]

            final_out = self.HGDModule(outs) + outs[-1]

            # # debug
            # x = self.body(x)
            # x = self.ContextGuidedModule(x)
            # x = self.JointModule(x)
            # final_out = self.upsampler(x)
            # final_out = self.MainBlock1(x)
            # final_out = self.MainBlock2(final_out)

            final_out = self.tail(final_out)
            final_out = self.add_mean(final_out)

            return final_out

        if self.training:
            return _forward(x)
        else:
            # import pdb
            # pdb.set_trace()
            b, c, h, w = x.size()
            H = h * self.args.scale
            W = w * self.args.scale

            h_stride = self.args.patch_size // self.args.scale if h >= self.args.patch_size // self.args.scale else h // 4 * 4
            w_stride = self.args.patch_size // self.args.scale if w >= self.args.patch_size // self.args.scale else w // 4 * 4
            h_crop, w_crop = h_stride, w_stride

            h_grids = max(h - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w - w_crop + w_stride - 1, 0) // w_stride + 1
            out = x.new_zeros((b, c, H, W))
            count_mat = x.new_zeros((1, 1, H, W))
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h)
                    x2 = min(x1 + w_crop, w)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_x = x[:, :, y1:y2, x1:x2]
                    crop_out = _forward(crop_x)
                    out += F.pad(crop_out, (int(x1)*self.args.scale, int(w - x2)*self.args.scale, int(y1)*self.args.scale, int(h - y2)*self.args.scale))
                    count_mat[:, :, y1*self.args.scale:y2*self.args.scale, x1*self.args.scale:x2*self.args.scale] += 1

            assert (count_mat == 0).sum() == 0

            return out / count_mat


if __name__ == '__main__':
    # test CGSRN
    import argparse
    args = argparse.Namespace()
    args.rgb_range = 255
    args.n_colors = 3
    args.n_feats = 128
    args.n_resblocks = 5
    args.n_resgroups = 2
    args.direct_up = False
    args.patch_size = 192
    args.scale = 4

    model = CGSRN(args)
    model.train()

    from torchsummary import summary

    summary(model.cuda(), input_size=(3, 48, 48), batch_size=1)

    # 300*(batch_size*1000)/batch_size=300000 次迭代
    # 设每次迭代需要x秒，那么训练完毕需要300000x秒，折合83.3333x小时
    # x的可接受范围在0.5s左右，也即每次迭代必须在0.5s左右结束

