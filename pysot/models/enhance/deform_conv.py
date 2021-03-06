import torch
from torch import nn
import torchvision.ops
from pysot.core.config import cfg

from pysot.models.enhance.ecanet import ECA
from pysot.models.enhance.dual_attn import PAM_Module, CAM_Calculate, CAM_Use


class DeformConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformConv2d, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class DeformConv(nn.Module):
    def __init__(self, in_channels=[256, 256, 256], out_channels=[256, 256, 256]):
        super(DeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attn = nn.ModuleList()
        self.cam_uses = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        for i in range(3):
            if cfg.ENHANCE.RPN.deform_conv:
                self.deform_convs.append(
                    DeformConv2d(in_channels=self.in_channels[i], out_channels=self.out_channels[i], kernel_size=3,
                                 padding=1))
            if cfg.ENHANCE.BACKBONE.cross_attn:
                # self.cam_cals.append(CAM_Calculate(self.in_channels[i]))
                # self.cam_uses.append(CAM_Use(self.in_channels[i]))
                self.cross_attn.append(ECA(self.in_channels[i]))

    def forward(self, z, x):
        z_out = []
        x_out = []

        for idx, (z_f, x_f) in enumerate(zip(z, x)):

            if cfg.ENHANCE.BACKBONE.cross_attn:
                z_attn = self.cross_attn[idx](z_f)
                x_attn = self.cross_attn[idx](x_f)

                zz_attn = z_f * z_attn.expand_as(z_f)
                xx_attn = x_f * x_attn.expand_as(x_f)

                zx_attn = z_f * x_attn.expand_as(z_f)
                xz_attn = x_f * z_attn.expand_as(x_f)

                # z_attention = self.cam_cals[idx](z_f)
                # x_attention = self.cam_cals[idx](x_f)
                #
                # zx_sc_feat = self.cam_uses[idx](z_f, x_attention)
                # xz_sc_feat = self.cam_uses[idx](x_f, z_attention)

                z_f = zz_attn + zx_attn
                x_f = xx_attn + xz_attn

            if cfg.ENHANCE.RPN.deform_conv:
                z_f = self.deform_convs[idx](z_f)
                x_f = self.deform_convs[idx](x_f)

            z_out.append(z_f)
            x_out.append(x_f)

        return z_out, x_out