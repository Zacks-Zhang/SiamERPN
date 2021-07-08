from torch import nn

from pysot.core.config import cfg
from pysot.models.enhance.dcn.deform_conv import DeformConvPack
from pysot.models.enhance.deform_attn.attn import PAM_Module, CAM_Calculate, CAM_Use

class DeformConv(nn.Module):
    def __init__(self, in_channels=[256, 256, 256], out_channels=[256, 256, 256]):
        super(DeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cam_cals = nn.ModuleList()
        self.cam_uses = nn.ModuleList()
        self.deform_convs = nn.ModuleList()
        for i in range(3):
            self.deform_convs.append(
                DeformConvPack(in_channels=self.in_channels[i], out_channels=self.out_channels[i], kernel_size=3,
                               padding=1))
            if cfg.ENHANCE.BACKBONE.cross_attn:
                self.cam_cals.append(CAM_Calculate(self.in_channels[i]))
                self.cam_uses.append(CAM_Use(self.in_channels[i]))

    def forward(self, z, x):
        z_out = []
        x_out = []

        for idx, (z_f, x_f) in enumerate(zip(z, x)):

            if cfg.ENHANCE.BACKBONE.cross_attn:

                z_attention = self.cam_cals[idx](z_f)
                x_attention = self.cam_cals[idx](x_f)

                zx_sc_feat = self.cam_uses[idx](z_f, x_attention)
                xz_sc_feat = self.cam_uses[idx](x_f, z_attention)

                z_f = zx_sc_feat
                x_f = xz_sc_feat

            z_f = self.deform_convs[idx](z_f)
            x_f = self.deform_convs[idx](x_f)

            z_out.append(z_f)
            x_out.append(x_f)

        return z_out, x_out