from torch import nn

from pysot.models.enhance.dcn.deform_conv import DeformConvPack


class DeformConv(nn.Module):
    def __init__(self, in_channels=[256, 256, 256], out_channels=[256, 256, 256]):
        super(DeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deform_convs = nn.ModuleList()
        for i in range(3):
            self.deform_convs.append(
                DeformConvPack(in_channels=self.in_channels[i], out_channels=self.out_channels[i], kernel_size=3,
                               padding=1))

    def forward(self, z, x):
        z_out = []
        x_out = []

        for idx, (z_f, x_f) in enumerate(zip(z, x)):
            z_f = self.deform_convs[idx](z_f)
            x_f = self.deform_convs[idx](x_f)

            z_out.append(z_f)
            x_out.append(x_f)

        return z_out, x_out