from torch import nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.in_c = [512, 1024, 2048]
        self.out_c = [256, 256, 256]
        for i in range(3):
            self.add_module('downsample' + str(i + 2),
                            nn.Sequential(
                                nn.Conv2d(self.in_c[i], self.out_c[i], kernel_size=1, bias=False, padding=0, stride=1),
                                nn.BatchNorm2d(self.out_c[i]),
                            ))

    def forward(self, x):
        out = []
        for i in range(3):
            downsampler = getattr(self, 'downsample' + str(i + 2))
            out.append(downsampler(x[i]))
            # print(x[i].shape)

        # out.append(self.upsample_add(x[3], x[2]))
        out[1] = self.upsample_add(out[2], out[1])
        out[0] = self.upsample_add(out[1], out[0])
        out.pop(-1)
        return out

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def downsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.adaptive_avg_pool2d(x, output_size=(H, W)) + y