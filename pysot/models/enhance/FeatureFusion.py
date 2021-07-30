from torch import nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

    def forward(self, x):
        out = []
        out.append(self.upsample_add(x[2], x[1]))
        out.append(self.downsample_add(x[0], x[1]))
        x = out
        return x

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def downsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.adaptive_avg_pool2d(x, output_size=(H, W)) + y