import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

EPS = 1e-8

def defaulttensor(sub_batch_size, device):
    """
    Returns a tensor for the given sub_size.

    Args:
        sub_batch_size: (int): write your description
        device: (todo): write your description
    """
    return torch.zeros(sub_batch_size).to(device)

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            in_dim: (int): write your description
            reduction_dim: (int): write your description
            bins: (int): write your description
            BatchNorm: (todo): write your description
        """
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class SimpleEncoder(nn.Module):
    def __init__(self):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
        """
        super(SimpleEncoder, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        bottleneck = list(resnet.layer1.children())
        self.layer1 = nn.Sequential(bottleneck[0], bottleneck[1], bottleneck[2])

        self.ppm = PPM(256, 64, (1, 2, 3, 6), torch.nn.BatchNorm2d)

    def forward(self, x):
        """
        Forward computation of the layer

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.ppm(x)
        # import ipdb
        # ipdb.set_trace()
        return x, x