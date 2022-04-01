import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def _weight_initialization(m, initialization="kaiming_normal"):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        if initialization == 'kaiming_normal':
            init.kaiming_normal_(m.weight)
        elif initialization == 'xavier_normal':
            init.xavier_normal_(m.weight)
        elif initialization == 'normal':
            init.normal_(m.weight)
        elif initialization == 'uniform':
            init.uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class ResNetBlock(nn.Module):
    """
    Args:
      in_planes (int):  Number of input planes.
      planes (int):     Number of output planes.
      stride (int):     Controls the stride.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out