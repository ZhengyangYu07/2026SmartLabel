import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from typing import Type, List

""" Wide Resnet """

def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2)) # type: ignore
        init.constant(m.bias, 0) # type: ignore
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1) # type: ignore
        init.constant(m.bias, 0) # type: ignore

class wide_basic(nn.Module):
    def __init__(self, in_planes: int, planes: int, dropout_rate: float, stride: int=1):
        super(wide_basic, self).__init__() # type: ignore
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate>0:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout_rate>0:
            out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class WideResNet_encoder(nn.Module):
    def __init__(self, dropout_rate: float=0, depth: int=28, widen_factor: int=2, num_classes: int=10):
        super(WideResNet_encoder, self).__init__() # type: ignore
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.out_feature = nStages[3]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: # pyright: ignore[reportUnnecessaryComparison]
                    nn.init.constant_(m.bias, 0.0)

    def _wide_layer(self, block: Type[nn.Module], planes: int, num_blocks: int, dropout_rate: float, stride: int) -> nn.Sequential:
        """ Create a wide layer with the specified number of blocks """
        strides = [stride] + [1]*(num_blocks-1)
        layers: List[nn.Module]= []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out