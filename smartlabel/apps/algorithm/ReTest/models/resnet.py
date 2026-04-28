import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Type, Union, List, Tuple, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_color_logger


logger = setup_color_logger()
logger.debug("Loading resnet model...")

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']

# model_list = __all__

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#ResNet基础块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: nn.Module | None=None):
        super(BasicBlock, self).__init__() # type: ignore
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#ResNet瓶颈块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: nn.Module | None=None):
        super(Bottleneck, self).__init__() # type: ignore
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#主ResNet类
class ResNet(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], in_channels: int=3, num_classes: int=1000):
        super(ResNet, self).__init__() # type: ignore
        self.inplanes = 64 
        # should modify the kernel_size
        # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0]) # type: ignore
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # type: ignore
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # type: ignore
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # type: ignore
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int=1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    #返回中间特征
    def feature_list(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        output_list: List[torch.Tensor]= []
        out = F.relu(self.bn1(self.conv1(x)))
        for _, module in self.layer1._modules.items(): # pyright: ignore[reportPrivateUsage]
            out = module(out)
        for _, module in self.layer2._modules.items(): # pyright: ignore[reportPrivateUsage]
            out = module(out)
        for _, module in self.layer3._modules.items(): # pyright: ignore[reportPrivateUsage]
            out = module(out)
        for _, module in self.layer4._modules.items(): # pyright: ignore[reportPrivateUsage]
            out = module(out)
            output_list.append(out) # type: ignore
            # print(out.size())
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        output_list.append(out)
        # print(out.size())
        y: torch.Tensor = self.fc(out)
        return y, output_list


def resnet18(pretrained: bool=False, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet18']))
    return model


def resnet34(pretrained: bool=False, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet34']))
    return model


def resnet50(pretrained: bool=False, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet50']))
    return model


def resnet101(pretrained: bool=False, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet101']))
    return model


def resnet152(pretrained: bool=False, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet152']))
    return model

if __name__ == "__main__":
    res = resnet18(pretrained=False)
    import torch
    input = torch.randn(1,3,32,32)
    fe = res.feature_list(input)
