import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Tuple


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float=0.0):
        super(BasicBlock, self).__init__() # type: ignore
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float=0.0):
        super(BottleneckBlock, self).__init__() # type: ignore
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)
#过渡层（压缩特征）
class TransitionBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, dropRate: float=0.0):
        super(TransitionBlock, self).__init__() # type: ignore
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)
#密集连接块
class DenseBlock(nn.Module):
    def __init__(self, nb_layers: int, in_planes: int, growth_rate: int, block: Type[nn.Module], dropRate: float=0.0):
        super(DenseBlock, self).__init__() # type: ignore
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block: Type[nn.Module], in_planes: int, growth_rate: int, nb_layers: int, dropRate: float) -> nn.Sequential:
        layers: List[nn.Module] = []
        for i in range(int(nb_layers)):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
#主DenseNet类
class DenseNet3(nn.Module):
    def __init__(self, depth: int, num_classes: int, growth_rate: int=12,
                 reduction: float=0.5, bottleneck: bool=True, dropRate: float=0.0):
        super(DenseNet3, self).__init__() # type: ignore
        in_planes = 2 * growth_rate
        n = (depth - 4) // 3
        if bottleneck == True:
            n = n // 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out2 = self.trans2(self.block2(out))
        out = self.block3(out2)
        out3 = self.relu(self.bn1(out))
        out = F.avg_pool2d(out3, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


    def feature_list(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out_list: List[torch.Tensor] = []
        
        out = self.conv1(x)
        for name, module in self.block1.layer._modules.items(): # pyright: ignore[reportPrivateUsage]
            out = module(out)     
        out = self.trans1(out)

        for name, module in self.block2.layer._modules.items(): # pyright: ignore[reportPrivateUsage]
            out = module(out)  
        out = self.trans2(out)
        
        for name, module in self.block3.layer._modules.items(): # pyright: ignore[reportPrivateUsage]
            out = module(out)
            if name == str(5) or name == str(10):
                out_list.append(out)
                
        out = self.relu(self.bn1(out))
        out_list.append(out)
        
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        
        return self.fc(out), out_list
    