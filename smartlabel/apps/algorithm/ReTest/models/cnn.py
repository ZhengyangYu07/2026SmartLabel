import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_color_logger


logger = setup_color_logger()
logger.debug("Loading cnn model...")

__all__ = ['CNN','CNN_small']


class CNN(nn.Module):
    def __init__(self, input_channel: int=3, n_outputs: int=10, dropout_rate: float=0.25, momentum: float=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super().__init__() # type: ignore
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(196, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=self.momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Linear(256, n_outputs)
        
        # initialization
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
                if m.bias is not None: # type: ignore
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


# just for mnist
class CNN_small(nn.Module):
    def __init__(self, input_channel: int=3, n_outputs: int=10, dropout_rate: float=0.25, momentum: float=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN_small, self).__init__() # type: ignore
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size = 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.classifier = nn.Linear(320, n_outputs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    cnn = CNN()
    encoder = nn.Sequential(*list(cnn.children())[:-2])
    print(encoder)
