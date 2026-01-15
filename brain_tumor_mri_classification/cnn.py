import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14400, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.fc(x)
        return x
    
class TweakedSimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30752, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.fc(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = False):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward (self, x):
        return self.relu(self.batchnorm2d(self.conv2d(x)))
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pooling):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, stride=1, padding=0),
            ConvBlock(red_3x3,out_3x3,kernel_size=3,stride=1,padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1, stride=1, padding=0),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1_pooling, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
    
class InceptionCNN(nn.Module):
    def __init__(self):
        super(InceptionCNN, self).__init__()

        self.conv1 = ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception = InceptionBlock(32, 16, 64, 96, 8, 16, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144*4, 4)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.inception(x)
        x = self.pool2(x)
        x = self.gap(x)
        x = self.fc(x)
        return x