import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        # upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = self.make_layer(ResidualBlock, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True)
        )
        self.block4 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=False)
        # block4 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        # block4.append(nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=False))
        # self.block4 = nn.Sequential(*block4)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(64))
        return nn.Sequential(*layers)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block1 + block3)

        return (torch.tanh(block4) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.in1(residual)
        residual = self.lrelu(residual)
        residual = self.conv2(residual)
        residual = self.in2(residual)

        return x + residual


# class UpsampleBLock(nn.Module):
    # def __init__(self, in_channels, up_scale):
        # super(UpsampleBLock, self).__init__()
        # self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, stride=1, padding=1, bias=False)
        # self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    # def forward(self, x):
        # x = self.conv(x)
        # x = self.pixel_shuffle(x)
        # x = self.lrelu(x)
        # return x
