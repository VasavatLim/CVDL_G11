import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation=F.relu, padding=0, dropoutp=0
    ) -> None:
        super(DoubleConv, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=padding
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=padding
        )
        self.drop = nn.Dropout2d(p=dropoutp)

        # batchnorm
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # activation function
        self.acc = activation

    def forward(self, x):
        # first conv
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.acc(x)

        # second conv
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.acc(x)

        return x


def stack(x, x_skip):
    diffY = x.size()[2] - x_skip.size()[2]
    diffX = x.size()[3] - x_skip.size()[3]

    x_skip = F.pad(
        x_skip, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
    )
    return torch.cat([x, x_skip], dim=1)


class UpSample(nn.Module):
    def __init__(self, in_channels) -> None:
        super(UpSample, self).__init__()

        # up-conv
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels) -> None:
        super(AttentionGate, self).__init__()

        # conv
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)

        # batchnorm
        self.normg = nn.BatchNorm2d(inter_channels)
        self.normx = nn.BatchNorm2d(inter_channels)

    def forward(self, x, g):
        # conv
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # normalize
        g1 = self.normg(g1)
        x1 = self.normx(x1)

        # add
        add = F.relu(g1 + x1)
        psi = self.psi(add)
        att = F.sigmoid(psi)

        # mul
        return x * att


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UNet, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        return self.conv(x)
