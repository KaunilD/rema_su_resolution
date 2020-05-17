import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    [conv2D] => [BN] => [ReLU]
    [conv2D] => [BN] => [ReLU]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    [DoubleConv] => [AvgPool2D]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_avgpool = nn.Sequential(
                DoubleConv(in_channels, out_channels),
                nn.AvgPool2d(kernel_size = 2)
            )

    def forward(self, x):
        return self.conv_avgpool(x)


class Up(nn.Module):
    """
    [trnaspposedConv2d] => [DoubleConv]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=2, stride=2)

        self.up_conv = nn.Sequential(
            self.up,
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.up_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.up_sample_0 = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=2, stride=2)

        self.drop_out = torch.nn.Dropout(p=0.25)

        self.down_block_1 = Down(in_channels=in_channels, out_channels=64)
        self.down_block_2 = Down(in_channels=64, out_channels=128)
        self.down_block_3 = Down(in_channels=128, out_channels=256)
        self.down_block_4 = Down(in_channels=256, out_channels=512)
        self.down_block_5 = Down(in_channels=512, out_channels=1024)

        self.up_block_1 = Up(in_channels=1024, out_channels = 512)
        self.up_block_2 = Up(in_channels=1024, out_channels = 256)
        self.up_block_3 = Up(in_channels=512, out_channels = 128)
        self.up_block_4 = Up(in_channels=256, out_channels = 64)

        self.out_conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)

    def forward(self, x):
        """
        x = Nx1x256x256
        """
        up_0 = self.up_sample_0(x) # x = Nx1x512x512
        up_0 = self.up_sample_0(up_0) # x = Nx1x512x512

        down_1 = self.down_block_1(up_0)

        down_1d = self.drop_out(down_1)
        down_2 = self.down_block_2(down_1d)

        down_2d = self.drop_out(down_2)
        down_3 = self.down_block_3(down_2d)

        down_3d = self.drop_out(down_3)
        down_4 = self.down_block_4(down_3d)

        down_4d = self.drop_out(down_4)
        down_5 = self.down_block_5(down_4d)

        down_5d = self.drop_out(down_5)

        up_1 = self.up_block_1(down_5d)
        up_1 = torch.cat([up_1, down_4d], axis = 1)

        up_2 = self.up_block_2(up_1)
        up_2 = torch.cat([up_2, down_3d], axis = 1)

        up_3 = self.up_block_3(up_2)
        up_3 = torch.cat([up_3, down_2d], axis = 1)

        up_4 = self.up_block_4(up_3)
        up_4 = torch.cat([up_4, down_1d], axis = 1)

        out = self.out_conv(up_4)


        return out

def main():

    # input
    inp = np.ones((1, 1, 256, 256), dtype=np.float32)
    inp = torch.from_numpy(inp)

    # model
    model = UNet()

    # out
    out = model(inp)
    print(out.shape)

if __name__=="__main__":
    main()
