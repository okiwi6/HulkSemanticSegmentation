import torch
import torch.nn as nn
import torch.nn.functional as F

from .building_blocks import ConvBatchLeakyReLU


class UNet1(nn.Module):
    def __init__(self, number_of_channels, number_of_classes):
        super(UNet1, self).__init__()

        self.conv1 = ConvBatchLeakyReLU(number_of_channels, 8, kernel_size=3, stride=1, padding=1, separable=False)
        self.conv2 = ConvBatchLeakyReLU(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = ConvBatchLeakyReLU(8, 8, kernel_size=3, stride=2, padding=1)

        self.conv3 = ConvBatchLeakyReLU(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBatchLeakyReLU(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = ConvBatchLeakyReLU(16, 16, kernel_size=3, stride=2, padding=1)

        self.conv5 = ConvBatchLeakyReLU(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv9 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv10 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv11 = ConvBatchLeakyReLU(48, 16, kernel_size=3, stride=1, padding=1)
        self.conv12 = ConvBatchLeakyReLU(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv13 = ConvBatchLeakyReLU(24, 8, kernel_size=3, stride=1, padding=1)
        self.conv14 = ConvBatchLeakyReLU(8, 8, kernel_size=3, stride=1, padding=1)

        self.final_conv = nn.Conv2d(8, number_of_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        pool1 = self.pool1(conv2)

        conv3 = self.conv3(pool1)
        conv4 = self.conv4(conv3)
        pool2 = self.pool2(conv4)

        conv5 = self.conv5(pool2)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)

        up8 = torch.cat((F.interpolate(conv10, scale_factor=2, mode='nearest'), conv4), dim=1)
        conv11 = self.conv11(up8)
        conv12 = self.conv12(conv11)

        up9 = torch.cat((F.interpolate(conv12, scale_factor=2, mode='nearest'), conv2), dim=1)
        conv13 = self.conv13(up9)
        conv14 = self.conv14(conv13)

        output = self.final_conv(conv14)
        return torch.sigmoid(output)

class UNet2(nn.Module):
    def __init__(self, number_of_channels, number_of_classes):
        super(UNet2, self).__init__()

        self.conv1 = ConvBatchLeakyReLU(number_of_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBatchLeakyReLU(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = ConvBatchLeakyReLU(8, 8, kernel_size=3, stride=2, padding=1)

        self.conv3 = ConvBatchLeakyReLU(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBatchLeakyReLU(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = ConvBatchLeakyReLU(16, 16, kernel_size=3, stride=2, padding=1)

        self.conv5 = ConvBatchLeakyReLU(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv9 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1, separable=False)
        self.conv10 = ConvBatchLeakyReLU(32, 32, kernel_size=3, stride=1, padding=1, separable=False)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv11 = ConvBatchLeakyReLU(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv12 = ConvBatchLeakyReLU(16, 16, kernel_size=3, stride=1, padding=1)

        self.upconv2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.conv13 = ConvBatchLeakyReLU(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv14 = ConvBatchLeakyReLU(8, 8, kernel_size=3, stride=1, padding=1)

        self.final_conv = nn.Conv2d(8, number_of_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        pool1 = self.pool1(conv2)

        conv3 = self.conv3(pool1)
        conv4 = self.conv4(conv3)
        pool2 = self.pool2(conv4)

        conv5 = self.conv5(pool2)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8) + conv5
        conv10 = self.conv10(conv9)
        up8 = self.upconv1(conv10) + conv4
        conv11 = self.conv11(up8)
        conv12 = self.conv12(conv11)
        up9 = self.upconv2(conv12) + conv2
        conv13 = self.conv13(up9)
        conv14 = self.conv14(conv13)

        output = self.final_conv(conv14)
        return torch.sigmoid(output)
