import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabv3(nn.Module):
    def __init__(self, number_of_input_channels, number_of_classes):
        super(DeepLabv3, self).__init__()
        self.deeplab = deeplabv3_resnet50(
            weights='COCO_WITH_VOC_LABELS_V1',
            weights_backbone='IMAGENET1K_V1'
        )
        self.deeplab.classifier[4] = nn.Conv2d(256, number_of_classes, kernel_size=(1,1), stride=(1,1))
    
    def forward(self, x):
        x = self.deeplab(x)['out']

        return x