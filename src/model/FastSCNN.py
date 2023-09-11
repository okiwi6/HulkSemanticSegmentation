import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSCNN(nn.Module):
    def __init__(self, number_of_input_channels, number_of_classes):
        super(FastSCNN, self).__init__()
