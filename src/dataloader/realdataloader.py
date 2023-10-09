from ..configuration.constants import NUM_INPUT_CHANNELS

import torch
from torch.utils.data import Dataset, DataLoader
from os import path
import glob
import numpy as np
from PIL import Image
from lightning import LightningDataModule

class RealImageDataset(Dataset):
    def __init__(self, folder, transform=None, use_ycbcr=False):
        self.dataset = glob.glob(path.join(folder, "*.png"))
        self.number_of_data_pairs = len(self.dataset)
        self.use_ycbcr = use_ycbcr

        self.transform = transform

    def __len__(self):
        return self.number_of_data_pairs

    def __getitem__(self, idx):
        image = Image.open(self.dataset[idx])
        if self.use_ycbcr:
            image = image.convert('YCbCr')
        image = np.array(image, dtype=np.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
            assert(image.shape[0] == NUM_INPUT_CHANNELS)
        
        if isinstance(image, torch.Tensor):
            assert(not torch.any(torch.isnan(image)))

        return image