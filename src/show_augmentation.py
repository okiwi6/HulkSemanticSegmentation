import sys
from lightning import Trainer
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from .dataloader.dataloader import DataModule
from .visualization.images import create_mask, display
from .configuration.constants import NUM_CLASSES

from PIL import Image

def main():
    # dataset = DataModule(batch_size=5, use_ycbcr=True)
    dataset = DataModule(batch_size=5, use_ycbcr=False)
    dataset.setup("fit")
    
    dataloader = dataset.train_dataloader()
    # dataloader = dataset.real_dataloader()

    cmap = mpl.colormaps["jet"].resampled(NUM_CLASSES)

    for idx, (images, masks) in enumerate(dataloader):
        print(torch.min(images), torch.max(images), images.shape, masks.shape)
        fig = display([images[0], masks[0]])
        plt.savefig(f"validation/augmented_{idx}")
        plt.close()
        if idx > 30:
            break




if __name__ == "__main__":
    main()