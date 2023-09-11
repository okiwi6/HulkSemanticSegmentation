from ..configuration.constants import NUM_CLASSES

import matplotlib.pyplot as plt

from ..configuration.constants import NUM_CLASSES

import matplotlib.pyplot as plt
import torch

def create_mask(pred_mask):
    # This function assumes pred_mask is in NCHW format
    assert pred_mask.shape[1] == NUM_CLASSES
    pred_mask = torch.argmax(pred_mask, dim=1)
    pred_mask = pred_mask.unsqueeze(-1)
    
    return pred_mask

def display(display_list, mode="nchw", suptitle=""):
    if mode == "nchw":
        if len(display_list[0].shape) == 3:
            display_list[0] = display_list[0].permute(1,2,0)

    titles = ["Input Image", "True Mask", "Predicted"]
    configs = [
        {},
        {"vmin":0, "vmax": NUM_CLASSES-1},
        {"vmin":0, "vmax": NUM_CLASSES-1}
    ]

    # Figure out type
    fig, axes = plt.subplots(1, len(display_list), figsize=(15,5))
    
    for idx, (image, title, config) in enumerate(zip(display_list, titles, configs)):
        axis = axes[idx]
        axis.imshow(image.cpu(), **config)
        axis.set_title(title)
        axis.axis("off")
    
    fig.tight_layout()
    fig.suptitle(suptitle)
    
    return fig