import sys
from lightning import Trainer
import torch
import matplotlib.pyplot as plt

from .model.LightningWrapper import LightningModel
from .dataloader.dataloader import DataModule
from .visualization.images import create_mask, display

def main():
    model = LightningModel.load_from_checkpoint(sys.argv[1], batch_size=1, map_location=torch.device("cpu"))

    torch.onnx.export(model.unet, torch.zeros((1,3,480,640)), "segmentation.onnx", verbose=False)
    print(model)

    model.enable_validate_plots = True
    dataset = DataModule(batch_size=5)
    dataset.setup("validate")
    dataloader = dataset.val_dataloader()

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            print(f"{batch_idx}/{len(dataloader)}")
            prediction_masks = model._step(images, masks)["predicted"]
            prediction_masks = create_mask(prediction_masks)
            for i in range(images.shape[0]):
                fig = display([images[i], masks[i], prediction_masks[i]])
                fig.savefig(f"validation/fig_{batch_idx}_{i}.png")
                plt.close()



if __name__ == "__main__":
    main()