import sys
from lightning import Trainer
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from .model.LightningWrapper import LightningModel
from .dataloader.dataloader import DataModule
from .visualization.images import create_mask, display
from .configuration.constants import NUM_CLASSES

from PIL import Image

def main():
    model = LightningModel.load_from_checkpoint(sys.argv[1], batch_size=20, map_location=torch.device("cpu"))
    # model = LightningModel.load_from_checkpoint(sys.argv[1], batch_size=20)
    # print(model.downscale_factor)
    # torch.onnx.export(model.unet, torch.zeros((1,3,480 // 4, 640 // 4)), "segmentation_down4_rgb_nchw.onnx", input_names=["data"], output_names=["output"], verbose=False)
    # # print(model)
    # return
    # model.enable_validate_plots = True
    # dataset = DataModule(batch_size=5, use_ycbcr=True)
    dataset = DataModule(batch_size=5, use_ycbcr=False)
    # dataset.setup("pred")
    dataset.setup("real")
    
    dataloader = dataset.real_dataloader()
    # dataloader = dataset.real_dataloader()

    cmap = mpl.colormaps["jet"].resampled(NUM_CLASSES)

    # for idx, (images, masks) in enumerate(dataloader):
    #     img1 = images[0].permute(1,2,0).numpy() * 255
    #     img1 = Image.fromarray(img1.astype(np.uint8), "RGB")
    #     mask1 = cmap(masks[0].numpy()) * 255
    #     mask1 = Image.fromarray(mask1[:,:,:3].astype(np.uint8), "RGB")
    #     # print(img1.shape, mask1.shape)

    #     img1.save(f"blended/{idx}_img.png")
    #     # mask1.save(f"blended/{idx}_mask.png")

    #     blended = Image.blend(img1, mask1, alpha=0.65)
    #     blended.save(f"blended/{idx}_blended.png")
    #     if idx > 50:
    #         break

    # model.eval()
    # with torch.no_grad():
    #     for batch_idx, (images, masks) in enumerate(dataloader):
    #         print(f"{batch_idx}/{len(dataloader)}")
    #         prediction_masks = model._step(images, masks)["predicted"]
    #         prediction_masks = create_mask(prediction_masks)
    #         for i in range(images.shape[0]):
    #             fig = display([images[i], masks[i], prediction_masks[i]])
    #             fig.savefig(f"validation/fig_{batch_idx}_{i}.png")
    #             plt.close()

    model.eval()
    with torch.no_grad():
        for batch_idx, images in enumerate(dataloader):
            print(f"{batch_idx}/{len(dataloader)} {images.shape}")
            prediction_masks = model.unet(images)
            prediction_masks = create_mask(prediction_masks)
            for i in range(images.shape[0]):
                fig = display([images[i], prediction_masks[i]])
                fig.savefig(f"validation/fig_{batch_idx}_{i}.png")
                plt.close()
            
    #         img1 = images[0].permute(1,2,0).numpy() * 255
    #         img1 = Image.fromarray(img1.astype(np.uint8), "RGB")
    #         mask1 = cmap(np.squeeze(prediction_masks[0].numpy())) * 255
    #         print(mask1.shape)
    #         mask1 = Image.fromarray(mask1[:,:,:3].astype(np.uint8), "RGB")
    #         # print(img1.shape, mask1.shape)

    #         img1.save(f"blended/{batch_idx}_img.png")
    #         # mask1.save(f"blended/{batch_idx}_mask.png")

    #         blended = Image.blend(img1, mask1, alpha=0.65)
    #         blended.save(f"blended/{batch_idx}_blended.png")
    #         if batch_idx > 50:
    #             break



if __name__ == "__main__":
    main()
