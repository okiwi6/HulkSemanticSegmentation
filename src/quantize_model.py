import nncf
import sys
from lightning import Trainer
import torch
import numpy as np

from .model.LightningWrapper import LightningModel
from .dataloader.dataloader import DataModule
from .configuration.constants import NUM_CLASSES

def main():
    model = LightningModel.load_from_checkpoint(sys.argv[1], loss_type="cross_entropy", map_location=torch.device("cpu"))
    print(f"The set downscale factor is {model.downscale_factor}")
    
    dataset = DataModule(batch_size=20, use_ycbcr=False)
    dataset.setup("fit")

    trainer = Trainer(
        devices=1,
        num_nodes=1,
        max_epochs=1,
        limit_test_batches=10,
        accelerator="cpu",
    )

    # trainer.test(model, dataset)
    # iou = trainer.callback_metrics["Validation/IoU"].item()

    # print(f"Test mIoU (Baseline): {iou}")

    def transform_fn(data_item):
        images, _targets = data_item
        return images
    
    calibration_dataset = nncf.Dataset(dataset.train_dataloader(), transform_fn)
    quantized_model = nncf.quantize(model.unet, calibration_dataset)
    model.unet = quantized_model

    torch.onnx.export(quantized_model, torch.zeros((1,3,480 // model.downscale_factor, 640 // model.downscale_factor)), "quantized_segmentation_down4_rgb_nchw.onnx", input_names=["data"], output_names=["output"], verbose=False)
    trainer.test(model, dataset)
    iou = trainer.callback_metrics["Validation/IoU"].item()
    
    print(f"Test mIoU (Quantized): {iou}")



if __name__ == "__main__":
    main()