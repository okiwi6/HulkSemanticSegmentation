from lightning import LightningModule
from torchmetrics import JaccardIndex
from torchmetrics.functional.image import total_variation
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from ..configuration.constants import CLASS_TO_LABEL
from ..visualization.images import display, create_mask
from ..visualization.confusion_matrix import fig_confusion_matrix
from .UNet import UNet2, UNet1

class LightningModel(LightningModule):
    def __init__(self, number_of_channels, number_of_classes, class_weights, learning_rate, learning_rate_reduction_factor, tv_regularization_weighting, label_smoothing, batch_size, enable_validation_plots=True, **kwargs):
        super().__init__()
        print(f"Unused keyword arguments: {kwargs}")
        self.unet = UNet2(number_of_channels, number_of_classes)
        self.num_classes = number_of_classes
        # self.unet = DeepLabV3MobileNetV3(number_of_classes)
        self.loss = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights), label_smoothing=label_smoothing)
        self.jaccard = JaccardIndex('multiclass', num_classes=number_of_classes)
        self.class_accuracy = JaccardIndex('multiclass', num_classes=number_of_classes, average=None)
        self.learning_rate = learning_rate
        self.learning_rate_reduction_factor = learning_rate_reduction_factor
        self.tv_regularization_weighting = tv_regularization_weighting
        self.label_smoothing = label_smoothing
        self.enable_validation_plots = enable_validation_plots
        self.batch_size = batch_size
        self.should_plot_this_step = False
        self.save_hyperparameters()

    def log_graph(self):
        self.logger.experiment.add_graph(self.unet, torch.zeros((1, 3, 640, 480)).to("cuda"))

    def _step(self, images, masks):
        # x = images.permute(0, 3, 1, 2)
        predicted_masks = self.unet(images)
        
        iou = self.jaccard(predicted_masks, masks)
        cross_entropy = self.loss(predicted_masks, masks.long())
        tv = total_variation(predicted_masks, reduction="mean")
        tv_target = total_variation(nn.functional.one_hot(masks.long(), num_classes=self.num_classes), reduction="mean")

        loss = cross_entropy + self.tv_regularization_weighting * torch.abs(tv - tv_target)
        return {
            "loss": loss,
            "cross_entropy": cross_entropy,
            "iou": iou,
            "tv": tv,
            "tv_target": tv_target,
            "predicted": predicted_masks
        }

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, masks = batch
        result = self._step(images, masks)

        loss = result["loss"]
        iou = result["iou"]
        cross_entropy = result["cross_entropy"]
        
        # Logging to TensorBoard (if installed) by default
        self.log("Train/Loss", loss, batch_size=self.batch_size)
        self.log("Train/IoU", iou, batch_size=self.batch_size)
        self.log("Train/CrossEntropy", cross_entropy, batch_size=self.batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        images, masks = batch
        step_result = self._step(images, masks)

        loss = step_result["loss"]
        iou = step_result["iou"]
        tv = step_result["tv"]
        tv_target = step_result["tv_target"]
        predicted_masks = step_result["predicted"]

        self.log("Validation/Loss", loss, sync_dist=True, batch_size=self.batch_size)
        self.log("Validation/IoU", iou, sync_dist=True, batch_size=self.batch_size)
        self.log("Validation/TotalVariation", torch.abs(tv - tv_target), sync_dist=True, batch_size=self.batch_size)

        class_ious = self.class_accuracy(predicted_masks, masks)
        self.log_dict(
            {
                f"Validation/{title}": value
                for title, value in zip(CLASS_TO_LABEL, class_ious)
            },
            sync_dist=True, 
            batch_size=self.batch_size
        )

        if self.should_plot_this_step:
            self.should_plot_this_step = False
            fig = display([images[0], masks[0], create_mask(predicted_masks)[0]])
            self.logger.experiment.add_figure("Validation/Image", fig, self.global_step)

            fig = fig_confusion_matrix(predicted_masks, masks)
            self.logger.experiment.add_figure("Validation/MultiClassConfusionMatrix", fig, self.global_step)
            plt.close("all")

    def on_validation_epoch_start(self):
        self.should_plot_this_step = self.enable_validation_plots

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=self.learning_rate_reduction_factor, patience=10, min_lr=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Validation/Loss"}
