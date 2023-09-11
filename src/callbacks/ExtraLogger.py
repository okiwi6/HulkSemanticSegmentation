from lightning import Callback
from ..visualization.images import display, create_mask
from ..visualization.confusion_matrix import fig_confusion_matrix

class ValidationFigureLogger(Callback):
    def on_validation_end(self, trainer, pl_module):
        val_dataloader = trainer.datamodule.val_dataloader()

        pl_module.eval()
        images, masks = next(iter(val_dataloader))
        loss, predicted = pl_module._step(images, masks)

        fig = display([images[0], masks[0], create_mask(predicted_masks)[0]])
        trainer.logger.experiment.add_figure("Validation/Image", fig, trainer.global_step)

        fig = fig_confusion_matrix(predicted_masks, masks)
        trainer.logger.experiment.add_figure("Validation/MultiClassConfusionMatrix", fig, trainer.global_step)

class GraphLogger(Callback):
    def on_train_start(self, trainer, pl_module):
        pl_module.log_graph()