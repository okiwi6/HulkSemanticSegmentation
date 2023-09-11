from .model.LightningWrapper import LightningModel
# from .model.PIDNet import get_pred_model
from .dataloader.dataloader import DataModule
from .configuration.constants import CLASS_WEIGHTS, NUM_INPUT_CHANNELS, NUM_CLASSES

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from .callbacks.ExtraLogger import ValidationFigureLogger, GraphLogger

def main():
    # model = get_pred_model('s', NUM_CLASSES)
    # print(model)
    # model(torch.zeros((1,3,480,640)))
    # return
    dataset = DataModule(batch_size=20)
    
    # model = LightningModel(
    #     NUM_INPUT_CHANNELS, 
    #     NUM_CLASSES, 
    #     class_weights=CLASS_WEIGHTS, 
    #     learning_rate=1e-2,
    #     learning_rate_reduction_factor=0.8,
    #     iou_weighting=2.5,
    #     tv_regularization_weighting=0,
    #     label_smoothing=0.1,
    # )
    # model = LightningModel.load_from_checkpoint("lightning_logs/version_73/checkpoints/epoch=59-step=83859.ckpt")
    model = LightningModel(
        NUM_INPUT_CHANNELS, 
        NUM_CLASSES, 
        class_weights=CLASS_WEIGHTS, 
        enable_validation_plots=False,
        learning_rate=1e-2,
        learning_rate_reduction_factor=0.8,
        iou_weighting=2.5,
        tv_regularization_weighting=0,
        label_smoothing=0.1,
    )
    logger = TensorBoardLogger(
        save_dir="logs_tmp",
        name="test",
        version=f"trial_1",
        default_hp_metric=False
    )

    trainer = Trainer(
        max_epochs=80,
        val_check_interval=0.25,
        callbacks = [
            LearningRateMonitor(logging_interval='step'), 
            StochasticWeightAveraging(swa_lrs=1e-2),
            ModelCheckpoint(
                f"checkpoints/",
                monitor="Validation/IoU",
                mode="max"
            ),
            EarlyStopping(
                monitor="Validation/IoU",
                min_delta=0.01,
                patience=3,
                mode="max"
            )
        ],
        logger=logger,
        fast_dev_run=False,
    )
    trainer.fit(model, dataset)

if __name__ == "__main__":
    main()
