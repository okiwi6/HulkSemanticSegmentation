from .model.LightningWrapper import LightningModel
# from .model.PIDNet import get_pred_model
from .dataloader.dataloader import DataModule
from .configuration.constants import CLASS_WEIGHTS, NUM_INPUT_CHANNELS, NUM_CLASSES

import torch
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from .callbacks.ExtraLogger import ValidationFigureLogger, GraphLogger

def main():
    # model = get_pred_model('s', NUM_CLASSES)
    # print(model)
    # model(torch.zeros((1,3,480,640)))
    # return
    batch_size = 6
    dataset = DataModule(batch_size=batch_size, use_ycbcr=False)
    
    # model = LightningModel.load_from_checkpoint("lightning_logs/version_110/checkpoints/epoch=0-step=4666.ckpt")
    model = LightningModel(
        NUM_INPUT_CHANNELS, 
        NUM_CLASSES, 
        class_weights=CLASS_WEIGHTS, 
        loss_type="focal",
        optimizer="lion",
        enable_validation_plots=True,
        learning_rate=2e-4,
        learning_rate_reduction_factor=0.8,
        tv_regularization_weighting=0.0,
        label_smoothing=0.28,
        batch_size=batch_size,
    )
    trainer = Trainer(
        max_epochs=80,
        min_epochs=5,
        val_check_interval=0.5,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks = [
            ModelCheckpoint(save_top_k=5, monitor="Validation/IoU", mode="max"),
            LearningRateMonitor(logging_interval='step'), 
            StochasticWeightAveraging(swa_lrs=0.00054),
            EarlyStopping(
                monitor="Validation/IoU",
                min_delta=0.001,
                patience=10,
                mode="max"
            )
        ],
        fast_dev_run=False,
    )
    trainer.fit(model, dataset)

if __name__ == "__main__":
    main()
