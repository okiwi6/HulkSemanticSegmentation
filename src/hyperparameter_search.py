import os
from datetime import timedelta
import sys

from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import optuna

from .callbacks.ExtraLogger import ValidationFigureLogger, GraphLogger
from .callbacks.Pruning import PyTorchLightningPruningCallback
from .model.LightningWrapper import LightningModel
from .dataloader.dataloader import DataModule
from .configuration.constants import CLASS_WEIGHTS, NUM_INPUT_CHANNELS, NUM_CLASSES

def compose_hyperparameters(trial: optuna.trial.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
        "tv_regularization_weighting": trial.suggest_float("tv_regularization_weighting", 1e-10, 1e-3, log=True),
        "label_smoothing": 0.0, # trial.suggest_float("label_smoothing", 0.0, 0.4),
        "swa_lrs": trial.suggest_float("swa_lrs", 1e-4, 1e-1),
        "use_class_weights": False, # trial.suggest_categorical("use_class_weights", [True, False]),
        "use_ycbcr": trial.suggest_categorical("use_ycbcr", [True, False]),
        "loss_type": "focal", # trial.suggest_categorical("loss_type", ["focal", "cross_entropy"])
        "optimizer": "lion", # trial.suggest_categorical("optimizer", ["lion", "adam"])
    }

GLOBAL_GPU_INDEX = 0

def objective(trial: optuna.trial.Trial):
    study_name = trial.study.study_name
    hyperparameters = compose_hyperparameters(trial)
        
    batch_size = 80

    dataset = DataModule(batch_size=batch_size, use_ycbcr=hyperparameters["use_ycbcr"])

    class_weights = CLASS_WEIGHTS
    if hyperparameters["use_class_weights"]:
        # Sample the class weights from the dataset
        class_weights = dataset.sample_class_weights()

    model = LightningModel(
        NUM_INPUT_CHANNELS, 
        NUM_CLASSES, 
        class_weights=class_weights,
        enable_validation_plots=False,
        learning_rate_reduction_factor=0.5,
        batch_size=batch_size,
        **hyperparameters
    )
    logger = TensorBoardLogger(
        save_dir="logs",
        name=study_name,
        version=f"trial_{trial.number}",
        default_hp_metric=False
    )

    dataset.setup("fit")
    train_length = len(dataset.train_dataset)
    steps = train_length // batch_size
    validation_interval = steps // 3

    print(f"Total steps: {steps}, Validation interval: {validation_interval}")

    trainer = Trainer(
        max_epochs=20,
        min_epochs=1,
        val_check_interval=validation_interval,
        devices=[GLOBAL_GPU_INDEX],
	callbacks = [
            LearningRateMonitor(logging_interval='step'), 
            StochasticWeightAveraging(swa_lrs=hyperparameters["swa_lrs"]),
            PyTorchLightningPruningCallback(
                trial,
                monitor="Validation/IoU"
            ),
            EarlyStopping(
                monitor="Validation/IoU",
                min_delta=0.001,
                patience=20,
                mode="max"
            )
        ],
        logger=logger,
        fast_dev_run=False,
    )
    
    trainer.fit(model, dataset)
    iou = trainer.callback_metrics["Validation/IoU"].item()
    
    return iou


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    study_name = sys.argv[1]
    storage = sys.argv[2]
    gpu_idx = sys.argv[3]
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{storage}.db",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    timeout = timedelta(weeks=4)
    GLOBAL_GPU_INDEX = int(gpu_idx)
    study.optimize(objective, timeout=timeout.total_seconds())
