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
        "learning_rate_reduction_factor": trial.suggest_float("learning_rate_reduction_factor", 1e-1, 1e-0, log=True),
        "tv_regularization_weighting": trial.suggest_float("tv_regularization_weighting", 1e-10, 1e-3, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 1.0),
        "batch_size": trial.suggest_int("batch_size", 1, 20),
        "swa_lrs": trial.suggest_float("swa_lrs", 1e-4, 1e-1),
    }

def objective(trial: optuna.trial.Trial):
    study_name = trial.study.study_name
    hyperparameters = compose_hyperparameters(trial)
        
    dataset = DataModule(batch_size=hyperparameters["batch_size"])
    model = LightningModel(
        NUM_INPUT_CHANNELS, 
        NUM_CLASSES, 
        class_weights=CLASS_WEIGHTS, 
        enable_validation_plots=False,
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
    steps = train_length // hyperparameters["batch_size"]
    validation_interval = steps // 3

    print(f"Total steps: {steps}, Validation interval: {validation_interval}")

    trainer = Trainer(
        max_epochs=20,
        val_check_interval=validation_interval,
        strategy=DDPStrategy(process_group_backend='nccl'),
        callbacks = [
            LearningRateMonitor(logging_interval='step'), 
            StochasticWeightAveraging(swa_lrs=hyperparameters["swa_lrs"]),
            PyTorchLightningPruningCallback(
                trial,
                monitor="Validation/IoU"
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
    # initial_performance = trainer.validate(model, dataset)[0]["Validation/IoU"]
    # logger.log_hyperparams(
    #     hyperparameters, metrics={"Validation/IoU": initial_performance}
    # )
    trainer.fit(model, dataset)
    iou = trainer.callback_metrics["Validation/IoU"].item()
    return iou

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    study_name = sys.argv[1]
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{study_name}.db",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    timeout = timedelta(weeks=4)
    study.optimize(objective, timeout=timeout.total_seconds())
