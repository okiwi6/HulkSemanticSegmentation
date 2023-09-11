# Hulk Semantic Segmentation
This repository contains ongoing efforts for semantic segmentation for Robocup SPL.

* clone the BHuman dataset from https://sibylle.informatik.uni-bremen.de/public/datasets/semantic_segmentation/dataset.zip and unzip it into the repository folder
* create a python 3.10 virtual environment with `python -m venv .venv`
* enter the virtual environment (`. .venv/bin/activate` in bash)
* install all dependencies with `pip install -r requirements.txt`
* perform the train/test split with `python -m src.split_test`
(this will perform a 90/10 train/test split and put the results into the datasplit folder)

You can now train the network with
`python -m src.main`.
All relevant global configuration can be found in `src.configuration.constants`.
The network is initialized in the `LightningWrapper` and can be substituted there.
The data augmentation pipeline in use is `src.augmentation.augmenter2`.

Hyperparameter search can be started with `python -m src.hyperparameter_search {STUDY_NAME}`, however
I used the script `start_hyperparameter_search.sh` which starts two processes for hyperparameter search on two GPUs since I ran into issues doing this in one process.
