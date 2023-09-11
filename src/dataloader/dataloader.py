from .sample import Sample
from ..configuration.constants import NUM_CLASSES, NUM_AUGMENTATION_VARIANTS, NUM_INPUT_CHANNELS
from ..augmentation.augmenter2 import Augmenter

import torch
from torch.utils.data import Dataset, DataLoader
from os import path
import glob
from itertools import chain
import numpy as np
from lightning import LightningDataModule

class ImageDataset(Dataset):
    def __init__(self, dataset_path, augmentation_variants, transform=None, target_transform=None):
        self.raw_paths = sorted(glob.glob(path.join(dataset_path, "raw", "*.png")), key=lambda data: int(self.filename(data)))
        self.augmented_positive_paths = [
            sorted(glob.glob(path.join(dataset_path, "augmented", f"{variant}_*.png")))
            for variant in range(augmentation_variants)
        ]
        self.augmented_negative_paths = sorted(glob.glob(path.join(dataset_path, "augmented", "1000000*.png")))
        self.segmentation_paths = sorted(glob.glob(path.join(dataset_path, "segmentation", "*.png")), key=lambda data: int(self.filename(data)))
        self.augmentation_variants = augmentation_variants
        self.dataset = list(chain.from_iterable(
            [self.create_positive_samples(i) for i in range(len(self.raw_paths))] +
            [self.create_negative_samples()]
        ))
        self.number_of_data_pairs = len(self.dataset)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.number_of_data_pairs

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, label = sample.load()

        if self.transform:
            image, label = self.transform(image, label)
            assert(image.shape[0] == NUM_INPUT_CHANNELS)
            assert(torch.all(label < NUM_CLASSES))
        
        if isinstance(image, torch.Tensor):
            assert(not torch.any(torch.isnan(image)))
        return image, label
    
    def filename(self, filepath):
        return path.splitext(path.basename(filepath))[0]

    def augmented_to_number_and_variant(self, filepath):
        augmentation_name = self.filename(filepath)
        (variant, image_number) = tuple(augmentation_name.rsplit("_"))
        print(variant, image_number)

    def augmented_is_positive_sample(self, filepath):
        augmentation_name = self.filename(filepath)
        return "_" in augmentation_name
    
    def create_positive_samples(self, idx):
        raw_path = self.raw_paths[idx]
        augmented_positive_paths = [self.augmented_positive_paths[variant][idx] for variant in range(self.augmentation_variants)]
        segmented_path = self.segmentation_paths[idx]
        pattern = str(int(path.splitext(path.basename(raw_path))[0].rsplit("_", 2)[-1]))
        assert pattern in raw_path
        assert pattern in segmented_path
        assert all(map(lambda data: pattern in data, augmented_positive_paths))

        raw_sample = [Sample(raw_path, segmented_path)]
        positive_augmented_samples = [Sample(augmented_path, segmented_path) for augmented_path in augmented_positive_paths]
        
        return raw_sample + positive_augmented_samples

    def create_negative_samples(self):
        f = lambda p: "1000000" in p
        segmented_negative_paths = sorted(list(filter(f, self.segmentation_paths)), key=lambda data: int(self.filename(data)))
        assert len(segmented_negative_paths) + len(self.raw_paths) == len(self.segmentation_paths)
        return [Sample(false_example, false_mask, positive=False) for false_example, false_mask in zip(self.augmented_negative_paths, segmented_negative_paths)]    

    def sample_class_histogram(self, n_samples):
        print("[Warn] Sampling histogram takes alot of time")
        samples = np.random.choice(np.arange(len(self)), size=n_samples)
        histogram = np.zeros(NUM_CLASSES)
        for sample in samples:
            label = self.dataset[sample].load_output()
            # For the +1, see https://numpy.org/doc/stable/reference/generated/numpy.histogram.html#Notes
            hist, _ = np.histogram(label, bins=np.arange(NUM_CLASSES + 1))
            histogram += hist
        return histogram / np.sum(histogram)

class DataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./datasplit", batch_size: int = 8, random_seed = 42, num_workers = 15):
        super().__init__()
        self.train_data_dir = path.join(data_dir, "train")
        self.test_data_dir = path.join(data_dir, "test")
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers


    def setup(self, stage: str):
        if stage in ["fit", "validate"]:
            augmenter = Augmenter(height=480, width=640)
            train = ImageDataset(self.train_data_dir, NUM_AUGMENTATION_VARIANTS, transform=augmenter.transform)

            # Split the train dataset into 80% train, 20% validation
            train_size = int(0.8 * len(train))
            validation_size = len(train) - train_size
            self.train_dataset, self.validation_dataset = torch.utils.data.random_split(train, [train_size, validation_size], generator=torch.Generator().manual_seed(self.random_seed))

        if stage == "pred":
            augmenter = Augmenter(height=480, width=640)
            self.test_dataset = ImageDataset(self.test_data_dir, NUM_AUGMENTATION_VARIANTS, transform=augmenter.test_transform)

           
    def train_dataloader(self):
        # pin_memory = True, persistens_workers = True
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        # pin_memory = True, persistens_workers = True
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,)
