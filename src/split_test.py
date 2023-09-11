from .dataloader.sample import Sample
from .configuration.constants import NUM_AUGMENTATION_VARIANTS
import glob
from os import path
from itertools import chain
import numpy as np


class DatasetManager:
    def filename(self, filepath):
            return path.splitext(path.basename(filepath))[0]

    def __init__(self, dataset_path, n_variants):
        self.n_variants = n_variants
        self.raw_paths = sorted(glob.glob(path.join(dataset_path, "raw", "*.png")), key=lambda data: int(self.filename(data)))
        self.augmented_positive_paths = [
            sorted(glob.glob(path.join(dataset_path, "augmented", f"{variant}_*.png")))
            for variant in range(n_variants)
        ]
        self.augmented_negative_paths = sorted(glob.glob(path.join(dataset_path, "augmented", "1000000*.png")))
        self.segmentation_paths = sorted(glob.glob(path.join(dataset_path, "segmentation", "*.png")), key=lambda data: int(self.filename(data)))
        self.dataset = (
            [self.create_positive_samples(i) for i in range(len(self.raw_paths))] 
            + self.create_negative_samples()
        )

    def create_positive_samples(self, idx):
        raw_path = self.raw_paths[idx]
        augmented_positive_paths = [self.augmented_positive_paths[variant][idx] for variant in range(self.n_variants)]
        segmented_path = self.segmentation_paths[idx]
        pattern = f"{idx}"
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

def random_sample(percent, total, rng=np.random.default_rng(seed=42)):
    arr = np.zeros(total)
    arr[:int(percent*total)] = 1
    rng.shuffle(arr)
    return np.where(arr == 1)[0]

def main():
    manager = DatasetManager("./dataset", NUM_AUGMENTATION_VARIANTS)
    print(len(manager.raw_paths))
    print(len(manager.dataset))

    def replacer_test(p):
        return p.replace("dataset", "datasplit/test")
    def replacer_train(p):
        return p.replace("dataset", "datasplit/train")

    indices = random_sample(0.1, len(manager.dataset))
    assert len(set(indices)) == len(indices)

    total_length = len(manager.dataset)
    print(f"Test data size: {len(indices)}/{total_length}")
    for i in range(total_length):
        if np.any(indices==i):
            print(f"{i}/{total_length}: Test")
            current_replacer = replacer_test
        else:
            print(f"{i}/{total_length}: Train")
            current_replacer = replacer_train
        
        samples = manager.dataset[i]
        if isinstance(samples, list):
            for sample in samples:
                sample.move_to(current_replacer)
        else:
            samples.move_to(current_replacer)



if __name__ == "__main__":
    main()