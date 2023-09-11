import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from ..configuration.constants import DOWNSCALE_FACTOR

class Augmenter:
    def __init__(self, height, width):
        self.__transform = A.Compose([
            A.Resize(height // DOWNSCALE_FACTOR, width // DOWNSCALE_FACTOR, interpolation=0),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(p=1, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
            # A.ChannelShuffle(p=0.5),

            A.RGBShift(r_shift_limit=0.2, g_shift_limit=0.2, b_shift_limit=0.2, p=1),
            A.Blur(),
            A.RandomBrightnessContrast(p=0.8),
            A.RandomSunFlare(src_radius=100),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=5, p=0.8),
            A.GaussNoise(p=1, var_limit=0.05),
            A.Compose([
                A.Rotate(limit=[-20,20], p=1, crop_border=True),
                A.Resize(height, width)
            ], p=0.5),
            # A.Lambda(name="Sun patches", image=random_shape_func_images, p=0.5),
            ToTensorV2()
        ])
        
        self.__test_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=1),
            A.Compose([
                A.Rotate(limit=[-20,20], p=1, crop_border=True),
                A.Resize(height, width)
            ], p=1),
            ToTensorV2()
        ])

    def transform(self, image, mask):
        transformed = self.__transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"]

    def test_transform(self, image, mask):
        transformed = self.__test_transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"]