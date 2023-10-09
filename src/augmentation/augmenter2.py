import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2

from ..configuration.constants import DOWNSCALE_FACTOR, CLASS_TO_LABEL

class Augmenter:
    def __init__(self, height, width):
        self.height = height // DOWNSCALE_FACTOR
        self.width  = width  // DOWNSCALE_FACTOR

        self.line_class_idx = CLASS_TO_LABEL.index("Line")
        try:
            self.field_class_idx = CLASS_TO_LABEL.index("Field")
        except ValueError:
            self.field_class_idx = CLASS_TO_LABEL.index("Other")

        self.__transform = A.Compose([
            # A.RandomGamma(p=1, gamma_limit=0.0001),
            A.Rotate(limit=[-20,20], p=0.2, crop_border=True),
            A.OneOrOther(transforms=[
                A.RandomSizedCrop(min_max_height=[self.height // 8, self.height // 2], height=self.height, width=self.width, w2h_ratio=self.width/self.height, p=1),
                A.Resize(self.height, self.width, interpolation=0),
            ], p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5, contrast_limit=0.3),
            A.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.4),
            # A.ChannelShuffle(p=0.1),

            A.RGBShift(r_shift_limit=0.1, g_shift_limit=0.3, b_shift_limit=0.1, p=0.5),
            A.Blur(blur_limit=[1, 3]),
            A.RandomSunFlare(src_radius=100),
            A.Lambda(image=self.add_bright_spots, p=0.5),
            A.RandomShadow(num_shadows_lower=3, num_shadows_upper=9, p=0.5),
            A.GaussNoise(p=0.5, var_limit=0.02),
            ToTensorV2()
        ])
        
        self.__test_transform = A.Compose([
            A.Rotate(limit=[-20,20], p=1, crop_border=True),
            A.Resize(self.height, self.width, interpolation=0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=1),
            ToTensorV2()
        ])

        self.__real_transform = A.Compose([
            A.Resize(self.height, self.width, interpolation=0),
            ToTensorV2()
        ])

    def transform(self, image, mask):
        # image, mask = self.decrease_line_contrast(image, mask, random_divide = 1 + np.random.rand() * 2)
        transformed = self.__transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"]

    def test_transform(self, image, mask):
        transformed = self.__test_transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"]

    def real_transform(self, image):
        transformed = self.__real_transform(image=image)
        return transformed["image"]

    def decrease_line_contrast(self, image, mask, random_divide):
        image[mask == self.line_class_idx] /= random_divide
        return image, mask

    def add_bright_spots(self, image, **kwargs):
        max_number_bright_spots = 7
        number_bright_spots = np.random.randint(3, max_number_bright_spots + 1)


        for _index in range(number_bright_spots):
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            vertex = []
            for _ in range(4):
                vertex.append((np.random.randint(0, self.width), np.random.randint(0, self.height)))

            vertices = np.array([vertex], dtype=np.int32)
            
            cv2.fillPoly(mask, vertices, 255)
            increase_factor = np.random.uniform(1.0, 5.0)
            image[:,:,0][mask == 255] = image[:,:,0][mask == 255] * increase_factor
            image[:,:,1][mask == 255] = image[:,:,1][mask == 255] * increase_factor
            image[:,:,2][mask == 255] = image[:,:,2][mask == 255] * increase_factor
        assert not np.any(np.isnan(image))

        return np.clip(image, 0.0, 1.0)
        
