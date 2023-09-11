import imgaug
from imgaug import augmenters as iaa
import cv2
import numpy as np

class Augmenter:
    def __init__(self):
        self.augmenter = iaa.Sequential(
                [
                    iaa.Fliplr(0.5),
                    iaa.Crop(percent=(0.0,0.2)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255.0)), # add gaussian noise to images
                    iaa.LinearContrast((0.2, 1.4)),
                    iaa.Sometimes(0.7, iaa.Lambda(
                        func_images=self.random_shape_func_images,
                        func_heatmaps=self.random_shape_func_heatmaps,
                        func_keypoints=self.random_shape_func_keypoints
                    )),
                    iaa.SomeOf((0, 2),
                        [
                            iaa.Multiply((0.8, 1.2), per_channel=False),
                            iaa.BlendAlphaFrequencyNoise(
                                exponent=(-2, 2),
                                foreground=iaa.Multiply((0.5, 4.0)),
                                background=iaa.LinearContrast((0.5, 2.0))
                            ),
                            iaa.MotionBlur(k=(3, 5)),
                            iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                            # Requires uint8 datatype
                            # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                            iaa.LinearContrast((0.5, 1.5), per_channel=0.5) # improve or worsen the contrast
                        ],
                        random_order=True
                    )
                ],
                random_order=False
            )

    def transform(self, image, labels):
        segmentation_map = imgaug.SegmentationMapsOnImage(labels, shape=labels.shape)
        image, labels = self.augmenter(image=image, segmentation_maps=segmentation_map)

        return image, labels.get_arr()

    def random_shape_func_images(self, images, random_state, parents, hooks):
        result = []
        try:
            for i, image in enumerate(images):
                img_lights = np.ones(image.shape, dtype=np.float32)
                for j in range(np.random.randint(1,15)): # up to n polygones
                    img_light = np.zeros(image.shape, dtype=np.uint8)
                    pts = np.random.randint(0, max(image.shape), (np.random.randint(3, 6), 2)) #each polygone between 3 and 6 points
                    pts = pts.reshape((-1,1,2))
                    cv2.fillPoly(img_light,[pts],(1, 1, 1))
                    img_lights += img_light.astype(np.float32) * np.random.uniform(0.1, 2.0)
                img_lights_normalized = img_lights / min(1, img_lights.max())
                result.append(np.clip(image.astype(np.float32) * (img_lights_normalized + 1), 0.0, 255.0).astype(np.float32))
        except e:
            print(f"error in augment: {e}")
            return images
        return result

    def random_shape_func_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def random_shape_func_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images