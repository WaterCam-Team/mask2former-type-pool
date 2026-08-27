import random
import numpy as np
import cv2
from detectron2.data import transforms as T
from detectron2.data.transforms import Transform


class WaterTransform(Transform):
    """A lightweight wrapper so we can return a valid Detectron2 Transform."""
    def __init__(self, aug_img):
        super().__init__()
        self.aug_img = aug_img

    def apply_image(self, img):
        return self.aug_img
    
    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_coords(self, coords):
        # no geometric change
        return coords


class WaterAug(T.Augmentation):
    def __init__(self, brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 p_glare=0.15, p_night=0.12, p_muddy=0.12, p_reflection=0.15):
        super().__init__()
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p_glare = p_glare
        self.p_glare = p_glare
        self.p_night = p_night
        self.p_muddy = p_muddy
        self.p_reflection = p_reflection

    def get_transform(self, image):
        img = image.copy()
        img = self.brightness(img)
        mode = random.randint(0,1)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)
        r = random.random()
        if r < self.p_glare:
            img = self._glare(img)
        elif r < self.p_glare + self.p_night:
            img = self._night(img)
        elif r < self.p_glare + self.p_night + self.p_muddy:
            img = self._muddy(img)
        elif r < self.p_glare + self.p_night + self.p_muddy + self.p_reflection:
            img = self._reflection(img)
        # Return valid Transform, not deprecated LambdaTransform
        return WaterTransform(img)

    # === individual augmentations ===

    def _glare(self, img):
        img = np.clip(img.astype(np.float32) * random.uniform(1.05, 1.25) + random.uniform(5, 25), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(255 * (hsv[..., 2] / 255) ** random.uniform(0.7, 0.9), 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _night(self, img):
        img = np.clip(img.astype(np.float32) * random.uniform(0.6, 0.9) + random.uniform(-15, 10), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.6, 0.9), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.6, 0.85), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _muddy(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + random.randint(3, 10)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 0.95), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _reflection(self, img):
        img = np.clip(img.astype(np.float32) * random.uniform(1.05, 1.15), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + random.randint(-3, 3)) % 180
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.95, 1.05), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def convert(self, img, alpha=1, beta=0):
        out = img.astype(np.float32) * alpha + beta
        out = np.clip(out, 0, 255)
        return out.astype(np.uint8)
    
    def brightness(self, img):
        if random.randint(0, 1):
            beta = random.uniform(-self.brightness_delta, self.brightness_delta)
            return self.convert(img, beta=beta)
        return img

    def contrast(self, img):
        if random.randint(0, 1):
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            return self.convert(img, alpha=alpha)
        return img

    def saturation(self, img):
        if random.randint(0, 1):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            alpha = random.uniform(self.saturation_lower, self.saturation_upper)
            hsv[..., 1] = np.clip(hsv[..., 1] * alpha, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if random.randint(0, 1):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
            delta = random.randint(-self.hue_delta, self.hue_delta)
            hsv[..., 0] = (hsv[..., 0] + delta) % 180
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img
