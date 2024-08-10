import numpy as np
import torch

class RandomBrightnessContrast:
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, p=0.1):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            brightness_factor = 1 + torch.rand(1) * self.brightness_limit * 2 - self.brightness_limit
            contrast_factor = 1 + torch.rand(1) * self.contrast_limit * 2 - self.contrast_limit
            img = torch.clamp(img * brightness_factor, 0, 1)
            mean = img.mean()
            img = (img - mean) * contrast_factor + mean
        return img


class ChannelDropout:
    def __init__(self, channel_drop_range=(1, 2), fill_value=0, p=0.5, protect_last=0):
        self.channel_drop_range = channel_drop_range
        self.fill_value = fill_value
        self.p = p
        self.protect_last = protect_last

    def __call__(self, img):
        if torch.rand(1) < self.p:
            num_channels = img.shape[0]
            droppable_channels = num_channels - self.protect_last
            max_drop = min(self.channel_drop_range[1], droppable_channels)
            adjusted_drop_range = (min(self.channel_drop_range[0], max_drop), max_drop)
            num_channels_to_drop = np.random.randint(adjusted_drop_range[0], adjusted_drop_range[1] + 1)
            channels_to_drop = np.random.choice(droppable_channels, num_channels_to_drop, replace=False)
            img[channels_to_drop, :, :] = self.fill_value
        return img