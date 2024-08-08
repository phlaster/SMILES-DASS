import os
import re
import pandas as pd
import numpy as np
import tifffile
from tqdm import tqdm
from random import choice, sample

import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from albumentations import Compose, RandomRotate90, RandomBrightnessContrast, ChannelDropout

import matplotlib.pyplot as plt

class SpectralIndex:
    def __init__(self, formula):
        self.formula = formula
        self.available_sensors = {
            "B": 0, "G": 1, "R": 2, "RE1": 3, "RE2": 4, "RE3": 5, "N": 6, "N2": 7, "S1": 8, "S2": 9
        }
        tokens = re.findall(r'\b[A-Za-z]+\b', formula)

        unknown_tokens = [token for token in tokens if token not in self.available_sensors.keys()]
        if unknown_tokens:
            raise ValueError(f"Unknown tokens in formula: {unknown_tokens}")

    def apply(self, array):
        band_vars = {key: array[val] for key, val in self.available_sensors.items()}
        try:
            index_result = eval(self.formula, {}, band_vars)
        except Exception as e:
            raise ValueError(f"Error in evaluating the formula: {e}")

        return index_result
        

class LandcoverDataset(Dataset):
    def __init__(self, img_path, mask_path, n_random=None, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.file_names = [
            f for f in os.listdir(img_path) if f.endswith('.tif')
        ] if not n_random else sample([
            f for f in os.listdir(img_path) if f.endswith('.tif')
        ], n_random)
        self.transform = Compose([
            RandomRotate90(),
            RandomBrightnessContrast(p=0.1),
            ChannelDropout(channel_drop_range=(1, 2), fill_value=0, p=0.5)
        ]) if transform is None else transform

        self.images = [self._load_and_preprocess_image(f) for f in tqdm(self.file_names, desc='Loading and preprocessing images')]
        self.masks = [self._load_and_preprocess_mask(f) for f in tqdm(self.file_names, desc='Loading and preprocessing masks')]

    def _load_and_preprocess_image(self, filename):
        image = tifffile.imread(os.path.join(self.img_path, filename))
        image = self.normalize(image)
        return np.moveaxis(image, -1, 0)  # Convert (H, W, C) to (C, H, W) for PyTorch

    def _load_and_preprocess_mask(self, filename):
        mask = tifffile.imread(os.path.join(self.mask_path, filename))
        return mask  # Directly return the class indices

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        if self.transform:
            augmented = self.transform(image=image.transpose(1, 2, 0), mask=mask)
            image = augmented['image'].transpose(2, 0, 1)
            mask = augmented['mask']
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

    def normalize(self, image):
        image = image.astype(np.float32)
        min_val = np.min(image, axis=(0, 1))
        max_val = np.max(image, axis=(0, 1))
        normalized = (image - min_val) / (max_val - min_val + 1e-8)
        return normalized

    def load_data(self, batch_size, shuffle=True, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def getinfo(self):
        num_images = len(self.images)
        img_shape = self.images[0].shape if num_images > 0 else "N/A"
        mask_shape = self.masks[0].shape if num_images > 0 else "N/A"
        img_dtype = self.images[0].dtype if num_images > 0 else "N/A"
        mask_dtype = self.masks[0].dtype if num_images > 0 else "N/A"

        print("Dataset Information:")
        print(f"Number of samples: {num_images}")
        print(f"Image Shape: {img_shape}")
        print(f"Mask Shape: {mask_shape}")
        print(f"Image Data Type: {img_dtype}")
        print(f"Mask Data Type: {mask_dtype}")
        print(f"Transformations: {self.transform}")


    def plot_sample(self, n, r=2, g=1, b=0, index=""):
        PALETTE = [
            [0, 204, 242],
            [230, 0, 77],
            [204, 204, 204],
            [100, 180, 50],
            [180, 230, 77]
        ]

        # Apply the palette to the mask
        def apply_palette(mask, palette):
            palette_array = np.array(palette)
            colored_mask = palette_array[mask]
            return colored_mask

        def adjust(picture, a=3.5, b=0.0):
            # Naive gamma correction
            return (picture**0.3 + b - np.min(picture)) / (np.max(picture) - np.min(picture) + 1e-8)

        image, mask = self[n]
        image = image.cpu().numpy().transpose(1, 2, 0)
        mask = mask.cpu().numpy()

        if index:
            si = SpectralIndex(index)
            applied = si.apply(image.transpose(2, 0, 1))  # (C, H, W) format expected
            rgb_image = (applied - np.min(applied)) / (np.max(applied) - np.min(applied) + 1e-8)
        else:
            rgb_image = adjust(image[..., [r, g, b]])

        colored_mask = apply_palette(mask, PALETTE)

        # Plot image and mask
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(rgb_image, cmap='gray' if index else None)
        axs[0].set_title('Index Image' if index else 'RGB Image')
        axs[0].axis('off')

        axs[1].imshow(colored_mask)
        axs[1].set_title('Mask')
        axs[1].axis('off')

        plt.show()