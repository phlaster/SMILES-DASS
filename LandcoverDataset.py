import multiprocessing
from functools import partial
import os
from random import choice, sample

from SpectralIndex import SpectralIndex
from utils import *

import numpy as np
import tifffile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms.v2 import Compose, RandomVerticalFlip, RandomHorizontalFlip
from Transforms import RandomBrightnessContrast, ChannelDropout
import matplotlib.pyplot as plt
from skimage import exposure

PALETTE = [
    [0, 204, 242],
    [230, 0, 77],
    [204, 204, 204],
    [100, 180, 50],
    [180, 230, 77]
]
CLASS_NAMES = [
    "Water",
    "Urban",
    "Bare soil",
    "Forest",
    "Grassland"
]


class LandcoverDataset(Dataset):
    def __init__(self, img_path, mask_path, batch_size, n_random=None, transforms=None, indexes=None, n_classes=5):
        self.num_classes = n_classes
        self.img_path = img_path
        self.mask_path = mask_path
        self.file_names = [f for f in os.listdir(img_path) if f.endswith('.tif') ]
        if n_random:
            self.file_names = sample(self._filter_by_size(self.file_names, img_path, (512, 512)), n_random)
        else:
            self.file_names = self._filter_by_size(self.file_names, img_path, (512, 512))
        
        self.indexes = indexes if indexes is not None else []
        self.spectral_indices = [SpectralIndex(index) for index in self.indexes]
        self.transforms = Compose([
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            ChannelDropout(channel_drop_range=(1, 2), fill_value=0, p=0.3) #, protect_last=len(self.indexes))
        ]) if transforms is None else transforms
        
        with multiprocessing.Pool(processes=os.cpu_count()) as pool: # otherwise too slow
            self.images = list(tqdm(
                pool.imap(self._load_and_preprocess_image, self.file_names),
                total=len(self.file_names),
                desc='Loading and preprocessing images'))
        self.masks = [self._load_and_preprocess_mask(f) for f in tqdm(self.file_names, desc='Loading and preprocessing masks')]
        self.loader = DataLoader(self, batch_size=batch_size, num_workers=os.cpu_count())

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        image = torch.tensor(image, dtype=torch.float32)
        if self.transforms:
            image = self.transforms(image)
        class_counts = torch.bincount(torch.tensor(mask).view(-1), minlength=self.num_classes)        
        return image.clone().detach(), torch.tensor(mask, dtype=torch.long), class_counts
    

    def _filter_by_size(self, file_names, img_path, target_size):
        filtered_files = []
        for filename in tqdm(file_names, desc='Filtering files with wrong dimensions:'):
            try:
                with tifffile.TiffFile(os.path.join(img_path, filename)) as tif:
                    if tif.pages[0].shape[:2] == target_size:
                        filtered_files.append(filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        return filtered_files

        
    def _load_and_preprocess_image(self, filename):
        image = tifffile.imread(os.path.join(self.img_path, filename))
        image = self._normalize(image)
        image = np.moveaxis(image, -1, 0)  # Convert (H, W, C) to (C, H, W) for PyTorch
        
        for si in self.spectral_indices:
            noise = np.random.normal(loc=0, scale=1e-8, size=image.shape)
            index_band = si.apply(image + noise)
            index_band = self._normalize(index_band[np.newaxis, ...])  # Add channel dimension and normalize
            image = np.concatenate((image, index_band), axis=0)
        
        return image

    def _load_and_preprocess_mask(self, filename):
        mask = tifffile.imread(os.path.join(self.mask_path, filename))
        return mask

    def _normalize(self, image):
        image = image.astype(np.float32)
        min_val = np.min(image, axis=(0, 1), keepdims=True)
        max_val = np.max(image, axis=(0, 1), keepdims=True)
        normalized = (image - min_val) / (max_val - min_val + 1e-8)
        return normalized

    
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
        print(f"Transformations: {self.transforms}")
        print(f"Spectral Indices: {self.indexes}")

    def plot_sample(self, n, r=2, g=1, b=0, index=""):
        def apply_palette(mask, palette):
            palette_array = np.array(palette)
            colored_mask = palette_array[mask]
            return colored_mask
       
        image, mask, _ = self[n]
        image = image.cpu().numpy().transpose(1, 2, 0)

        if index:
            si = SpectralIndex(index)
            transposed = image.transpose(2, 0, 1) # (C, H, W) format expected
            noise = np.random.normal(loc=0, scale=1e-8, size=transposed.shape)
            applied = si.apply(transposed + noise)  
            rgb_image = self._normalize(applied)
        else:
            rgb_image = exposure.rescale_intensity(image[..., [r, g, b]], out_range=(0, 1)) ** 0.3

        colored_mask = apply_palette(mask, PALETTE)

        # Plot image and mask
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(rgb_image, cmap='gray' if index else None)
        axs[0].set_title('Index Image' if index else 'RGB Image')
        axs[0].axis('off')

        axs[1].imshow(colored_mask)
        axs[1].set_title('Mask')
        axs[1].axis('off')

        plt.show()
        
    def plot_prediction(self, model, n, r=2, g=1, b=0, index=""):
        def apply_palette(mask, palette):
            palette_array = np.array(palette)
            return palette_array[mask]

        device = get_device()
        model.model.to(device)
        model.model.eval()

        image, mask, _ = self[n]
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        mask_np = mask.cpu().numpy()

        if index:
            si = SpectralIndex(index)
            transposed = image.transpose(2, 0, 1) # (C, H, W) format expected
            noise = np.random.normal(loc=0, scale=1e-8, size=transposed.shape)
            applied = si.apply(transposed + noise)  
            rgb_image = (applied - np.min(applied)) / (np.max(applied) - np.min(applied) + 1e-8)
        else:
            rgb_image = exposure.rescale_intensity(image_np[..., [r, g, b]], out_range=(0, 1)) ** 0.3

        colored_mask = apply_palette(mask_np, PALETTE)

        # Predict the mask
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            predicted = model.model(image)
            _, predicted_mask = torch.max(predicted, 1)
            predicted_mask = predicted_mask.squeeze(0).cpu().numpy()

        colored_predicted_mask = apply_palette(predicted_mask, PALETTE)

        num_classes = len(PALETTE)
        accuracy_scores = metric_accuracy(mask_np, predicted_mask, num_classes)
        accuracy = round(np.mean(accuracy_scores), 2)

        # Plot image, mask, predicted mask, and accuracy scores
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        axs[0].imshow(rgb_image, cmap='gray' if index else None)
        axs[0].set_title(f'Index Image {n}' if index else f'RGB Image {n}')
        axs[0].axis('off')

        axs[1].imshow(colored_mask)
        axs[1].set_title(f'Mask {n}')
        axs[1].axis('off')

        axs[2].imshow(colored_predicted_mask)
        axs[2].set_title(f'Predicted Mask {n}')
        axs[2].axis('off')

        # Bar plot for accuracy scores
        colors = [tuple(c / 255 for c in color) for color in PALETTE]
        axs[3].bar(range(num_classes), accuracy_scores, color=colors)
        axs[3].set_ylim(0, 1)
        axs[3].set_ylabel('Accuracy Score')
        axs[3].set_title(f'Class Accuracy Scores, mean: {accuracy}')
        axs[3].set_xticks(range(num_classes))
        axs[3].set_xticklabels(CLASS_NAMES)

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        plt.show()
        
        
    def rand_samp_names(self):
        name = choice(self.file_names)
        img_name = os.path.join(self.img_path, name)
        mask_name = os.path.join(self.mask_path, name)
        return img_name, mask_name