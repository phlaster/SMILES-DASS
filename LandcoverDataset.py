from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import tifffile
from random import choice

class LandcoverDataset(Dataset):
    def __init__(self, img_path, mask_path, n1, n2):
        self.img_path = img_path
        self.mask_path = mask_path
        self.file_names = self.filenames_in_dir(img_path)[n1:n2]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image = tifffile.imread(self.img_path + self.file_names[idx] + '.tif')
        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
        image = self._image_padding(image).astype(np.float32)
        
        mask = tifffile.imread(self.mask_path + self.file_names[idx] + '.tif')
        mask = self._mask_padding(mask)

        return image, mask
    
    
    def _image_padding(self, image, target_size=512):
        height, width = image.shape[1:3]
        pad_height = max(0, target_size - height)
        pad_width = max(0, target_size - width)
        padded_image = np.pad(image, ((0, 0), (0, pad_height),
                                      (0, pad_width)), mode='reflect')
        return padded_image


    def _mask_padding(self, mask, target_size=512):
        height, width = mask.shape
        pad_height = max(0, target_size - height)
        pad_width = max(0, target_size - width)
        padded_mask = np.pad(mask, ((0, pad_height), (0, pad_width)),
                             mode='reflect')
        return padded_mask
        
    def rand_samp_names(self):
        name = choice(self.file_names)
        img_name = f"{self.img_path}/{name}.tif"
        mask_name = f"{self.mask_path}/{name}.tif"
        
        return img_name, mask_name
        
    @staticmethod
    def filenames_in_dir(img_path):
        name = []
        # given a directory iterates over the files
        for _, _, filenames in os.walk(img_path):
            for filename in filenames:
                f = filename.split('.')[0]
                name.append(f)

        df =  pd.DataFrame({'id': name}, index = np.arange(0, len(name))
                           ).sort_values('id').reset_index(drop=True)
        df = df['id'].values

        return np.delete(df, 0)