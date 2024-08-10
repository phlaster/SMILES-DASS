import os
import tifffile
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import choice

from PIL import Image

PALLETE = [
    [0, 204, 242],
    [230, 0, 77],
    [204, 204, 204],
    [100, 180, 50],
    [180, 230, 77],
    [255, 230, 166],
    [150, 77, 255]
]
CLASSES = ["open water", "settlements", "bare soil", "forest", "grassland"]

def normalize(band):
    return (band - band.min()) / (band.max() - band.min() + 1e-8)
    
def brighten(band, a=2.0, b=0.25):
    return np.clip(a * band + b, 0.0, 1.0)

def decode_one_hot(one_hot_mask):
    return np.argmax(one_hot_mask, axis=-1).astype('uint8')

def convert_old(im_path, r=3, g=2, b=1):
    with rasterio.open(im_path) as fin:
        red = fin.read(r)
        green = fin.read(g)
        blue = fin.read(b)

    red_b = brighten(red, a=0.13, b=0)
    blue_b = brighten(blue, a=0.13, b=0)
    green_b = brighten(green, a=0.13, b=0)

    red_bn = normalize(red_b)
    green_bn = normalize(green_b)
    blue_bn = normalize(blue_b)

    return np.dstack((blue_b, green_b, red_b)), np.dstack((red_bn, green_bn, blue_bn))

def plot_data(image_path, mask_path, r=3, g=2, b=1):
    plt.figure(figsize=(12, 12))
    pal = [value for color in PALLETE for value in color]
    
    plt.subplot(1, 2, 1)
    _, img = convert_old(image_path, r=r, g=g, b=b)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    mask = tifffile.imread(mask_path)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(pal)
    plt.imshow(mask)    
    
    plt.show();


    
