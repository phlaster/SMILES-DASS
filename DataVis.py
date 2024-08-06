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

def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band - band_min) / ((band_max - band_min)))
    
def brighten(band):
    alpha = 0.13
    beta = 0
    return np.clip(alpha * band + beta, 0, 255)

def convert(im_path, r=3, g=2, b=1):
    with rasterio.open(im_path) as fin:
        red = fin.read(r)
        green = fin.read(g)
        blue = fin.read(b)

    red_b = brighten(red)
    blue_b = brighten(blue)
    green_b = brighten(green)

    red_bn = normalize(red_b)
    green_bn = normalize(green_b)
    blue_bn = normalize(blue_b)

    return np.dstack((blue_b, green_b, red_b)), np.dstack((red_bn, green_bn, blue_bn))

def plot_data(image_path, mask_path, r=3, g=2, b=1):
    plt.figure(figsize=(12, 12))
    pal = [value for color in PALLETE for value in color]
    
    plt.subplot(1, 2, 1)
    _, img = convert(image_path, r=r, g=g, b=b)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    mask = tifffile.imread(mask_path)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(pal)
    plt.imshow(mask)    
    
    plt.show();


def peek_dataset(ds, r=3, g=2, b=1):
    name = choice(ds.filenames_in_dir(ds.img_path))
    img_path = f"{ds.img_path}/{name}.tif"
    mask_path = f"{ds.mask_path}/{name}.tif"
    
    plt.figure(figsize=(12, 12))
    pal = [value for color in PALLETE for value in color]
    
    plt.subplot(1, 2, 1)
    _, img = convert(img_path, r=r, g=g, b=b)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    mask = tifffile.imread(mask_path)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(pal)
    plt.imshow(mask)    
    
    plt.show();
    

def match_scores(mask_raw, prediction):
    class_scores = {}
    class_names = ["open water", "settlements", "bare soil", "forest", "grassland"]
    
    for i in range(5):
        cls_mask = mask_raw == i
        cls_prediction = prediction == i
        
        correct_pixels = np.sum(cls_mask & cls_prediction)
        total_pixels = np.sum(cls_mask)
        
        if total_pixels > 0:
            correct_percentage = (correct_pixels / total_pixels) * 100
        else:
            correct_percentage = 0
        
        class_scores[class_names[i]] = {
            'correct': correct_percentage
        }
    
    return class_scores


def compare_prediction(img_path, mask_path, prediction, r=3, g=2, b=1):    
    fig = plt.figure(figsize=(24, 12))  # Adjusted figure size

    # Define gridspec with ratios for three image columns and one barplot column
    gs = fig.add_gridspec(1, 4)

    # Create subplots using gridspec
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])


    pal = [value for color in PALLETE for value in color]
    
    # First subplot
    _, img = convert(img_path, r=r, g=g, b=b)
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')  # Hide axes

    # Second subplot
    mask_raw = tifffile.imread(mask_path)
    mask = Image.fromarray(mask_raw).convert('P')
    mask.putpalette(pal)
    ax2.imshow(mask)    
    ax2.set_title('Ground Truth Mask')
    ax2.axis('off')  # Hide axes
    
    # Third subplot
    predicted_mask_img = Image.fromarray(prediction.astype('uint8')).convert('P')
    predicted_mask_img.putpalette(pal)
    ax3.imshow(predicted_mask_img)
    ax3.set_title('Predicted Mask')
    ax3.axis('off')  # Hide axes
    
    # Fourth subplot for correctness scores
    class_names = ["open water", "settlements", "bare soil", "forest", "grassland"]
    class_colors = {class_names[cls]: np.array(pal[cls*3:(cls+1)*3]) / 255 for cls in range(5)}
    class_scores = match_scores(mask_raw, prediction)
    
    # Get the maximum height of the bars for aspect adjustment
    max_height = max([score['correct'] for score in class_scores.values()])

    for cls, score in class_scores.items():
        ax4.barh(cls, score['correct'], color=class_colors[cls])
        ax4.text(score['correct'], cls, f"{score['correct']:.2f}%", va='center')
    
    ax4.set_title('Accuracy Scores')
    ax4.set_xlim(0, 100)
    ax4.set_yticks(list(class_scores.keys()))
    ax4.set_yticklabels([f'{cls}' for cls in class_scores.keys()])

    # Adjust the aspect ratio of the barplot to make it square
    ax4.set_aspect(1/ax4.get_data_ratio())

    plt.subplots_adjust(wspace=0.1, hspace=0.5)  # Adjust space between subplots
    plt.show()


