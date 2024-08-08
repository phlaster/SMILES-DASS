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

    

def match_scores(
    mask_raw,
    prediction,
    classes=CLASSES
):
    class_scores = {}
    
    for i in range(len(classes)):
        cls_mask = mask_raw == i
        cls_prediction = prediction == i
        
        correct_pixels = np.sum(cls_mask & cls_prediction)
        total_pixels = np.sum(cls_mask)
        
        if total_pixels > 0:
            correct_percentage = (correct_pixels / total_pixels) * 100
        else:
            correct_percentage = 0
        class_scores[classes[i]] = correct_percentage
    return class_scores


def compare_prediction(img_path, mask_path, prediction):    
    fig = plt.figure(figsize=(24, 12))

    gs = fig.add_gridspec(1, 4)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    pal = [value for color in PALLETE for value in color]
    
    # 1
    _, img = convert_old(img_path)
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 2
    mask_raw = tifffile.imread(mask_path)
    mask = Image.fromarray(mask_raw).convert('P')
    mask.putpalette(pal)
    ax2.imshow(mask)    
    ax2.set_title('Ground Truth Mask')
    ax2.axis('off')  # Hide axes
    
    # 3
    predicted_mask_img = Image.fromarray(prediction.astype('uint8')).convert('P')
    predicted_mask_img.putpalette(pal)
    ax3.imshow(predicted_mask_img)
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    
    # Fourth subplot for correctness scores
    class_scores = match_scores(mask_raw, prediction)
    class_colors = {CLASSES[i]: np.array(pal[i*3:(i+1)*3]) / 255 for i in range(len(CLASSES))}
    
    max_height = max([score for score in class_scores.values()])

    for cls, score in class_scores.items():
        ax4.barh(cls, score, color=class_colors[cls])
        ax4.text(score, cls, f"{score:.2f}%", va='center')
    
    ax4.set_title('Accuracy Scores')
    ax4.set_xlim(0, 100)
    ax4.set_yticks(list(class_scores.keys()))
    ax4.set_yticklabels([f'{cls}' for cls in class_scores.keys()])

    # Adjust the aspect ratio of the barplot to make it square
    ax4.set_aspect(1/ax4.get_data_ratio())

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.show()

    
