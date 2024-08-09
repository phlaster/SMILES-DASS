import numpy as np
import os
import tifffile
import torch

def get_filenames_in_dir(path):
    filenames = np.array([
        filename.split('.')[0]
        for filename in os.listdir(path)
        if os.path.isfile(os.path.join(path, filename))
    ])
    return filenames

def tif_info(filename):
    with tifffile.TiffFile(filename) as tif:
        img = tif.asarray()
        
        ndim = img.ndim
        shape = img.shape
        
        if ndim == 2:
            num_bands = 1
        elif ndim == 3:
            num_bands = shape[0] if shape[2] <= 4 else shape[2]
        else:
            num_bands = "Unknown"
        
        x_resolution = tif.pages[0].tags.get('XResolution', None)
        y_resolution = tif.pages[0].tags.get('YResolution', None)
        resolution_unit = tif.pages[0].tags.get('ResolutionUnit', None)
        
        if x_resolution:
            x_resolution = x_resolution.value[0] / x_resolution.value[1]
        if y_resolution:
            y_resolution = y_resolution.value[0] / y_resolution.value[1]
        if resolution_unit:
            resolution_unit = resolution_unit.value
        
        dtype = img.dtype
        bit_depth = dtype.itemsize * 8
        
        if num_bands == "Unknown":
            unique_colors = "Unknown"
        else:
            unique_colors = []
            for i in range(num_bands):
                if num_bands == 1:
                    band = img
                elif shape[2] <= 4:
                    band = img[:, :, i]
                else:
                    band = img[i]
                unique_colors.append(len(np.unique(band)))
        
        summary = f"""
Filename: {filename}
Dimensions: {ndim}D
Shape: {shape}
Number of bands: {num_bands}
Resolution: {x_resolution}*{y_resolution} {resolution_unit}
Data type: {dtype}
Bit depth: {bit_depth}
Unique colors per band: {unique_colors}
        """
        return summary
    
    
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device

def metric_accuracy(mask_true, mask_predicted, num_classes):
    accuracy = np.zeros(num_classes)
    for class_id in range(num_classes):
        correct = np.sum((mask_true == class_id) & (mask_predicted == class_id))
        total = np.sum(mask_true == class_id)
        accuracy[class_id] = correct / total if total > 0 else 0
    return accuracy

def metric_recall(mask_true, mask_predicted, num_classes):
    recall = np.zeros(num_classes)
    for class_id in range(num_classes):
        true_positives = np.sum((mask_true == class_id) & (mask_predicted == class_id))
        false_negatives = np.sum((mask_true == class_id) & (mask_predicted != class_id))
        recall[class_id] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall

def metric_precision(mask_true, mask_predicted, num_classes):
    precision = np.zeros(num_classes)
    for class_id in range(num_classes):
        true_positives = np.sum((mask_true == class_id) & (mask_predicted == class_id))
        false_positives = np.sum((mask_true != class_id) & (mask_predicted == class_id))
        precision[class_id] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision

def metric_f1(mask_true, mask_predicted, num_classes):
    precision = metric_precision(mask_true, mask_predicted, num_classes)
    recall = metric_recall(mask_true, mask_predicted, num_classes)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0  # Handle cases where precision + recall is 0
    return f1
