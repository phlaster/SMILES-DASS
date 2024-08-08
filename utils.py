import numpy as np
import os
import tifffile

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