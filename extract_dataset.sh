#!/bin/bash

TRAIN_SAMPLES=150
VALID_SAMPLES=30
TEST_SAMPLES=50

# Function to copy a specified number of files
copy_files() {
    local src_dir=$1
    local dest_dir=$2
    local num_files=$3

    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"

    # Copy the specified number of files
    find "$src_dir" -type f | head -n "$num_files" | while read -r file; do
        rsync -a "$file" "$dest_dir/"
    done
}

# Define the source and destination paths
src_base="datasets"
dest_base="datasets_small"

# Copy files for Russia
copy_files "$src_base/Russia/test/images" "$dest_base/Russia/test/images" $TEST_SAMPLES
copy_files "$src_base/Russia/test/masks" "$dest_base/Russia/test/masks" $TEST_SAMPLES
copy_files "$src_base/Russia/train/images" "$dest_base/Russia/train/images" $TRAIN_SAMPLES
copy_files "$src_base/Russia/train/masks" "$dest_base/Russia/train/masks" $TRAIN_SAMPLES
copy_files "$src_base/Russia/valid/images" "$dest_base/Russia/valid/images" $VALID_SAMPLES
copy_files "$src_base/Russia/valid/masks" "$dest_base/Russia/valid/masks" $VALID_SAMPLES

# Copy files for USA
copy_files "$src_base/USA/train/images" "$dest_base/USA/train/images" $TRAIN_SAMPLES
copy_files "$src_base/USA/train/masks" "$dest_base/USA/train/masks" $TRAIN_SAMPLES
copy_files "$src_base/USA/valid/images" "$dest_base/USA/valid/images" $VALID_SAMPLES
copy_files "$src_base/USA/valid/masks" "$dest_base/USA/valid/masks" $VALID_SAMPLES