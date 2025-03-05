import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models,transforms,datasets
import time
import cv2
from collections import Counter

'''
#split dataset into test set and train/validation set
test_file = "CVDL_G11/CVDL_Bsc-main/labs/01_image_classification/annotations/test.txt"
trainval_file = "CVDL_G11/CVDL_Bsc-main/labs/01_image_classification/annotations/trainval.txt"

def print_head(file_path, n=10):
    with open(file_path, "r") as file:
        lines = file.readlines()
    print("".join(lines[:n]))

print("First 10 lines of test.txt:")
print_head(test_file)

print("\nFirst 10 lines of trainval.txt:")
print_head(trainval_file)
'''

'''
image_folder = "CVDL_G11/CVDL_Bsc-main/labs/01_image_classification/images"

# Get list of images
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

print(f"Total images: {len(image_files)}")

image_sizes = []
aspect_ratios = []
color_compositions = []

for img_name in image_files[:100]:  # Check first 100 images
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    
    if img is not None:
        h, w, c = img.shape  # h: height, w: width, c: channels
        
        # Resolution
        image_sizes.append((w, h))
        
        # Aspect Ratio
        aspect_ratios.append(w / h)
        
        # Color Composition: Calculate mean RGB values
        mean_color = np.mean(img, axis=(0, 1))  # Mean across height and width
        color_compositions.append(mean_color)

# Convert sizes to NumPy array
image_sizes = np.array(image_sizes)
mean_size = np.mean(image_sizes, axis=0)
min_size = np.min(image_sizes, axis=0)
max_size = np.max(image_sizes, axis=0)

# Calculate Aspect Ratio statistics
mean_aspect_ratio = np.mean(aspect_ratios)
min_aspect_ratio = np.min(aspect_ratios)
max_aspect_ratio = np.max(aspect_ratios)

# Calculate mean color composition
mean_color_composition = np.mean(color_compositions, axis=0)

print(f"Resolution Analysis:")
print(f"Mean resolution: {mean_size}")
print(f"Min resolution: {min_size}")
print(f"Max resolution: {max_size}")

print(f"\nAspect Ratio Analysis:")
print(f"Mean aspect ratio: {mean_aspect_ratio}")
print(f"Min aspect ratio: {min_aspect_ratio}")
print(f"Max aspect ratio: {max_aspect_ratio}")

print(f"\nColor Composition Analysis:")
print(f"Mean RGB color composition: {mean_color_composition}")
'''

trainval_file_path = 'CVDL_G11/CVDL_Bsc-main/labs/01_image_classification/annotations/trainval.txt'

# Read the file to get image names (assuming each line contains an image file name with its corresponding label)
with open(trainval_file_path, 'r') as f:
    lines = f.readlines()

# List to store breed labels
labels = []

for line in lines:
    line = line.strip()  # Remove leading/trailing whitespaces/newlines
    breed = line.split('.')[0]  # This assumes the breed is in the filename and it's before the first '.'
    
    # Extract the breed name by removing numbers after the breed (e.g., "yorkshire_184" becomes "yorkshire")
    breed_name = breed.split('_')[0]
    
    labels.append(breed_name)

# Count occurrences of each breed
breed_counts = Counter(labels)

# Print the breed counts
for breed, count in breed_counts.items():
    print(f"{breed}: {count} images")