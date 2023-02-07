# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:16:24 2023

@author: Saqib Qamar
"""
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def augment_img(img):
    
    # Rotation
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),random.uniform(-45, 45),1)
    img = cv2.warpAffine(img, M, (cols, rows))
    

    # Scaling
    scale_percent = random.uniform(0.5, 1.5)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    

    # Cropping
    x, y, c = img.shape
    start_x = random.randint(0, x - 300)
    start_y = random.randint(0, y - 300)
    img = img[start_x:start_x+300, start_y:start_y+300]

    return img



def normalize_img(img):
    # Normalize the image
    img = img / 255.0
    return img



# Access the images from the folder
img_folder = "C:/Users/SAQIB/Desktop/spore/train"
img_files = [f for f in os.listdir(img_folder) if f.endswith('.tif')]



# Augment and normalize the images
augmented_img_folder = "C:/Users/SAQIB/Desktop/spore/augment/train"
if not os.path.exists(augmented_img_folder):
    os.makedirs(augmented_img_folder)



for idx, file in enumerate(img_files):
    img = cv2.imread(os.path.join(img_folder, file))
    img = augment_img(img)
    img = normalize_img(img)
    cv2.imwrite(os.path.join(augmented_img_folder, f"augmented_{idx}.jpg"), img)
    

#### Another Method

import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import os

# Load all images from a folder
images = []
folder = "C:/Users/SAQIB/Desktop/spore/train"
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
        images.append(img)

# Resize images
resized_images = []
resized_size = (2048, 1668)
for img in images:
    resized_img = cv2.resize(img, resized_size)
    resized_images.append(resized_img)

# Data augmentation pipeline
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
augmentation_pipeline = iaa.Sequential([
    iaa.Crop(px=(0, 16)),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-16, 16),
        order=[0, 1],
        cval=(0, 255),
        mode=ia.ALL
    ))
], random_order=True)

# Augment and normalize the images
augmented_images = augmentation_pipeline.augment_images(resized_images)
normalized_images = []
for img in augmented_images:
    normalized_img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    normalized_images.append(normalized_img)

# Save the augmented and normalized images to a new folder
new_folder = "C:/Users/SAQIB/Desktop/spore/augment/train"
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

for i, img in enumerate(normalized_images):
    cv2.imwrite(os.path.join(new_folder, "image_{}.jpg".format(i)), img * 255)

