#!/usr/bin/env python

#########################################
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
from skimage import io, transform
import matplotlib.pyplot as plt
from data_preprocess import Nuclie_data, get_transforms, get_mask, image_convert, image_convert, mask_convert, plot_img

##################################

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

#loading the data
data_dir = '/Users/linyang/Downloads/stage1_train'
data = Nuclie_data(data_dir)

#data size
print(data.__len__())

#splitting to trainset and validation set and loading the data with batch size of 10
trainset, validset = random_split(data, [580, 90])
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=10, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=10)

#view images and their corresponding combined marks
plot_img(5, train_loader, device)

