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
import data_preprocess

##################################

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

#loading the data
data_dir = '/Users/linyang/Downloads/stage1_train'
data = data_preprocess.Nuclie_data(data_dir)

#data size
print(data.__len__())


#randomly view 'no_' number of images
data_preprocess.plot_img(5, data, device)

#import model
import torchvision.models.segmentation
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# replace the classifier with a new one, that hasnum_classes which is user-defined
num_classes = 2  # 1 class (nuclei) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

#############################Trainning model################################
train_set = 580
for i in range(1001):
    # images, targets = get_target()
    idx = random.randint(0, train_set - 1)
    images, targets = data.__getitem__(idx)
    # iter_ = iter(train_loader)
    # images,masks = next(iter_)
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    print(f"start iteration {i}")
    optimizer.zero_grad()
    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())
    losses.backward()
    optimizer.step()
    print(i, 'loss:', losses.item())
    if i % 100 == 0:
        torch.save(model.state_dict(), str(i) + ".torch")



