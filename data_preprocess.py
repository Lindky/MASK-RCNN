#!/usr/bin/env python

######load libraries########

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
from skimage import io, transform
import matplotlib.pyplot as plt
import random


###########################

###loading the data
#class Nuclie_data(Dataset):
#        def __init__(self,path):
#            self.path = path
#            self.folders = os.listdir(path)
#            self.transforms = get_transforms(0.5, 0.5)
#        
#        def __len__(self):
#            return len(self.folders)
#              
#        
#        def __getitem__(self,idx):
#            image_folder = os.path.join(self.path,self.folders[idx],'images/')
#            mask_folder = os.path.join(self.path,self.folders[idx],'masks/')
#            image_path = os.path.join(image_folder,os.listdir(image_folder)[0])
#            
#            img = io.imread(image_path)[:,:,:3].astype('float32')
#            img = transform.resize(img,(128,128))
#            
#            mask = get_mask(mask_folder, 128, 128 ).astype('float32')
#            #augmentation
#            augmented = self.transforms(image=img, mask=mask)
#            img = augmented['image']
#            mask = augmented['mask']
#            return (img,mask) 


##get the bounding box for each individual mask 
def get_bounding_box(mask_folder,IMG_HEIGHT, IMG_WIDTH):
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    boxes = []
    for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder,mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            ###get the bounding box
            pos = np.where(mask_)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
              
    return boxes



##loading the data
class Nuclie_data(Dataset):
        def __init__(self,path):
            self.path = path
            self.folders = os.listdir(path)
            self.transforms = get_transforms(0.5, 0.5)
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_folder = os.path.join(self.path,self.folders[idx],'images/')
            mask_folder = os.path.join(self.path,self.folders[idx],'masks/')
            image_path = os.path.join(image_folder,os.listdir(image_folder)[0])
            
            #convert the RGBA into RGB
            img = io.imread(image_path)[:,:,:3].astype('float32')
            img = transform.resize(img,(128,128))
            
            mask = get_mask(mask_folder, 128, 128 ).astype('float32')
            boxes = get_bounding_box(mask_folder, 128, 128 )

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            #augmentation
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            #target structure
            final_target = []
            final_image = []
            num_objs = len(boxes)
            target = {}
            target["boxes"] = boxes 

            #there is only one class 
            target["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
            target["masks"] = mask
            final_target.append(target)
            final_image.append(img)
            return (final_image,final_target) 

##convert the image into tensors
def get_transforms(mean, std):
            list_transforms = []
            
            list_transforms.extend(
                    [
                HorizontalFlip(p=0.5), # only horizontal flip as of now
                    ])
            list_transforms.extend(
                    [
            Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
                    ])
            list_trfms = Compose(list_transforms)
            return list_trfms


##integrate the multiple masks into one mask
def get_mask(mask_folder,IMG_HEIGHT, IMG_WIDTH):
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder,mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_,axis=-1)
            mask = np.maximum(mask, mask_)
              
    return mask


# converting tensor to image
def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    std = np.array((0.5,0.5,0.5))
    mean = np.array((0.5,0.5,0.5))
    image  = std * image + mean
    image = image.clip(0,1)
    image = (image * 255).astype(np.uint8)
    return image

def mask_convert(mask):
    mask = mask.clone().cpu().detach().numpy()
    #mask = mask.transpose((1,2,0))
    std = np.array((0.5))
    mean = np.array((0.5))
    mask  = std * mask + mean
    mask = mask.clip(0,1)
    mask = np.squeeze(mask)
    return mask

##view the images and combined masks
#def plot_img(no_,train_loader, device):
#    iter_ = iter(train_loader)
#    images,masks = next(iter_)
#    images = images.to(device)
#    masks = masks.to(device)
#    plt.figure(figsize=(10,6))
#    for idx in range(0,no_):
#         image = image_convert(images[idx])
#         plt.subplot(2,no_,idx+1)
#         plt.title('image')
#         plt.imshow(image)
#    for idx in range(0,no_):
#         mask = mask_convert(masks[idx])
#         plt.subplot(2,no_,idx+no_+1)
#         plt.title('mask')
#         plt.imshow(mask,cmap='gray')
#    plt.show()


#randomly view the images and combined masks
def plot_img(no_, data, device):
    #iter_ = iter(train_loader)
    #images,masks = next(iter_)
    #images = images.to(device)
    #masks = masks.to(device)
    #plt.figure(figsize=(10,6))
    data_size = data.__len__()
    images = []
    masks =[]
    plt.figure(figsize=(10,6))
    for i in range(0,no_):
        idx=random.randint(0,data_size-1)
        ig, ms = data.__getitem__(idx)
        image = ig[0].to(device)
        image = image_convert(image)
        plt.subplot(2,no_,i+1)
        plt.title('image')
        plt.imshow(image)
        
        mask = ms[0]
        mask = mask['masks']
        mask = mask.to(device)
        mask = mask_convert(mask)
        plt.subplot(2,no_,i+no_+1)
        plt.title('mask')
        plt.imshow(mask,cmap='gray')
    plt.show()
