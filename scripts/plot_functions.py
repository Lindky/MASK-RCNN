#!/usr/bin/env python

######load libraries########
import cv2
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from data_process import image_convert, combined_mask

###########################

def plot_box(img, box):

    colors = (255, 0, 0)
    thickness = 1
    
    for idx in range(len(box)):
        tmp_box = box[idx]
        x_min = tmp_box[0]
        y_min = tmp_box[1]
        x_max = tmp_box[2]
        y_max = tmp_box[3]
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))
        if idx == 0:
           plt_box = cv2.rectangle(image_convert(img), start_point, end_point, colors, thickness)
        else:
           plt_box = cv2.rectangle(plt_box, start_point, end_point, colors, thickness)

    return plt_box

def plot_img(data, no_, device):
    images = []
    masks = []
    for i in range(no_):
        num = data.__len__()       
        n = random.randint(0,num -1)
        image, target = data.__getitem__(n)
        images.append(image)
        masks.append(target['masks'])
    images = list(image.to(device) for image in images)
    masks = list(mask.to(device) for mask in masks)

    plt.figure(figsize=(10,6))
    for idx in range(0,no_):
        image = image_convert(images[idx])
        plt.subplot(2,no_,idx+1)
        plt.title('image')
        plt.imshow(image)
    for idx in range(0,no_):
        mask = combined_mask(masks[idx], 128, 128)
        plt.subplot(2,no_,idx+no_+1)
        plt.title('mask')
        plt.imshow(mask,cmap='gray')
    plt.show()


def get_predicted_box(pred, cutoff = 0.5):
    
    filtered_box = []
    for i in range(len(pred[0]['boxes'])):
        box=pred[0]['boxes'][i].detach().cpu().numpy()
        scr=pred[0]['scores'][i].detach().cpu().numpy()
        if scr > cutoff :
           filtered_box.append(box)
    return filtered_box

def get_predicted_masks(image, pred, IMG_HEIGHT = 128, IMG_WIDTH = 128, cutoff = 0.5):
    
    ig = image.detach().cpu().numpy()
    im2 =ig.copy()
    im2 = im2.transpose((1,2,0))
    for i in range(len(pred[0]['masks'])):
        scr=pred[0]['scores'][i].detach().cpu().numpy()
        if scr > 0.5 :   
           msk=pred[0]['masks'][i,0].detach().cpu().numpy()
           im2[:,:,0][msk>0.5] = random.randint(0,255)
           im2[:, :,1][msk > 0.5] = random.randint(0,255)
           im2[:, :,2][msk > 0.5] = random.randint(0,255)
    return im2.astype(np.uint8)


def merged_box_img(img, box):

    colors = (255, 0, 0)
    thickness = 1
    
    for idx in range(len(box)):
        tmp_box = box[idx]
        x_min = tmp_box[0]
        y_min = tmp_box[1]
        x_max = tmp_box[2]
        y_max = tmp_box[3]
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))
        if idx == 0:
           plt_box = cv2.rectangle(img, start_point, end_point, colors, thickness)
        else:
           plt_box = cv2.rectangle(plt_box, start_point, end_point, colors, thickness)

    return plt_box

def get_prediction(data, num_, model, device):
    imag = data.__getitem__(num_)
    img = []
    img.append(imag)
    img = list(image.to(device) for image in img)

    # testing the model 
    with torch.no_grad():
         pred = model(img)

    return imag, pred

def get_combined_masks(imag, pred, cutoff = 0.5):

    imag_mak = add_masks_to_image(image_convert(imag), pred)
    pred_box = get_predicted_box(pred)
    img = merged_box_img(imag_mak, pred_box)

    return img

def plot_testing_img(data, no_, model, device, plot_mask = True):
    images = []
    pred_ims = []
    pred_boxs = []
    n = len(no_)

    for idx in range(n):
        imag, pred = get_prediction(data, no_[idx], model, device)
        pred_img = get_combined_masks(imag, pred, cutoff = 0.5)

        pd_box = get_predicted_box(pred, cutoff = 0.5)
        pred_box = plot_box(imag, pd_box)

        images.append(imag)
        pred_ims.append(pred_img)
        pred_boxs.append(pred_box)

    plt.figure(figsize=(10,7))
    for idx in range(0,n):
         image = image_convert(images[idx])
         plt.subplot(2,n,idx+1)
         plt.title('image')
         plt.imshow(image)

    for idx in range(0,n):
         #mask = mask_convert(masks[idx])
         if plot_mask:
            p = pred_ims[idx]
         else:
            p = pred_box[idx]
         plt.subplot(2,n,idx+n+1)
         plt.title('predicted mask')
         plt.imshow(p,cmap='gray')
    plt.show()

def add_masks_to_image(ig, pred, IMG_HEIGHT = 128, IMG_WIDTH = 128, cutoff = 0.7):
    im2 =ig.copy()
    for i in range(len(pred[0]['masks'])):
        scr=pred[0]['scores'][i].detach().cpu().numpy()
        if scr > 0.5 :   
           msk=pred[0]['masks'][i,0].detach().cpu().numpy()
           im2[:,:,0][msk>0.5] = random.randint(0,255)
           im2[:, :,1][msk > 0.5] = random.randint(0,255)
           im2[:, :,2][msk > 0.5] = random.randint(0,255)
    return im2.astype(np.uint8)
