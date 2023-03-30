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
import cv2
import pandas as pd
import torchvision.models.detection.mask_rcnn

from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from coco_eval import CocoEvaluator
from sklearn import metrics
###########################

###loading the data with combined masks 

class Nuclie_data(Dataset):
        def __init__(self,path):
            self.path = path
            self.folders = os.listdir(path)
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            #do image augmentation by chance 
            n = random.randint(0,1)
            self.transforms = get_transforms(0.5, 0.5, n)

            image_folder = os.path.join(self.path,self.folders[idx],'images/')
            mask_folder = os.path.join(self.path,self.folders[idx],'masks/')
            image_path = os.path.join(image_folder,os.listdir(image_folder)[0])
            
            #convert the RGBA into RGB
            image = io.imread(image_path)[:,:,:3].astype('float32')
            image = transform.resize(image,(128,128))
            
            masks = get_mask(mask_folder, 128, 128 )

            if n == 1:
               aug_masks = []
               for i in range(len(masks)):
                   aug_mask_ = self.transforms(image = image, mask=masks[i], a = 1)
                   aug_masks.append(aug_mask_['mask'])
               aug_img = self.transforms(image=image, a = 1)
               img = aug_img['image']
               masks = aug_masks
               
            boxes = get_bounding_box(masks)

            #convert the format
            if n == 0:
               mask = merged_mask(masks)
               augmented = self.transforms(image=image, mask=mask, a = 0)
               img = augmented['image']
               mask = augmented['mask']
            else: 
               mask = merged_mask(masks).astype('float32')
            
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            num_objs = len(boxes)

            # suppose all instances are not crow
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes 

            #there is only one class 
            target["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
            target["masks"] = mask
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            return (img,target)



##integrate the multiple masks into one mask
def get_mask(mask_folder,IMG_HEIGHT, IMG_WIDTH):
    masks = []
    for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder,mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            masks.append(mask_)
    return masks

##convert the image into tensors
def get_transforms(mean, std, n):
            print(n)
            list_transforms = []
            
            list_transforms.extend(
                    [
                HorizontalFlip(p = n), # only horizontal flip as of now
                    ])
            list_transforms.extend(
                    [
            Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
                    ])
            list_trfms = Compose(list_transforms)
            return list_trfms

##get the bounding box for each individual mask 
def get_bounding_box(masks):
    
    if torch.is_tensor(masks[0]):
       tmp = []
       for idx in range(len(masks)):
           mk = masks[idx]
           mk = mk.clone().cpu().numpy()
           tmp.append(mk)
       masks = tmp
       
    boxes = []
    for idx in range(len(masks)):
            mask_ = masks[idx]

            ###get the bounding box
            pos = np.where(mask_)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
              
    return boxes

##integrate the multiple masks into one mask
def merged_mask(masks, IMG_HEIGHT = 128, IMG_WIDTH = 128):
    if torch.is_tensor(masks[0]):
       tmp = []
       for idx in range(len(masks)):
           mk = masks[idx]
           mk = mk.clone().cpu().numpy()
           tmp.append(mk)
       masks = tmp
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for idx in range(len(masks)):
            mask_ = masks[idx]
            mask_ = np.expand_dims(mask_,axis=-1)
            mask = np.maximum(mask, mask_)
    return mask


class individual_mask_data(Dataset):
        def __init__(self,path):
            self.path = path
            self.folders = os.listdir(path)
            self.transforms = get_transforms(0.5, 0.5, 0)
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_folder = os.path.join(self.path,self.folders[idx],'images/')
            mask_folder = os.path.join(self.path,self.folders[idx],'masks/')
            image_path = os.path.join(image_folder,os.listdir(image_folder)[0])
            
            #convert the RGBA into RGB
            img = io.imread(image_path)[:,:,:3].astype('float32')
            img = transform.resize(img,(128,128))
            
            #mask = get_mask(mask_folder, 128, 128 ).astype('float32')
            boxes, masks = get_individual_box(mask_folder, 128, 128 )

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            #img = torch.as_tensor(img, dtype=torch.float32)
            image_id = torch.tensor([idx])
            
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            
            #augmentation
            augmented = self.transforms(image=img)
            img = augmented['image']
            #mask = augmented['mask']
            num_objs = len(boxes)
            
            # suppose all instances are not crow
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            
            target = {}
            target["boxes"] = boxes 

            #there is only one class 
            target["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            
            return (img,target) 

##get the bounding box for each individual mask 
def get_individual_box(mask_folder,IMG_HEIGHT, IMG_WIDTH):
    boxes = []
    masks = []
    for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder,mask_)).astype('float32')
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            ###get the bounding box
            pos = np.where(mask_)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(mask_)
              
    return boxes, masks

def combined_mask(masks,IMG_HEIGHT, IMG_WIDTH):
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    masks = masks.clone().cpu().detach().numpy()
    for idx in range(len(masks)):
            mask_ = masks[idx]
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_,axis=-1)
            mask = np.maximum(mask, mask_)
              
    return mask
 
def convert_to_coco_api(ds, rangea = range(581, 670)):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in rangea:
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds.__getitem__(img_idx)
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def evaluate_IoU(data, model, rangea, device):
    coco_api = convert_to_coco_api(data, rangea)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco_api, iou_types)

    cpu_device = torch.device("cpu")
    for idx in rangea:
        image, target = data.__getitem__(idx)
    
        #make sure the input format is a list 
        images = []
        images.append(image)
        images = list(image.to(device) for image in images)

        targets = []
        targets.append(target)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator

def pixel_accuracy(data, model, idx, device):
    imag, target = data.__getitem__(idx)
    img = []
    img.append(imag)
    img = list(image.to(device) for image in img)

    # testing the model 
    with torch.no_grad():
        pred = model(img)

    ## get all of predicted boxes 
    pred_box = get_predicted_box(pred, cutoff = 0.5)
    p_0 = plot_box(imag, pred_box)

    ##get groung truth
    ground_truth = plot_box(imag, target['boxes'])
    
    ## pixel accuracy
    acc = metrics.accuracy_score(ground_truth.reshape(-1), p_0.reshape(-1))
    return acc

def get_predicted_box(pred, cutoff = 0.7):
    
    filtered_box = []
    for i in range(len(pred[0]['boxes'])):
        box=pred[0]['boxes'][i].detach().cpu().numpy()
        scr=pred[0]['scores'][i].detach().cpu().numpy()
        if scr > cutoff :
           filtered_box.append(box)
    return filtered_box

def get_predicted_masks(pred, IMG_HEIGHT = 128, IMG_WIDTH = 128, cutoff = 0.7):
    pred_masks =pred[0]['masks']
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for idx in range(len(pred_masks)):
            scr = pred[0]['scores'][idx].detach().cpu().numpy()
            if scr > cutoff :
               mask_ = pred_masks[idx].detach().cpu().numpy()
               mask_ = np.squeeze(mask_)
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
