# MASK-RCNN
Final project for the class MBP1413H

**MASK-RCNN-full.ipynb** is the main script used for model training. <br>
 <br>
The functions used in the **MASK-RCNN-full.ipynb** are saved in the scripts folder. <br>
Software: **Pytorch** <br>

**Model evaluation scripts** saved in the evaluation folder  

**dataset:** https://www.kaggle.com/competitions/data-science-bowl-2018/data

**Note:** Once you downloaded the dataset from the website, please check the number of file you have. It should contain 670 folders in ***stage1_train*** folder. If you found you have more than 670 files, indicating there is a hidden file. Please remove it before following this tutorial.

### Final model
https://drive.google.com/file/d/1-ijGaZUF8F6MG7K-jPfwW6BDXG6GXF_c/view?usp=sharing

MASK-RCNN paper: 
https://doi.org/10.48550/arXiv.1703.06870

Google docs:
https://docs.google.com/document/d/1Wb2Zp-FZGfT_w5dRbFDn7HJrSirsg5RQvndXyvh9u0s/edit

## Colab
To run this model, we use online machine learning tool called Colaboratory (Colab): https://colab.research.google.com

Colab is a free Jupyter notebook environment that runs entirely in the cloud. Most importantly, it does not require a setup. (Most of libraries are pre-installed). It provides free GPU in cloud to do then simple image proccessing, which is suitable for this project.

This is a example how to train MASK-RCNN using pre-trained pytorch model on Colab: <br>
https://github.com/Lindky/MASK-RCNN/blob/main/MASK-RCNN-full.ipynb

### references: <br>
https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
https://github.com/sagieppel/Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code/blob/7a8c899f8cc7ecd57d704767b30b116bccf44e78/train.py#L2 <br>
https://www.kaggle.com/code/robinteuwens/mask-rcnn-detailed-starter-code/notebook#Training
https://bjornkhansen95.medium.com/mask-r-cnn-for-segmentation-using-pytorch-8bbfa8511883
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

### Note: 
1. To train the model on Colab, you first need to upload the training dataset to the google drive. **Make sure you upload your data to the path: '/content/MyDrive'**, but not sync your data from your computer to the google drive. (Colan can access to your google drive but not the private computor backup folder)

2. You can use this code to mount the google drive on Colab:
```
from google.colab import drive
drive.mount('/content/MyDrive')
```
After running this command, now you can access your google drvie from Colab 

**3. The training dataset should be uploaded to Colab virtual machine (VM):**

How to upload the file from google drive to Colab:
https://www.youtube.com/watch?v=BuuH0wsJ8-k



## Evaluation

There are some certain matrics people usually used for the machine learning model evaluation: <br>
### 1. AP (Precision), Recall, mAP (mean Average Precision).

TP = True Positive. <br /> FP = False Positive   <br /> FN = False Negative
<br>

$$AP = \frac {TP} {TP + FP}$$

<br>

$$Recall = \frac {TP} {TP + FN}$$

**reference**: https://hasty.ai/docs/mp-wiki/metrics/map-mean-average-precision

### 2. Overall Accuracy
2.1 Convert the predicted image and ground truth into 1-D array. <br>
2.2 Compare the pixels between image and ground truth <br />
**Notes:** May not precisely evaluate the image segmentation.

### 3. Intersection over Union (IoU)
Standard way to evaluate the image segmentation.
$$IoU = \frac {TP} {TP + FP + FN}$$

IoU explaination with example: <br>
https://www.youtube.com/watch?v=0FmNxqLFeYo&t=544s <br>
https://hasty.ai/docs/mp-wiki/metrics/iou-intersection-over-union
<br>
<br>
COCO evaluation format explaination: <br> 
https://stackoverflow.com/questions/56002220/are-these-the-expected-results-of-tensorflow-object-detection-evaluation-using-m

**Notes:** The COCO evaluation functions (**coco_eval.py**) come from https://github.com/pytorch/vision/tree/main/references/detection


## Optimization 
This step is complicated and also very time-consuming.

Tuning hyper-parameters is usually the first step when optimizing the model. There is an reference to introduce what kinds of hyper-parameters we should be aware of. 

**reference**: https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8
