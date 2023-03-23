# MASK-RCNN
Final project for the class MBP1413H


The functions used in the Mask-RCNN.py are saved in the functions folder.
Software: Pytorch 

**dataset:** https://www.kaggle.com/competitions/data-science-bowl-2018/data

**Note:** Once you downloaded the dataset from the website, please check the number of file you have. It should contain 670 folders in ***stage1_train*** folder. If you found you have more than 670 files, indicating there is a hidden file. Please remove it before following this tutorial.

MASK-RCNN paper: 
https://doi.org/10.48550/arXiv.1703.06870

## Colab
To run this model, we use online machine learning tool called Colaboratory (Colab): https://colab.research.google.com

Colab is a free Jupyter notebook environment that runs entirely in the cloud. Most importantly, it does not require a setup. (Most of libraries are pre-installed). It provides free GPU in cloud to do then simple image proccessing, which is suitable for this project.

This is a example how to train MASK-RCNN using pre-trained pytorch model on Colab:

https://github.com/Lindky/MASK-RCNN/blob/main/Mask-RCNN.ipynb

### Note: 
1. To train the model on Colab, you first need to upload the training dataset to the google drive. **Make sure you upload your data to the path: '/content/MyDrive'**, but not sync your data from your computer to the google drive. (Colan can access to your google drive but not the private computor backup folder)

2. You can use this code to mount the google drive on Colab:
```
from google.colab import drive
drive.mount('/content/MyDrive')
```
After running this command, now you can access your data in google drvie from Colab 
e.g.

```
#loading the data
data_dir = '/content/MyDrive/MyDrive/stage1_train/'
data = Nuclie_data(data_dir)
```

## Run model on local
If you want to train the model on your local computer, make sure that your GPU device is CUDA (most commonly used for most of machine learning studies)

To check your GPU device:

```
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
```

Note: When you install ***numpy***, make sure the version is 1.23.5 to aviod some potential errors.
