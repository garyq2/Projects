import os
import xml.etree
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob
from sklearn.model_selection import KFold

import sys
import random
random.seed(42)
import math

ROOT_DIR = '../'
sys.path.append(ROOT_DIR)

import mrcnn.utils
import mrcnn.config
import mrcnn.model

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from imagedataset import ImageDataset, parse_dataset, read_train_test_val_split
#########################################################################################

#if __name__ == "__main__":

train_dir = '../data/stage_2_train_images/'
test_dir = '../data/stage_2_test_images/'
anns = pd.read_csv('../data/stage_2_train_labels.csv')

image_fps, image_annotations = parse_dataset(train_dir, anns=anns)
ds = pydicom.read_file(image_fps[0]) # read dicom image from filepath 
image = ds.pixel_array # get image array

#############################################################################
# prep training and testing dataset
ORIG_SIZE = 1024
train, test, val = read_train_test_val_split()

dataset_train = ImageDataset(train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()
dataset_val = ImageDataset(test, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()

##############################################################################
# Load and display Images

print(anns.describe())


anns_non_na = anns.copy().dropna()











##############################################################################
# Load and display pneumonia Images

class_ids = [0]
while class_ids[0] == 0:  ## look for a mask
    image_id = random.choice(dataset_train.image_ids)
    image_fp = dataset_train.image_reference(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(10, 4))
plt.imshow(image)

npw = np.where(mask == True)
x = npw[1][0]
y = npw[0][0]
height = npw[0][-1] - y
width  = npw[1][-1] - x
plt.gca().add_patch(Rectangle((x,y),width,height,
                    edgecolor='blue',
                    facecolor='none',
                    lw=4))
#plt.axis('off')
print(image_fp)
print(class_ids)
plt.show()

##################################################################################
# Load and display healthy Images

class_ids = [1]
while class_ids[0] == 1:  ## look for a mask
    image_id = random.choice(dataset_train.image_ids)
    image_fp = dataset_train.image_reference(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(10, 4))
plt.imshow(image)
plt.show()




