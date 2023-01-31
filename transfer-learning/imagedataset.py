import os
import xml.etree
import numpy as np
import cv2

import matplotlib.pyplot as plt
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

class ImageDataset(utils.Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # adds our only class
        self.add_class('pneumonia', 1, 'Lung Opacity')
        
        # adds images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', 
                image_id=i, 
                path=fp,
                annotations=annotations, 
                orig_height=orig_height, 
                orig_width=orig_width
            )
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


def parse_dataset(dicom_dir, anns): 
    image_fps = list(set(glob.glob(dicom_dir+'/'+'*.dcm')))
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations 


   
def read_train_test_val_split():
    f = open("train_test_val_split.json")
    data = json.load(f)

    train = data['image_fps_train']
    test = data['image_fps_test']
    val = data['image_fps_val']
    f.close()
    return train, test, val



def build_train_test_val_split(image_fps, train_per=0.7, test_per=0.15):
    ## Should only be run if json file does not exist

    image_fps_list = list(image_fps)
    image_fps_list.sort()
    random.shuffle(image_fps_list)
    l = len(image_fps_list)
    train = int(l*train_per)
    test = int(l*test_per)
    train_test_val_split = {
        'image_fps_train' : image_fps_list[:train],
        'image_fps_test'  : image_fps_list[train:(train+test)],
        'image_fps_val'   : image_fps_list[(train+test):]
    }
    ttv = json.dumps(train_test_val_split)
    f = open("train_test_val_split.json","w")
    f.write(ttv)
    f.close


