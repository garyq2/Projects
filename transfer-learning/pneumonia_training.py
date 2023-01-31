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
from imgaug import augmenters as iaa

from imagedataset import ImageDataset, parse_dataset, read_train_test_val_split, build_train_test_val_split

#########################################################################################
class ImageConfig(Config):
    NAME = 'pneumonia'
    BACKBONE = 'resnet50'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)

    IMAGES_PER_GPU = 4 #32
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 4 
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.8
    DETECTION_NMS_THRESHOLD = 0.01
    VALIDATION_STEPS = 125*8
    STEPS_PER_EPOCH = 815*8
#########################################################################################

if __name__ == "__main__":
    #############################################################################
    augmentation = iaa.Sequential([
        iaa.OneOf([ ## geometric transform
            iaa.Affine(                                 # Generates Global Distortions
                scale={"x": (0.99, 1.01), "y": (0.99, 1.01)},
                translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                rotate=(-5, 5)
            ),
            iaa.Affine(                                 # Generates Global Distortions
                rotate=(-2, 2),
            )
        ]),
        iaa.OneOf([                                     # Either brightens or contrasts
            iaa.Multiply((0.95, 1.05)),
            iaa.ContrastNormalization((0.99, 1.01)),
        ])
    ])
    #############################################################################
    
    config = ImageConfig()
    train_dir = '../data/stage_2_train_images/'
    trest_dir = '../data/stage_2_test_images/'
    anns = pd.read_csv('../data/stage_2_train_labels.csv')

    image_fps, image_annotations = parse_dataset(train_dir, anns=anns)
    ds = pydicom.read_file(image_fps[0])  
    image = ds.pixel_array 

    #############################################################################
    # prep training and testing dataset
    ORIG_SIZE = 1024
    train, test, val = read_train_test_val_split()

    dataset_train = ImageDataset(train, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_train.prepare()
    dataset_test = ImageDataset(test, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_test.prepare()


    ##############################################################################
    # Loading Model Weights
    model_dir = '../models/'
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=model_dir)

    model.load_weights(model_dir+"mask_rcnn_coco.h5", 
        by_name=True, 
        exclude=[
            "mrcnn_class_logits", 
            "mrcnn_bbox_fc",
            "mrcnn_bbox",
            "mrcnn_mask"
         ]
    )
    #model.load_weights("../models/pneumonia20221125T0838/mask_rcnn_pneumonia_0007.h5", by_name=True, )

    ##############################################################################
    # Training Model

    
    LEARNING_RATE = 0.0005
    model.train(dataset_train, dataset_test,
                learning_rate=LEARNING_RATE,
                epochs=5,
                layers='heads',
                augmentation=None)
    
    LEARNING_RATE = 0.00001
    model.train(dataset_train, dataset_test,
                learning_rate=LEARNING_RATE,
                epochs=10,
                layers='heads',
                augmentation=augmentation)
    
    LEARNING_RATE = 0.00001
    model.train(dataset_train, dataset_test,
                learning_rate=LEARNING_RATE,
                epochs=12,
                layers='3+',
                augmentation=augmentation)
    
    LEARNING_RATE = 0.000001
    model.train(dataset_train, dataset_test,
                learning_rate=LEARNING_RATE,
                epochs=13,
                layers='all',
                augmentation=augmentation)

    history = model.keras_model.history.history
    epochs = range(1,len(next(iter(history.values())))+1)
    hist = pd.DataFrame(history, index=epochs)
    hist.to_csv("training_history.csv")


























