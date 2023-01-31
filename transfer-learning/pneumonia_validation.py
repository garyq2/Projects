import os
import xml.etree
import numpy as np
import cv2
import seaborn as sn

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
class InferenceConfig(ImageConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
#########################################################################################
#########################################################################################
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors 
#########################################################################################
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

#########################################################################################
if __name__ == "__main__":

    ORIG_SIZE = 1024
    config = ImageConfig()

    train_dicom_dir = '../data/stage_2_train_images/'
    test_dicom_dir = '../data/stage_2_test_images/'
    anns = pd.read_csv('../data/stage_2_train_labels.csv')

    image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)
    ds = pydicom.read_file(image_fps[0]) # read dicom image from filepath 
    image = ds.pixel_array # get image array

    #############################################################################
    # prep training dataset
    train, test, val = read_train_test_val_split()

    dataset_val = ImageDataset(val, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_val.prepare()
    ##############################################################################


    config = InferenceConfig() 
    model_dir = '../models/'
    model = modellib.MaskRCNN(mode='inference', config=config, model_dir=model_dir)
    model.load_weights("../models/pneumonia20221125T0838/mask_rcnn_pneumonia_0013.h5", by_name=True,)

    IoU = []
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []
    miss_class = []

    for count, image_id in enumerate(dataset_val.image_ids):
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config, image_id)
        results = model.detect([original_image])[0]
        rois = results['rois']
        class_ids = results['class_ids']
        scores = results['scores']
        mask = results['masks'] 
        m = 0
        if len(gt_bbox) > 0 and len(rois) > 0:
            for boxA in gt_bbox:
                for boxB in rois:
                    m = max(m,bb_intersection_over_union(boxA, boxB))

            true_positive.append(image_id)
            IoU.append(m)
            print(count)
        elif len(gt_bbox) > 0 and len(rois) == 0:
            false_negative.append(image_id)
        elif len(gt_bbox) == 0 and len(rois) > 0:
            false_positive.append(image_id)
        elif len(gt_bbox) == 0 and len(rois) == 0:
            true_negative.append(image_id)


    fp = len(false_positive)
    tp = len(true_positive)
    tn = len(true_negative)
    fn = len(false_negative)

    array =[[tn,fp],[fn, tp]]
    df_cm = pd.DataFrame(array, range(2), range(2))

    fpr = fp/(fp+tn)
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)

    accuracy = (tn+tp)/(tp+tn+fp+fn)
    #acc.append(accuracy)

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g') 
    plt.show()
    ##############################################################################

    dataset = dataset_val
    fig = plt.figure(figsize=(30, 30))
    for i in range(3):

        image_id = random.choice(dataset.image_ids)
        
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, config, image_id)
        
        print(original_image.shape)
        
        plt.subplot(3, 1, 1+i)
        results = model.detect([original_image]) #, verbose=1)
        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], 
                                    colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
    plt.show()

    ##############################################################################
    h = pd.read_csv("training_history.csv", index_col=0)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.set_title('Loss')
    ax1.plot(list(h.index), h['loss'], label = "training")
    ax1.plot(list(h.index), h['val_loss'], label = "validation")
    ax1.legend(loc="upper right")

    ax2.set_title('Class Loss')
    ax2.plot(list(h.index), h['rpn_class_loss'], label = "training")
    ax2.plot(list(h.index), h['val_rpn_class_loss'], label = "validation")
    ax2.legend(loc="upper right")

    ax3.set_title('Bounding Box Loss')
    ax3.plot(list(h.index), h['rpn_bbox_loss'], label = "training")
    ax3.plot(list(h.index), h['val_rpn_bbox_loss'], label = "validation")
    ax3.legend(loc="upper right")

    ax4.set_title('MRCNN Loss')
    ax4.plot(list(h.index), h['mrcnn_class_loss'], label = "training")
    ax4.plot(list(h.index), h['val_mrcnn_class_loss'], label = "validation")
    ax4.legend(loc="upper right")

    ax5.set_title('MRCNN Bounding Box Loss')
    ax5.plot(list(h.index), h['mrcnn_bbox_loss'], label = "training")
    ax5.plot(list(h.index), h['val_mrcnn_bbox_loss'], label = "validation")
    ax5.legend(loc="upper right")

    ax6.set_title('MRCNN Mask Loss')
    ax6.plot(list(h.index), h['mrcnn_mask_loss'], label = "training")
    ax6.plot(list(h.index), h['val_mrcnn_mask_loss'], label = "validation")
    ax6.legend(loc="upper right")

    plt.show()



























