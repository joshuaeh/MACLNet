# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")

from sklearn.model_selection import train_test_split
import albumentations as A
import tensorflow as tf
from config import *
from utils import *
import numpy as np
import cv2
import os
import random

train_augment = A.Compose([ 
    A.RandomCrop(height = CROP_SIZE[0], width = CROP_SIZE[1], p = 1), 
    A.CLAHE(p = 0.5),   
    A.RandomGamma(p = 0.5),
    A.OneOf([
            A.VerticalFlip(p = 0.25),
            A.HorizontalFlip(p = 0.25),
            A.Transpose(p = 0.25),
            A.RandomRotate90(p = 0.25),
        ], p = 1.0),
    A.OneOf([
             A.GridDistortion(p = 0.3),
             A.OpticalDistortion(distort_limit = 2, shift_limit = 0.5, p = 0.3),
             A.ElasticTransform(p = 0.3, alpha = 120, sigma = 120 * 0.05, alpha_affine = 120 * 0.03),
        ], p = 1.0),
    A.OneOf([
             A.RandomBrightness(p = 0.25),
             A.RandomContrast(p = 0.25),
    ], p = 1.0),
])

test_augment = A.RandomCrop(height = CROP_SIZE[0], width = CROP_SIZE[1], p = 1.0)

def getDataset(batch_size, resize_size, crop_size, input_img_paths, target_img_paths, data_type):

    input_img_paths = sorted([os.path.join(IMAGES_DIR, x) for x in os.listdir(IMAGES_DIR)])
    target_img_paths = sorted([os.path.join(MASKS_DIR, x) for x in os.listdir(MASKS_DIR)])
    return

class DataSetter(tf.data.Dataset):
    """A Dataloader based on the Tensorflow API for Image Augmentation and Segmentation"""
    def __init__(self, images_dir,
        masks_dir, 
        image_size, 
        channels=(3,3), 
        crop_percent=None, 
        seed=None, 
        segment=True, 
        compose=False, 
        one_hot_encoding=False, 
        palette=None):
        """
        
        """
        # make parameters accessible
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.channels = channels
        if crop_percent is not None:
            if 0.0 < crop_percent <= 1.0:
                self.crop_percent = tf.constant(crop_percent, tf.float32)
            elif 0 < crop_percent <= 100:
                self.crop_percent = tf.constant(crop_percent / 100., tf.float32)
            else:
                raise ValueError("Invalid value entered for crop size. Please use an \
                                  integer between 0 and 100, or a float between 0 and 1.0")
        else:
            self.crop_percent = None
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, 1000)
        self.segment = segment
        self.compose = compose
        self.one_hot_encoding = one_hot_encoding
        self.palette = palette

        # get image paths while we're at it
        self.image_paths = sorted([os.path.join(self.images_dir, x) for x in os.listdir(self.images_dir)])
        self.image_masks = sorted([os.path.join(self.masks_dir, x) for x in os.listdir(self.masks_dir)])

    def __len__(self):
        return 