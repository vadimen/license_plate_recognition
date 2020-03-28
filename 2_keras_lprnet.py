#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:46:37 2020

@author: vadim

this code was made by analyzing 
this https://github.com/SQMah/Plate-Reading-Network/blob/master/LPRNet.py
this https://arxiv.org/pdf/1806.10447.pdf

"""
import tensorflow as tf
import numpy as np
import time
import cv2
import os
import random
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D,MaxPooling3D,ZeroPadding2D
from tensorflow.keras.layers import Input, Dense, Activation,Dropout,Reshape
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model

CHARS = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
     'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
     'W', 'X', 'Y', 'Z', 'I', 'O', '-'
     ]

def small_basic_block(nr_filters):
	def func(inp):
		x = Conv2D(filters=nr_filters//4, kernel_size=1, strides=1,activation='relu')(inp)
		x = BatchNormalization()(x)
		x = ZeroPadding2D(padding=(1,0))(x)
		x = Conv2D(filters=nr_filters//4, kernel_size=(3,1), strides=1,activation='relu')(x)
		x = BatchNormalization()(x)
		x = ZeroPadding2D(padding=(0,1))(x)
		x = Conv2D(filters=nr_filters//4, kernel_size=(1,3), strides=1, activation='relu')(x)
		x = BatchNormalization()(x)
		x = Conv2D(filters=nr_filters, kernel_size=1, strides=1, activation='relu')(x)
		return BatchNormalization()(x)
	return func

#LPRNet
class_number = len(CHARS)
img_width = 94#cols
img_height = 24#rows
x = Input(shape=(24,94,3))#data_format is channels_last
x = Conv2D(filters=64, kernel_size=(3,3), strides=1,activation='relu')(x)
x = BatchNormalization()(x)
#x = Reshape((-1, x.shape[1],x.shape[2],x.shape[3]), input_shape=(-1,x.shape[1],x.shape[2],x.shape[3]))(x)
#x = MaxPooling3D(pool_size = (1, 3, 3), strides = (1, 1, 1))(x)
#x = Reshape((x.shape[2],x.shape[3],x.shape[4]), input_shape=(-1,x.shape[2],x.shape[3],x.shape[4]))(x)
x = MaxPooling2D(pool_size=(3,3), strides=(1,1))(x)
x = small_basic_block(nr_filters=128)(x)

#x = Reshape((-1, x.shape[1],x.shape[2],x.shape[3]), input_shape=(-1,x.shape[1],x.shape[2],x.shape[3]))(x)
#x = MaxPooling3D(pool_size = (3, 3, 2), strides = (1, 1, 1), padding='same')(x)
#x = Reshape((x.shape[2],x.shape[3],x.shape[4]), input_shape=(-1,x.shape[2],x.shape[3],x.shape[4]))(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2,1))(x)
x = small_basic_block(nr_filters=256)(x)
x = small_basic_block(nr_filters=256)(x)

x = MaxPooling2D(pool_size=(3,3), strides=(2,1))(x)
x = Dropout(rate=0.5)(x)
x = Conv2D(filters=256, kernel_size=(4,1), strides=1)(x)
x = Dropout(rate=0.5)(x)
x = Conv2D(filters=class_number, kernel_size=(1,13), strides=1)(x)
x = BatchNormalization()(x)