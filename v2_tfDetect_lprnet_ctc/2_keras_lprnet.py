#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:46:37 2020

@author: vadim

this code was made by analyzing 
https://github.com/DeepSystems/supervisely-tutorials/blob/master/anpr_ocr/src/image_ocr.ipynb
this https://github.com/SQMah/Plate-Reading-Network/blob/master/LPRNet.py
this https://arxiv.org/pdf/1806.10447.pdf
https://github.com/xuexingyu24/License_Plate_Detection_Pytorch/blob/master/LPRNet/model/LPRNET.py

--la output din small_basic_block in unele locuri nu e dimensiunea care trebuie

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
from tensorflow.keras.optimizers import SGD
from os.path import join
import random
import itertools
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing import image
import tensorflow.keras.callbacks
import pandas as pd

data_dir = '/home/vadim/Desktop/plate_and_nr_dataset/'
data = pd.read_csv('/home/vadim/Desktop/plate_and_nr_dataset/dataset.csv')

img_links = list(data.iloc[:,0].values)
plates = list(data.iloc[:,1].values)

max_plate_len = max(list(map(lambda x:len(x), plates)))

alphabet = []
for plate in plates:
	for c in plate:
		alphabet.append(c)

alphabet = list(set(alphabet))
alphabet.sort()

sess = tf.Session()
K.set_session(sess)

def labels_to_text(labels):
	return ''.join(list(map(lambda x:alphabet[int(x)] ,labels)))

def text_to_labels(text):
	return list(map(lambda x:alphabet.index(x), text))

class TextImageGenerator:
	def __init__(self, img_w, img_h, 
			        batch_size, downsample_factor):
		self.samples = []
		self.downsample_factor = downsample_factor
		self.img_h = img_h
		self.img_w = img_w
		self.batch_size = batch_size
		self.samples_len = len(plates)
		self.indexes = list(range(self.samples_len))
		self.cur_index = 0
	
	def build_data(self):
		self.imgs = np.zeros((self.samples_len, self.img_h, self.img_w))
		for i,l in enumerate(img_links):
			img = cv2.imread(data_dir+l)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (self.img_w, self.img_h))
			img = img.astype(np.float32)
			img /= 255
			# width and height are backwards from typical Keras convention
			# because width is the time dimension when it gets fed into the RNN
			self.imgs[i, :, :] = img
			
	def get_output_size(self):
		return len(plates) + 1
	
	def next_sample(self):
		self.cur_index += 1
		if self.cur_index >= self.samples_len:
			self.cur_index = 0
			random.shuffle(self.indexes)
		return self.imgs[self.indexes[self.cur_index]], plates[self.indexes[self.cur_index]]
	
	def next_batch(self):
		# width and height are backwards from typical Keras convention
		# because width is the time dimension when it gets fed into the RNN
		while True:
			if K.image_data_format() == 'channel_first':
				X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
			else:
				X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
			Y_data = np.ones([self.batch_size, max_plate_len])
			
			input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor-2)
			label_length = np.zeros((self.batch_size, 1))
			
			for i in range(self.batch_size):
				img,text = self.next_sample()
				img = img.T
				if K.image_data_format() == 'channel_first':
					img = np.expand_dims(img, 0)
				else:
					img = np.expand_dims(img, -1)
				X_data[i] = img
				Y_data[i] = text_to_labels(text) + [0]*(max_plate_len-len(text))
				label_length[i] = len(text)
				
				inputs = {
						'the_input':X_data,
						'the_labels':Y_data,
						'input_length':input_length,
						'label_length':label_length
				}
				outputs = {'ctc':np.zeros([self.batch_size])}
				yield (inputs, outputs)

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
class_number = len(alphabet)
img_width = 94#cols
img_height = 24#rows
inp = Input(shape=(img_height,img_width,3))#data_format is channels_last
x = Conv2D(filters=64, kernel_size=(3,3), strides=1,activation='relu')(inp)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3,3), strides=(1,1))(x)
x = small_basic_block(nr_filters=128)(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2,1))(x)
x = small_basic_block(nr_filters=256)(x)
x = small_basic_block(nr_filters=256)(x)

x = MaxPooling2D(pool_size=(3,3), strides=(2,1))(x)
x = Dropout(rate=0.5)(x)
x = Conv2D(filters=256, kernel_size=(4,1), strides=1)(x)
x = Dropout(rate=0.5)(x)
x = Conv2D(filters=class_number, kernel_size=(1,13), strides=1)(x)
y_pred = BatchNormalization()(x)
y_pred = Reshape((y_pred.shape[2],y_pred.shape[3]))(y_pred)

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:,2:, :]
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

labels = Input(name='the_labels', shape=[max_plate_len], dtype='float32')
input_len = Input(name='input_length',shape=[1], dtype='int64')
label_len = Input(name='label_length',shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_len, label_len])
sgd = SGD(learning_rate=0.02, decay=1e-6, momentum=0.9, nesterov=True)

model = Model(inputs=[inp, labels, input_len, label_len], outputs=loss_out)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc':lambda y_true,y_pred:y_pred}, optimizer=sgd)


train_gen = TextImageGenerator(94,24,30,4)
train_gen.build_data()
valid_gen = TextImageGenerator(94,24,30,4)
valid_gen.build_data()

#test_func = K.function([inp], [y_pred])
model.fit_generator(generator=train_gen.next_batch(),
					steps_per_epoch=train_gen.samples_len,
					epochs=1,
					validation_data=valid_gen.next_batch(),
					validation_steps=valid_gen.samples_len)