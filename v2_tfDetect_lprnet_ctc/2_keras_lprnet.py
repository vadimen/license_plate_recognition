#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:46:37 2020
First time run a successful epoch Tue April 14

@author: Vadim Placinta

this works in tf v1
this is LPRnet with CTC loss
demo video with lprnet working https://youtu.be/tzRYvfbdKMk

this code was made by analyzing:
- https://github.com/DeepSystems/supervisely-tutorials/blob/master/anpr_ocr/src/image_ocr.ipynb
- https://github.com/SQMah/Plate-Reading-Network/blob/master/LPRNet.py
- https://arxiv.org/pdf/1806.10447.pdf
- https://github.com/xuexingyu24/License_Plate_Detection_Pytorch/blob/master/LPRNet/model/LPRNET.py

TO-DO:
- make it not overfit

"""

import numpy as np
import time
import cv2
import os
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, MaxPooling3D
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Reshape
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sess = tf.Session()
K.set_session(sess)

def labels_to_text(labels):
	return ''.join(list(map(lambda x: alphabet[int(x)], labels)))

def text_to_labels(text):
	return list(map(lambda x: alphabet.index(x), text))

def apply_random_effect(img, eff):
	shape = (img.shape[1], img.shape[0])
	if eff == 0:#apply noise
		noise = cv2.imread('noise.jpg')
		noise = cv2.resize(noise, shape)
		alpha = random.randint(55,90)/100  # between 0.55 and 0.9 is good
		return cv2.addWeighted(img, alpha, noise, (1.0 - alpha), 0.0)
	elif eff == 1:#apply dirt
		dirt = cv2.imread('dirt.jpg')
		dirt = cv2.resize(dirt, shape)
		alpha = random.randint(55, 90) / 100  # between 0.55 and 0.9 is good
		return cv2.addWeighted(img, alpha, dirt, (1.0 - alpha), 0.0)
	else:#apply blur
		return cv2.blur(img,(random.randint(1,4),random.randint(1,4))) #every param in range 1:4

class TextImageGenerator:
	def __init__(self, img_w, img_h,batch_size,data_dir, alphabet, max_plate_len):
		self.samples = []
		self.img_h = img_h
		self.img_w = img_w
		self.batch_size = batch_size
		self.cur_index = 0
		self.data_dir = data_dir
		self.alphabet = alphabet
		self.max_plate_len = max_plate_len

	def build_data(self):
		data = pd.read_csv(self.data_dir + '/dataset.csv')
		img_links = list(data.iloc[:, 0].values)
		self.plates = list(data.iloc[:, 1].values)

		self.samples_len = len(self.plates)
		self.indexes = list(range(self.samples_len))

		if K.image_data_format() == 'channel_first':
			#uint8 is needed for apply_random_effect() which is applied in next_batch()
			#otherwise apply_random_effect() returns an error
			self.imgs = np.zeros((self.samples_len, 3, self.img_h, self.img_w), dtype=np.uint8)
		else:
			self.imgs = np.zeros((self.samples_len, self.img_h, self.img_w, 3), dtype=np.uint8)

		for i, l in enumerate(img_links):
			img = cv2.imread(self.data_dir + l)
			self.imgs[i, :, :, :] = img

	def next_sample(self):
		self.cur_index += 1
		if self.cur_index >= self.samples_len:
			self.cur_index = 0
			random.shuffle(self.indexes)
		return self.imgs[self.indexes[self.cur_index]], self.plates[self.indexes[self.cur_index]]

	def next_batch(self):
		global dim_of_samples
		while True:
			if K.image_data_format() == 'channel_first':
				X_data = np.ones([self.batch_size, 3, self.img_h, self.img_w])
			else:
				X_data = np.ones([self.batch_size, self.img_h, self.img_w, 3])

			Y_data = np.ones([self.batch_size, self.max_plate_len])
			input_length = np.ones((self.batch_size, 1)) * dim_of_samples  # is dim in samples outputed by network
			label_length = np.zeros((self.batch_size, 1))

			for i in range(self.batch_size):
				img, text = self.next_sample()
				img = apply_random_effect(img, random.randint(1, 3))
				#cv2.imwrite('img{}.jpg'.format(random.randint(1, 90)), img)
				img = img.astype(np.float32)
				img /= 255
				X_data[i] = img
				Y_data[i] = text_to_labels(text) + [0] * (self.max_plate_len - len(text))
				label_length[i] = len(text)

			inputs = {
				'input_1': X_data,
				'the_labels': Y_data,
				'input_length': input_length,
				'label_length': label_length
			}
			outputs = {'ctc': np.zeros([self.batch_size])}
			yield inputs, outputs

hu = 'he_uniform'

def small_basic_block(nr_filters):
	def func(inp):
		#found that for "relu" layers you should use "he_uniform"
		x = Conv2D(filters=nr_filters // 4, kernel_size=1, strides=1, kernel_initializer=hu, activation='relu')(inp)
		#x = BatchNormalization()(x)
		x = ZeroPadding2D(padding=(1, 0))(x)
		x = Conv2D(filters=nr_filters // 4, kernel_size=(3, 1), strides=1, kernel_initializer=hu, activation='relu')(x)
		#x = BatchNormalization()(x)
		x = ZeroPadding2D(padding=(0, 1))(x)
		x = Conv2D(filters=nr_filters // 4, kernel_size=(1, 3), strides=1, kernel_initializer=hu, activation='relu')(x)
		#x = BatchNormalization()(x)
		x = Conv2D(filters=nr_filters, kernel_size=1, strides=1, kernel_initializer=hu, activation='relu')(x)
		x = BatchNormalization()(x)
		return Activation('relu')(x)

	return func

#alphabet should be common for both train and validation datasets
# I think $ is gonna be null label, it should be last
alphabet = [' ','-','0','1','2','3','4','5','6','7','8','9','A',
			'B','C','D','E','F','G','H','I','J','K','L','M','N',
			'O','P','Q','R','S','T','U','V', 'W','X','Y','Z','$']
#max plate len also should be common
max_plate_len = 15

train_data_dir = 'plate_and_nr_dataset/train/'
train_gen = TextImageGenerator(94, 24, 32, train_data_dir, alphabet, max_plate_len)
train_gen.build_data()
validation_data_dir = 'plate_and_nr_dataset/validation/'
valid_gen = TextImageGenerator(94, 24, 20, validation_data_dir, alphabet, max_plate_len)
valid_gen.build_data()

class_number = len(alphabet)
img_width = 94  # cols
img_height = 24  # rows

if K.image_data_format() == 'channel_first':
	shape = (3, img_height, img_width)
else:
	shape = (img_height, img_width, 3) #i get this

# LPRNet
inp = Input(shape=shape)
x = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer=hu, strides=1)(inp)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
x = small_basic_block(nr_filters=128)(x)
x = Reshape((x.shape[1],x.shape[2],x.shape[3], 1))(x)#add temporary dim to reduce nr of filters(dim before the added)
#don't know why (3,3,1) but it does some necessary dim reduction
x = MaxPooling3D(padding='valid', pool_size=(3, 3, 1), strides=(2, 1, 2))(x)#we should reduce 128 to 64,div by 2
x = Reshape((x.shape[1],x.shape[2],x.shape[3]))(x)#delete added dim
x = small_basic_block(nr_filters=256)(x)
x = small_basic_block(nr_filters=256)(x)
x = Reshape((x.shape[1],x.shape[2],x.shape[3], 1))(x)#same procedure as before
x = MaxPooling3D(padding='valid', pool_size=(3, 3, 1), strides=(2, 1, 4))(x)#now reduce 256 to 64, div by 4
x = Reshape((x.shape[1],x.shape[2],x.shape[3]))(x)
x = Dropout(rate=0.5)(x)
x = Conv2D(filters=256, kernel_size=(4, 1), kernel_initializer=hu, strides=1)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(rate=0.5)(x)
x = Conv2D(filters=class_number, kernel_size=(1, 13), kernel_initializer=hu, strides=1)(x)
y_pred = BatchNormalization()(x)
y_pred = Activation('relu')(y_pred)
y_pred = Reshape((y_pred.shape[2], y_pred.shape[3]))(y_pred)

dim_of_samples = y_pred.shape[1].value

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

labels = Input(name='the_labels', shape=[max_plate_len], dtype='float32')
input_len = Input(name='input_length', shape=[1], dtype='int64')
label_len = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_len, label_len])
#sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)

model = Model(inputs=[inp, labels, input_len, label_len], outputs=loss_out)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_tru, y_prd: y_prd}, optimizer='adam')

def train(ep):
	model.fit_generator(generator=train_gen.next_batch(),
						steps_per_epoch=train_gen.batch_size,
						epochs=ep,
						validation_data=valid_gen.next_batch(),
						validation_steps=valid_gen.batch_size)

#train(1)

def save_model():
	model_json = model.to_json()
	with open('lprnet_model.json', 'w') as json_file:
		json_file.write(model_json)
	model.save_weights('lprnet_weights.h5')
	print('saved model to disk')

def load_model_weights():
	# load json and create model
	#json_file = open('model.json', 'r')
	#loaded_model_json = json_file.read()
	#json_file.close()
	#loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("lprnet_weights.h5")#probably should be done before compile
	print("Loaded model from disk")


# TESTING
import itertools

#input = batch output from lprnet
#output = best path decoded text for each sample
def decode_batch(out):
	ret = []
	bs = out.shape[0]
	for i in range(bs):
		out_best = list(np.argmax(out[i, :], 1))
		out_best = [k for k,_ in itertools.groupby(out_best)]
		st = ''.join(list(map(lambda x: alphabet[int(x)] if int(x) < len(alphabet)-1 else '', out_best)))
		ret.append(st)
	return ret

def test():
	#don't forget to load model :D
	for inp_value, _ in train_gen.next_batch():
		bs = inp_value['input_1'].shape[0]
		X_data = inp_value['input_1']
		net_out_value = sess.run(y_pred, feed_dict={inp: X_data})
		pred_texts = decode_batch(net_out_value)
		labels = inp_value['the_labels']
		texts = []
		for label in labels:
			text = labels_to_text(label)
			texts.append(text)

		for i in range(bs):
			print('Predicted: %s\nTrue: %s' % (pred_texts[i], texts[i]))
			img = X_data[i][:, :, 0]
			cv2.imshow('img',img)

			if cv2.waitKey(0) & 0xFF == ord('q'):
				break
		break

	cv2.destroyAllWindows()