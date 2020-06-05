#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this works in both tf v1 and v2
Created on Sun Jan 19 21:10:03 2020

@author: vadim

aici e ssd mobilenet antrenat de mine in tensorflow pt a detecta doar license plates
"""
zz
import numpy as np
import tensorflow as tf
import pathlib
import cv2
import time
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
path = '/home/vadim/Desktop/udemy_advanced_computer_vision/TensorFlow/workspace/training_demo/trained-inference-graphs/my_trained_ssd_mobilenet_for_license_plates_2/saved_model'
detection_model = tf.saved_model.load(str(path))
detection_model = detection_model.signatures['serving_default']

# loading label map
PATH_TO_LABELS = '/home/vadim/Desktop/udemy_advanced_computer_vision/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # input trebuie sa fie tensor, convertim folosind `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # model asteapta un batch de imagini,deci mai adaugam o axa cu `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    # facem predictia
    output_dict = model(input_tensor)
    # iesirile sunt batchuri
    # convertim in vectori numpy, si luam index [0] pentru a scoate dimensiunea de batch
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes trebuie sa fie ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    return output_dict


def show_inference_for_single_image(model, image_path):
    image_np = cv2.imread(image_path)
    output_dict = run_inference_for_single_image(model, image_np)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imshow("image detection", image_np)
    global k
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()


# cap = cv2.VideoCapture("/home/vadim/Desktop/video.mp4")
#
# while True:
# 	ret, image = cap.read()
# 	
# 	if ret==True:
# 		start_time = time.time()
# 		
# 		image_np = image
# 		output_dict = run_inference_for_single_image(detection_model, image_np)
# 		
# 		vis_util.visualize_boxes_and_labels_on_image_array(
# 					image_np,
# 				output_dict['detection_boxes'],
# 				output_dict['detection_classes'],
# 				output_dict['detection_scores'],
# 				category_index,
# 				instance_masks=output_dict.get('detection_masks_reframed', None),
# 				use_normalized_coordinates=True,
# 				line_thickness=8)
# 		
# 		print('{:.2f} FPS'.format(1/(time.time()-start_time)))
# 		cv2.imshow("image detection", image)
# 	else: break

# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 			break
# cv2.destroyAllWindows()

import os

path = "/home/vadim/Desktop/license-plates_images/040603/"
arr = os.listdir(path)
k = 0

for i in arr:
    show_inference_for_single_image(detection_model, path + "/" + i)
    if k == ord('q'):
        break
cv2.destroyAllWindows()

# show_inference_for_single_image(detection_model, "/home/vadim/Desktop/Diploma_Project/opencvMobileNetSSDObjectDetection/img0.jpg")
