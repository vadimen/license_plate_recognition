#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:07:36 2020

@author: vadim
"""

import cv2
import numpy as np
import sys
import time
from openalpr import Alpr

alpr = Alpr("us", "/home/vadim/software/openalpr/config/openalpr.conf", "/home/vadim/software/openalpr/runtime_data/")
if not alpr.is_loaded():
    print("Error loading alpr")
    sys.exit(1)
    
alpr.set_top_n(20)
alpr.set_default_region("md")

cap = cv2.VideoCapture("/home/vadim/Desktop/video2.mp4")
COLORS = np.array([0,255,0,3], dtype=float)
COLORS2 = np.array([0,215,0,50], dtype=float)

print("loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

while(True):
    ret, image = cap.read()
    if ret:
        start_time = time.time()
        (h,w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        for i in np.arange(0, detections.shape[2]):
            #extract the index of the claass label
            idx = int(detections[0,0,i,1])
            if idx == 7: #if it's 'car' class
                confidence = detections[0,0,i,2]
                if confidence>0.2:
                    box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    label = "{}: {:.2f}%".format('car', confidence*100)
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS, 2)
                    y = startY-15 if startY-15>15 else startY+15
                    cv2.putText(image,label,(startX,y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)
                    
                    if startX<0: startX = 0
                    if endX<0: endX = 0
                    if startY<0: startY = 0
                    if endY<0: endY = 0
                    
                    img2 = image[startY:endY, startX:endX]
                    #cv2.imshow('test',img2)
                    #cv2.imwrite("img{}.jpg".format(i), img2)
                    results = alpr.recognize_ndarray(img2)
                    if len(results['results'])>0: 
                        print("|||||||||||||||||||||||||||||||")
                        print("plate        --> {}".format(results['results'][0]['candidates'][0]['plate']))
                        print("confidence   --> {}".format(results['results'][0]['candidates'][0]['confidence']))
                    
                    fps = "{:3.0f} FPS".format(1/(time.time()-start_time))
                    cv2.putText(image, fps, (w-85, h-50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS2, 2)
        
        cv2.imshow('frame',image)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()