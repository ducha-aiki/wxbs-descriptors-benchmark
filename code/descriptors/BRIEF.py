#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import os
import sys
#sys.path.insert(0, '/home/ubuntu/dev/opencv-3.1/build/lib')
from aux.numpy_sift import SIFTDescriptor 
import cv2
import time
import numpy as np
from skimage.feature import BRIEF
try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
except:
    print ("Wrong input format. Try BRIEF.py img.jpg out.txt")
    sys.exit(1)
image = cv2.imread(input_img_fname,0)
h,w = image.shape
print(h,w)
BR = BRIEF(patch_size = w - 1)
n_patches =  h/w
keypoints = np.zeros((n_patches,2))
t = time.time()
for i in range(n_patches):
    keypoints[i,:] = np.array([i*w + float(w)/2., float(w)/2.])
BR.extract(image, keypoints)
descriptors_for_net = BR.descriptors
np.savetxt(output_fname, descriptors_for_net, delimiter=' ', fmt='%i')
