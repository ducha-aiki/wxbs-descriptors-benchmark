#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import os
import sys
#sys.path.insert(0, '/home/ubuntu/dev/opencv-3.1/build/lib')
from aux.numpy_sift import SIFTDescriptor 
import cv2
import time
import numpy as np
try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
except:
    print ("Wrong input format. Try pySIFT.py img.jpg out.txt")
    sys.exit(1)
image = cv2.imread(input_img_fname,0)
h,w = image.shape
SD = SIFTDescriptor(patchSize = w)
n_patches =  h/w
descriptors_for_net = np.zeros((n_patches, 128))
t = time.time()
for i in range(n_patches):
    patch =  image[i*(w): (i+1)*(w), 0:w]
    descriptors_for_net[i,:] =  SD.describe(patch).flatten()# / 255.
np.savetxt(output_fname, descriptors_for_net.astype(np.int32), delimiter=' ', fmt='%i')
