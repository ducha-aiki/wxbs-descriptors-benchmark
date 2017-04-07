#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import os
import sys
from  scipy.misc import imresize as resize
import cv2
import time
import numpy as np
try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
except:
    print "Wrong input format. Try Pixels11.py img.jpg out.txt"
    sys.exit(1)
image = cv2.imread(input_img_fname,0)
h,w = image.shape
n_patches =  h/w
patches_flat = np.ndarray((n_patches, 11*11), dtype=np.float32)
t = time.time()
for i in range(n_patches):
    patch =  image[i*(w): (i+1)*(w), 0:w]
    patch = resize(patch, (11,11))
    std1 = patch.std()
    if std1 < 0.000001:
        std1 = 1.
    patch = (patch - patch.mean()) / std1
    norm1 = np.linalg.norm(patch.flatten(),2)
    if norm1 < 0.000000001:
        norm1 = 1.0
    patch = patch /  norm1
    patches_flat[i,:] = patch.flatten()
np.savetxt(output_fname, patches_flat, delimiter=' ', fmt='%10.7f')
