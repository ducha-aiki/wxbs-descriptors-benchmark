#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import math
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32, cuda.root=/usr/local/cuda/, lib.cnmem=0.5 "
sys.path.append('/usr/local/lib/python2.7/site-packages')
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Activation, Input
from keras.layers import merge, Convolution2D, Lambda, Reshape, ZeroPadding2D
from keras.optimizers import Adam
from copy import deepcopy
import theano as T #tensorflow doesn`t have atan2 function yet https://github.com/tensorflow/tensorflow/issues/6095

def L2norm(x):
    return x / K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
def L1norm(x):
    return x / K.sum(K.abs(x), axis=1, keepdims=True)
def MOD(x,a):
    return K.mod(x,a)
def EQ(x,a):
    return K.equal(x,a)
def FL(x):
    return K.floor(x)
def MulConst(x,y):
    return x * y;
def KAtan2(x):
    return T.tensor.arctan2(x[1], x[0])
def sameshape(input_shape):
    return input_shape
def KAtan2_shape(input_shape):
    return input_shape[0]
def CircularGaussKernel(kernlen=21):
    halfSize = kernlen / 2;
    r2 = halfSize*halfSize;
    sigma2 = 0.9 * r2;
    disq = 0;
    kernel = np.zeros((kernlen,kernlen))
    for y in range(kernlen):
        for x in range(kernlen):
            disq = (y - halfSize)*(y - halfSize) +  (x - halfSize)*(x - halfSize);
            if disq < r2:
                kernel[y,x] = math.exp(-disq / sigma2)
            else:
                kernel[y,x] = 0
    return kernel
def get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins):
    bin_weight_stride = int(round(2.0 * math.floor(patch_size / 2) / float(num_spatial_bins + 1)))
    bin_weight_kernel_size = int(2 * bin_weight_stride - 1);
    return bin_weight_kernel_size, bin_weight_stride
def get_sift_model(inputs, img_rows = 65, num_ang_bins = 8, num_spatial_bins = 4, clipval = 0.2, rootsift = False):
    gk = CircularGaussKernel(kernlen=img_rows)
    gauss_kernel = K.variable(value=gk)
    gauss_kernel.trainable = False
    grad_x = Convolution2D(1, 3, 1, border_mode='valid', init='uniform', name = 'gx')(inputs)
    grad_x.trainable = False
    grad_x = ZeroPadding2D(padding=(1, 0), dim_ordering='th')(grad_x)
    grad_x.trainable = False
    grad_y = Convolution2D(1, 1, 3, border_mode='valid', init='uniform', name = 'gy')(inputs)
    grad_y.trainable = False
    grad_y = ZeroPadding2D(padding=(0,1), dim_ordering='th')(grad_y)
    grad_y.trainable = False
    grad_x_2 = Lambda(lambda x: x ** 2, output_shape = sameshape)(grad_x)
    grad_y_2 = Lambda(lambda x: x ** 2, output_shape = sameshape)(grad_y)
    grad_sum_sq = merge([grad_x_2, grad_y_2], mode='sum', concat_axis=1)
    magnitude = Lambda(lambda x: x ** 0.5, output_shape = sameshape)(grad_sum_sq)
    gauss_weigthed_magn = Lambda(MulConst, arguments={'y': gauss_kernel}, output_shape = sameshape)(magnitude)
    #angle = merge([grad_y, grad_x], mode='atan2', concat_axis=1)
    angle = merge([grad_y, grad_x], mode=KAtan2, output_shape = KAtan2_shape)
    o_big =  Lambda(lambda x: (x + 2.0*math.pi)/ (2.0*math.pi) * float(num_ang_bins), output_shape = sameshape)(angle)

    bo0_big =  Lambda(FL, output_shape = sameshape)(o_big)
    munis_bo0_big =   Lambda(lambda x: -x, output_shape = sameshape)(bo0_big )
    wo1_big = merge([o_big, munis_bo0_big], mode='sum', concat_axis=1)
    
    bo0_big =  Lambda(MOD, arguments = {'a':num_ang_bins}, output_shape = sameshape)(bo0_big)
    bo0_big_plus1  = Lambda(lambda x: (x  +1.), output_shape = sameshape)(bo0_big) 
    bo1_big =  Lambda(MOD, arguments = {'a':num_ang_bins}, output_shape = sameshape)(bo0_big_plus1)
    
    wo0_big =  Lambda(lambda x: 1. - x, output_shape = sameshape)(wo1_big)
    wo0_big = merge([wo0_big, gauss_weigthed_magn], mode='mul', concat_axis=1)
    wo1_big = merge([wo1_big, gauss_weigthed_magn], mode='mul', concat_axis=1)
    ang_bins = []
    bin_weight_kernel_size, bin_weight_stride = get_bin_weight_kernel_size_and_stride(img_rows, num_spatial_bins)
    for i in range(0, num_ang_bins):
        mask1 =  Lambda(EQ, arguments = {'a': i}, output_shape = sameshape)(bo0_big)
        amask1 =  Lambda(EQ, arguments = {'a': i}, output_shape = sameshape)(bo1_big)
        weights1 = merge([mask1,wo0_big], mode = 'mul', concat_axis = 1)
        weights11 = merge([amask1,wo1_big], mode = 'mul', concat_axis = 1)
        ori0 =  merge([weights1, weights11], mode='sum', concat_axis=1)
        bin_weight = Convolution2D(1, bin_weight_kernel_size, bin_weight_kernel_size, 
                                   border_mode='valid', init='uniform',
                                   subsample = [bin_weight_stride, bin_weight_stride], 
                                   name = 'bin_weight'+str(i))(ori0)
        bin_weight.trainable = False
        ang_bins.append(bin_weight)
    ang_bin_merged = merge(ang_bins, mode='concat', concat_axis=1)
    flatten = Flatten()(ang_bin_merged)
    l2norm =  Lambda(L2norm, output_shape = sameshape)(flatten)
    clipping =  Lambda(lambda x: K.minimum(x,clipval), output_shape = sameshape)(l2norm)
    l2norm_again = Lambda(L2norm, output_shape = sameshape)(clipping)
    if rootsift:
        l2norm_again = Lambda(L1norm, output_shape = sameshape)(l2norm_again)
        l2norm_again = Lambda(K.sqrt, output_shape = sameshape)(l2norm_again)
    #l2norm_again = Reshape((8,4,4))(l2norm_again)
    return l2norm_again
def getPoolingKernel(kernel_size = 25):
    step = 1. / float(np.floor( kernel_size / 2.));
    x_coef = np.arange(step/2., 1. ,step)
    xc2 = np.hstack([x_coef,[1], x_coef[::-1]])
    kernel = np.outer(xc2.T,xc2)
    kernel = np.maximum(0,kernel)
    return kernel
def initializeSIFT(model):
    for layer in model.layers:
        l_name = layer.get_config()['name']
        if l_name == 'gy':
            new_weights = np.array([[[[-1, 0, 1]]]], dtype=np.float32)
        elif l_name == 'gx':
            new_weights = np.array([[[[-1], [0], [1]]]], dtype=np.float32)
        elif 'bin_weight' in l_name:
            kernel_size  = layer.get_weights()[0].shape[-1]
            nw = getPoolingKernel(kernel_size = kernel_size)
            new_weights = np.array(nw.reshape((1, 1, kernel_size, kernel_size)))
        else:
            continue
        w_all = layer.get_weights()
        biases = np.array(w_all[1])
        biases[:] = 0
        w_all_new = [new_weights, biases]
        layer.set_weights(w_all_new)
        layer.trainable = False
    #print('Weights are preloaded')
    return model
def getCompiledSIFTModel(patch_size = 65, rootsift = False):
    inputs = Input((1, patch_size, patch_size), name='main_input')
    kerassift = get_sift_model(inputs, rootsift = rootsift)
    model = Model(input=inputs, output=kerassift)
    model.compile(optimizer=Adam(1e-5), loss='mse')
    model = initializeSIFT(model)
    return model