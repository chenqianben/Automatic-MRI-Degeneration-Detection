#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Lambda                       #Lambda wraps arbitrary expression as a Layer object.
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import add, concatenate

#To save and reload models
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model

#For data augmentation
from keras.preprocessing.image import ImageDataGenerator 

import tensorflow as tf
from keras import backend as K 
import keras
import numpy as np

from utils import img_height, img_width


# In[2]:


def block(inp, output_channels, k_size, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal', if_pooling = True):
    inp = Conv2D(output_channels, k_size, padding = padding, activation = activation, kernel_initializer = kernel_initializer)(inp)
    inp = Conv2D(output_channels, k_size, padding = padding, activation = activation, kernel_initializer = kernel_initializer)(inp)
    if if_pooling:
        inp = MaxPooling2D(k_size, strides = (2,2), padding = padding)(inp)
    return inp 


# In[8]:


# Design our model architecture here
def unet_model(input_shape):
    n_ch_exps = [3, 4, 5, 6, 7, 7]
    kernels = (5, 5)
    
    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (1, input_shape[0], input_shape[1])
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
    input_shape = (input_shape[0], input_shape[1], 1)

    inp = Input(shape=input_shape)
    encodeds = []

    # encoder
    enc = inp
    for l_idx, n_ch in enumerate(n_ch_exps):
        enc = Conv2D(2 ** n_ch, kernels,
                     strides=(2, 2), padding='same',
                     kernel_initializer='he_normal')(enc)
        #enc = BatchNormalization()(enc)
        enc = LeakyReLU(name='encoded_{}'.format(l_idx),
                        alpha=0.2)(enc)
        encodeds.append(enc)

    # decoder
    dec = enc
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  
        #dec = Conv2D(2 ** n_ch, kernels,padding='same',kernel_initializer='he_normal')(dec)
        dec = Conv2DTranspose(2 ** n_ch, kernels,
                              strides=(2, 2), padding='same',
                              kernel_initializer='he_normal',
                              activation='relu',
                              name='decoded_{}'.format(l_idx))(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]],
                          axis=ch_axis)

    outp = Conv2DTranspose(1, kernels,
                           strides=(2, 2), padding='same',
                           kernel_initializer='glorot_normal',
                           activation='sigmoid',
                           name='decoded_{}'.format(l_idx + 1))(dec)

    unet = Model(inputs=inp, outputs=outp)
    
    return unet


# In[9]:


# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):                       
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer()) 
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1. 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def costum_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)







