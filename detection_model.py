#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model,layers

#Layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

import os
import numpy as np 

from utils import label_size, modify_label_size, input_size

num_classes = 2
class Net(Model):
    def __init__(self, classes, chanDim=-1):
        super(Net,self).__init__()
        
        # initialize the layers in the first (CONV => RELU) * 2 => POOL
        # layer set
        self.conv1A = Conv2D(16, (3, 3), padding="same",activation = tf.nn.relu)
        self.conv1B = Conv2D(16, (1, 1), padding="same",activation = tf.nn.relu)
        self.bn1A = BatchNormalization(axis=chanDim)
        self.conv1C = Conv2D(16, (3, 3), padding="same",activation = tf.nn.relu)
        self.conv1D = Conv2D(16, (1, 1), padding="same",activation = tf.nn.relu)
        self.bn1B = BatchNormalization(axis=chanDim)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        
        # initialize the layers in the second (CONV => RELU) * 2 => POOL
        # layer set
        self.conv2A = Conv2D(32, (3, 3), padding="same",activation = tf.nn.relu)
        self.conv2B = Conv2D(32, (1, 1), padding="same",activation = tf.nn.relu)
        self.bn2A = BatchNormalization(axis=chanDim)
        self.conv2C = Conv2D(32, (3, 3), padding="same",activation = tf.nn.relu)
        self.conv2D = Conv2D(32, (1, 1), padding="same",activation = tf.nn.relu)
        self.bn2B = BatchNormalization(axis=chanDim)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        
        # initialize the layers in the second (CONV => RELU) * 2 => POOL
        # layer set
        self.conv3A = Conv2D(32, (3, 3), padding="same",activation = tf.nn.relu)
        self.conv3B = Conv2D(32, (1, 1), padding="same",activation = tf.nn.relu)
        self.bn3A = BatchNormalization(axis=chanDim)
        self.conv3C = Conv2D(32, (3, 3), padding="same",activation = tf.nn.relu)
        self.conv3D = Conv2D(32, (1, 1), padding="same",activation = tf.nn.relu)
        self.bn3B = BatchNormalization(axis=chanDim)
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        
        # initialize the layers in our fully-connected layer set
        self.flatten = Flatten()
        self.dense3 = Dense(256)
        self.act3 = Activation("relu")
        self.bn3 = BatchNormalization()
        self.do3 = Dropout(0.5)
        
        # initialize the layers in our fully-connected layer set
        self.flatten = Flatten()
        self.dense4 = Dense(64)
        self.act4 = Activation("relu")
        self.bn4 = BatchNormalization()
        self.do4 = Dropout(0.5)
        
        # initialize the layers in the softmax classifier layer set
        self.dense5 = Dense(classes)
        self.softmax = Activation("softmax")
        
        # set forward pass
    def call(self, x, is_training = False):
        # build the first (CONV => RELU) * 2 => POOL layer set
        x = self.conv1A(x)
        x = self.conv1B(x)
        x = self.bn1A(x, is_training)
        x = self.conv1C(x)
        x = self.conv1D(x)
        x = self.bn1B(x, is_training)
        x = self.pool1(x)

        # build the second (CONV => RELU) * 2 => POOL layer set
        x = self.conv2A(x)
        x = self.conv2B(x)
        x = self.bn2A(x, is_training)
        x = self.conv2C(x)
        x = self.conv2D(x)
        x = self.bn2B(x, is_training)
        x = self.pool2(x)
        
        # build the second (CONV => RELU) * 2 => POOL layer set
        x = self.conv3A(x)
        x = self.conv3B(x)
        x = self.bn3A(x, is_training)
        x = self.conv3C(x)
        x = self.conv3D(x)
        x = self.bn3B(x, is_training)
        x = self.pool3(x)

        # build our FC layer set
        x = self.flatten(x)
        x = self.dense3(x)
        x = self.act3(x)
        x = self.bn3(x, is_training)
        x = self.do3(x)
        
        x = self.flatten(x)
        x = self.dense4(x)
        x = self.act4(x)
        x = self.bn4(x, is_training)
        x = self.do4(x)

        # build the softmax classifier
        x = self.dense5(x)
        if not is_training:
            x = self.softmax(x)

        # return the constructed model
        return x

if __name__ == '__main__':
    def see_model():
        net1 = Net(num_classes)
        net1.build(input_shape=(None, input_size[0], input_size[1],1))
        print(np.sum([np.prod(v.get_shape().as_list()) for v in net1.trainable_variables]))
        net1.summary()

    # 查看模型结构
    see_model()




