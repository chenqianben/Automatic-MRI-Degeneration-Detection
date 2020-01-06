#!/usr/bin/env python
# coding: utf-8

# In[3]:


from data import read_train_data

from skimage.transform import resize
import skimage.io as io
import cv2
import pydicom  # process dcm files

import numpy as np
import matplotlib.pyplot as plt


# In[5]:


train_data = read_train_data()
print(train_data.shape)


# In[ ]:




