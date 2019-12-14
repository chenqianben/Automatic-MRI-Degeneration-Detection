#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from skimage.filters import gaussian, median
from skimage.morphology import disk

from skimage.feature import canny
from skimage.measure import find_contours

from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture

import numpy as np


# In[ ]:


def kmeans(im,n):
    data = im.ravel()[:,np.newaxis]

    model = KMeans(n_clusters = n)
    model.fit(data)
    label_pred = model.labels_
    label_pred = label_pred.reshape(512,512)
    if np.mean(label_pred) < 0.5:
        label_pred = np.where(label_pred==0,1,0)
    return label_pred

def gmm(im,n):
    data = im.ravel()[:,np.newaxis]
    model = GaussianMixture(n_components = n)
    model.fit(data)
    label_pred = model.predict(data)
    label_pred = label_pred.reshape(512,512)
    if np.mean(label_pred) < 0.5:
        label_pred = np.where(label_pred==0,1,0)
    return label_pred

