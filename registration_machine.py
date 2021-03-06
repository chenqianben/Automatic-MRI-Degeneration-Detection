#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pydicom
import numpy as np
from skimage.transform import resize,rescale
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import os
import pandas as pd

# hough ellipse transform
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse, resize
from skimage.draw import ellipse_perimeter
from skimage.color import gray2rgb
import skimage.io as io

# inmediate data process
from utils import normalize, get_rois, plot_axis, read_data
from utils import read_rois_and_axis_data

# segmentation
from segmentation_model import unet_model

# detection
from detection_model import Net, num_classes
from detection_machine import Detection

# hyperparameters
from utils import input_size, img_height, img_width


# In[ ]:


class Semi_auto_registration():
    def __init__(self, Y1_pred, Y2_pred, src_rois_axis_ens, src_input_size, 
                 ims_T1, ims_T2, ims_T2star, axis_ens_T1, axis_ens_T2, axis_ens_T2stare):
        self.Y1_pred = Y1_pred
        self.Y2_pred = Y2_pred
        self.ims_T1 = ims_T1
        self.ims_T2 = ims_T2
        self.ims_T2star = ims_T2star
        self.axis_ens_T1 = axis_ens_T1
        self.axis_ens_T2 = axis_ens_T2
        self.axis_ens_T2star = axis_ens_T2star
        self.src_size = (512,512)
        self.src_roi_size = Y1_pred[0].shape[1:]
        self.src_input_size = src_input_size
        self.src_rois_axis_ens = src_rois_axis_ens.copy()
        
    def get_rois(self, type_im):
        if type_im == 'T1':
            ims = self.ims_T1
            tgt_axis_ens = self.axis_ens_T1
        elif type_im == 'T2':
            ims = self.ims_T2
            tgt_axis_ens = self.axis_ens_T2  
        elif type_im == 'T2star':
            ims = self.ims_T2
            tgt_axis_ens = self.axis_ens_T2
        else:
            print('wrong type(should be T1, T2 or T2star)')
            return
        
        rois_ens = []
        Y1_pred_ens = []
        Y2_pred_ens = []
        for i, (im, tgt_axis) in enumerate(zip(ims, tgt_axis_ens)):
            y1_pred = self.Y1_pred[i].copy()
            y2_pred = self.Y2_pred[i].copy()
            
            # 注意下面src何tgt都有按照y轴从上到下排序的过程
            # registration of mask
            tgt_size = im.shape
            tgt_roi_size = (int(tgt_size[0]/self.src_size[0]*self.src_roi_size[0]), int(tgt_size[1]/self.src_size[1]*self.src_roi_size[1]))
            y1_pred = resize(y1_pred.transpose(1,2,0), tgt_roi_size).transpose(2,0,1) 
            y2_pred = resize(y2_pred.transpose(1,2,0), tgt_roi_size).transpose(2,0,1) 
            
            order = np.argsort(self.src_rois_axis_ens[i][:,1])
            y1_pred = y1_pred[order]
            y2_pred = y2_pred[order]
            
            # read rois
            rois, tgt_axis = self.get_rois_from_points_axis(im, tgt_axis, tgt_roi_size)
            order = np.argsort(tgt_axis[:,1])
            rois = rois[order]
            
            while len(rois) < len(y1_pred):
                y1_pred = np.delete(y1_pred, obj = 0, axis = 0)
                y2_pred = np.delete(y2_pred, obj = 0, axis = 0)
                
            rois_ens.append(rois)
            Y1_pred_ens.append(y1_pred)
            Y2_pred_ens.append(y2_pred)
            
        return rois_ens, Y1_pred_ens, Y2_pred_ens
                
    def get_rois_from_points_axis(self, im, axis, rec_size):
        rois = []
        new_axis = []
        for x,y in axis:
            if y+rec_size[0]//2 < im.shape[0] and y-rec_size[0]//2>0:
                roi = im[y-rec_size[0]//2:y+rec_size[0]//2, x-rec_size[1]//2:x+rec_size[1]//2]
                rois.append(roi)
                new_axis.append([x,y])
        return np.array(rois), np.array(new_axis)
    
    def get_values(self, rois_ens, Y1_pred, Y2_pred, th):
        Y1_mean = []
        Y1_max = []
        Y1_min = []
        Y2_mean = []
        Y2_max = []
        Y2_min = []
        names_ens = []
        for i,(rois, y1_pred, y2_pred) in enumerate(zip(rois_ens, Y1_pred, Y2_pred)): 
            Y1_mean.append(None)
            Y1_max.append(None)
            Y1_min.append(None)
            Y2_mean.append(None)
            Y2_max.append(None)
            Y2_min.append(None)
            names_ens.append(names[i])
            for roi, y1_p, y2_p in zip(rois, y1_pred, y2_pred):
                (y1_min, y1_mean, y1_max), (y2_min, y2_mean, y2_max) = self.find_values(roi, y1_p, y2_p, th)
        
                Y1_mean.append(y1_mean)
                Y1_max.append(y1_max)
                Y1_min.append(y1_min)
                Y2_mean.append(y2_mean)
                Y2_max.append(y2_max)
                Y2_min.append(y2_min)
                names_ens.append(names[i])
            
        data_pred = pd.DataFrame(zip(names_ens, Y1_mean, Y1_max, Y1_min, Y2_mean, Y2_max, Y2_min),
                                 columns=['path',
                                         'valeur moyenne (centre)','valeur max (centre)','valeur min (centre)',
                                         'valeur moyenne (cote)','valeur max (cote)','valeur min (cote)'])
        return data_pred
    
    def draw_figures(self, tgt_rois_ens, Y1_pred, Y2_pred,th):
        for j, (tgt_rois, y1_pred, y2_pred) in enumerate(zip(tgt_rois_ens, Y1_pred, Y2_pred)):
            print(names[j])
            plt.figure(figsize = (14, 7))
            for i, (tgt_roi, y1_p, y2_p) in enumerate(zip(tgt_rois, y1_pred, y2_pred)):   
                y1_p_copy = y1_p.copy()
                y2_p_copy = y2_p.copy()
                y1_p_copy[y1_p_copy>th] = 1 
                y1_p_copy[y1_p_copy<th] = 0
                y2_p_copy[y2_p_copy>th] = 1 
                y2_p_copy[y2_p_copy<th] = 0
                
                plt.subplot(6,7,i+1)
                plt.imshow(tgt_roi, cmap='gray')
                plt.subplot(6,7,i+1+7)
                plt.imshow(rois_ens[j][i].squeeze(), cmap='gray')
                plt.subplot(6,7,i+1+2*7)
                plt.imshow(y1_p, cmap='gray')
                plt.subplot(6,7,i+1+3*7)
                plt.imshow(y1_p_copy, cmap='gray')
                plt.subplot(6,7,i+1+4*7)
                plt.imshow(y2_p, cmap='gray')
                plt.subplot(6,7,i+1+5*7)
                plt.imshow(y2_p_copy, cmap='gray')
            plt.show()

    def find_values(self, roi, y1_p, y2_p, th):
        # remove overlap
        y1_p[y2_p>y1_p] = 0
        y2_p[y1_p>y2_p] = 0
        
        # make mask
        y1_p_copy = y1_p.copy()
        y2_p_copy = y2_p.copy()
        
        y1_p_copy[y1_p_copy>th] = 1 
        y1_p_copy[y1_p_copy<th] = 0

        y2_p_copy[y2_p_copy>th] = 1 
        y2_p_copy[y2_p_copy<th] = 0
        
        w1 = y1_p.sum()
        w2 = y2_p.sum()
        
        # max
        max1 = (roi*y1_p_copy).max()
        max2 = (roi*y2_p_copy).max()

        # mean
        mean1 = (roi*y1_p).sum()/w1
        mean2 = (roi*y2_p).sum()/w2

        # min
        min1 = np.min((roi*y1_p_copy)[(roi*y1_p_copy).nonzero()])    # 找到非0以外的最小值
        min2 = np.min((roi*y2_p_copy)[(roi*y2_p_copy).nonzero()])
        
        return (min1, mean1, max1), (min2, mean2, max2)