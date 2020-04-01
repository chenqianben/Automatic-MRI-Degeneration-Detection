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
    
    def get_values(self, rois_ens, Y1_pred, Y2_pred):
        Y1_mean = []
        Y1_max = []
        Y1_min = []
        Y2_mean = []
        Y2_max = []
        Y2_min = []
        for rois, y1_pred, y2_pred in zip(rois_ens, Y1_pred, Y2_pred):
            #y1_pred = (y1_pred > y1_pred.mean())   # (8, 36, 36),如果加上这段代码，就是希望做0/1的mask,否则就是smooth的mask
            n1_pred = y1_pred.sum()                # 对八张图片全部加和
            #y2_pred = (y2_pred > y2_pred.mean())
            n2_pred = y2_pred.sum() 
            
            y1_mean, y1_max, y1_min = (rois * y1_pred).sum()/n1_pred, (rois * y1_pred).max(), (rois * y1_pred).min()
            y2_mean, y2_max, y2_min = (rois * y2_pred).sum()/n2_pred, (rois * y2_pred).max(), (rois * y2_pred).min()
        
            Y1_mean.append(y1_mean)
            Y1_max.append(y1_max)
            Y1_min.append(y1_min)
            Y2_mean.append(y2_mean)
            Y2_max.append(y2_max)
            Y2_min.append(y2_min)
            
        data_pred = pd.DataFrame(zip(names, Y1_mean, Y1_max, Y1_min, Y2_mean, Y2_max, Y2_min),
                                 columns=['path',
                                         'valeur moyenne (centre)','valeur max (centre)','valeur min (centre)',
                                         'valeur moyenne (cote)','valeur max (cote)','valeur min (cote)'])
        return data_pred
    
    def draw_figures(self, rois_ens, Y1_pred, Y2_pred):
        for j, (rois, y1_pred, y2_pred) in enumerate(zip(rois_ens, Y1_pred, Y2_pred)):
            print(names[j])
            plt.figure(figsize = (14, 7))
            for i, (roi, y1_p, y2_p) in enumerate(zip(rois, y1_pred, y2_pred)):
                plt.subplot(3,7,i+1)
                plt.imshow(roi, cmap='gray')
                plt.subplot(3,7,i+1+7)
                plt.imshow(y1_p, cmap='gray')
                plt.subplot(3,7,i+1+2*7)
                plt.imshow(y2_p, cmap='gray')
            plt.show()

    def find_values(self, rois, y1_pred, y2_pred):
        for roi, y1_p, y2_p in zip(rois, y1_pred, y2_pred):
            