#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import shutil   # copy file

from skimage.transform import resize
import skimage.io as io
import pydicom  # process dcm files
from skimage.feature import hog
from skimage.exposure import adjust_gamma
from read_roi import read_roi_file, read_roi_zip

import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
import random

from utils import label_size, modify_label_size, input_size
from utils import rois_dict_to_axis, get_rois, read_data


if_normalized = True
if_modify_size = True


# read training data of dcm files
root_path = 'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_brut'
ims_T1,_,ims_T2,_,_,pos,_,_,_ = read_data(root_path, if_normalized=True)

# 1. read positive labels
def read_pos_labels(dir_labels, dir_data, type_data=1, if_modify_size = True, if_normalized = True):
    '''
    input:
    dir_labels: the directory where labels are restored 
    dir_data: the directory where corresponding data are restored
    type_data: 1(respectively 2) symbols T1 data(respectively T2 data)
    
    output:
    train,test,val data where each element is a ensemble of negative labels in one image
    
    '''
    train_inds = np.arange(1,12)
    val_inds = np.arange(12,16)
    
    train_labels = []
    val_labels = []
    for root, dirnames, filenames in os.walk(dir_labels):
        for filename in filenames:
            ind_brebis = filename[:-6]
            ind_jour = filename[-5]
        
            rois_dict = read_roi_zip(os.path.join(dir_labels,filename))
            axis_ens = np.array(rois_dict_to_axis(rois_dict))
            
            type_data = 'T2_TSE_SAG' if type_data == 2 else 'T1_TSE_SAG'
            dir_a_data = os.path.join(os.path.join(os.path.join(dir_data,'Brebis'+ind_brebis),ind_jour),type_data)
            for f_root, _, f_data in os.walk(dir_a_data):
                im = io.imread(os.path.join(f_root,f_data[0]), as_gray=True)
                if if_normalized:
                    im = (im-im.mean())/im.std()
                break
                
            im_rois = np.array(get_rois(im, axis_ens, if_modify_size))
            if int(ind_brebis) in train_inds:
                train_labels.append(im_rois)
            elif int(ind_brebis) in val_inds:
                val_labels.append(im_rois)
                
    train_labels = np.concatenate(train_labels, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    return train_labels, val_labels

dir_labels = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\positive ROI\T1 SAG'
dir_data = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process'
train_pos_labels_T1, val_pos_labels_T1 = read_pos_labels(dir_labels, dir_data, 1,if_modify_size,if_normalized)
             
dir_labels = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\positive ROI\T2 SAG'
dir_data = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process'
train_pos_labels_T2, val_pos_labels_T2 = read_pos_labels(dir_labels, dir_data, 2,if_modify_size,if_normalized)


# data augmentation crop
def crop_labels_into_inputs(ims, input_size):
    '''crop labels(modify_label_size) into inputs(input_size)'''
    inputs = []
    if input_size[0] > ims.shape[1] or input_size[1] > ims.shape[2]:
        return 
    for im in ims:
        for i in range(ims.shape[1]-input_size[0]+1):
            for j in range(ims.shape[2]-input_size[1]+1):
                inputs.append(im[i:i+input_size[0], j:j+input_size[1]])
    return np.array(inputs)

train_pos_labels_T1 = crop_labels_into_inputs(train_pos_labels_T1, input_size)
val_pos_labels_T1 = crop_labels_into_inputs(val_pos_labels_T1, input_size)

train_pos_labels_T2 = crop_labels_into_inputs(train_pos_labels_T2, input_size)
val_pos_labels_T2 = crop_labels_into_inputs(val_pos_labels_T2, input_size)

# read specific neg labels
def read_neg_labels(dir_labels, dir_data, type_data=1, if_modify_size = True, if_normalized = True):
    '''
    input:
    dir_labels: the directory where labels are restored 
    dir_data: the directory where corresponding data are restored
    type_data: 1(respectively 2) symbols T1 data(respectively T2 data)
    
    output:
    train,test,val data where each element is a ensemble of negative labels in one image
    
    '''
    train_inds = np.arange(1,12)
    val_inds = np.arange(12,16)
    
    train_labels = []
    val_labels = []

    for root, dirnames, filenames in os.walk(dir_labels):
        for filename in filenames:
            ind_brebis = filename[:-6]
            ind_jour = filename[-5]
            
            rois_dict = read_roi_zip(os.path.join(dir_labels,filename))
            axis_ens = np.array(rois_dict_to_axis(rois_dict))
            
            type_data = 'T2_TSE_SAG' if type_data == 2 else 'T1_TSE_SAG'
            dir_a_data = os.path.join(os.path.join(os.path.join(dir_data,'Brebis'+ind_brebis),ind_jour),type_data)
            for f_root, _, f_data in os.walk(dir_a_data):
                im = io.imread(os.path.join(f_root,f_data[0]), as_gray=True)
                if if_normalized:
                    im = (im-im.mean())/im.std()
                break
                
            im_rois = np.array(get_rois(im, axis_ens, if_modify_size))
            if int(ind_brebis) in train_inds:
                train_labels.append(im_rois)
            elif int(ind_brebis) in val_inds:
                val_labels.append(im_rois)

    train_labels = np.concatenate(train_labels, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    return train_labels, val_labels
dir_labels = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\negative ROI\T1 SAG'
dir_data = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process'
train_neg_labels_T1s, val_neg_labels_T1s = read_neg_labels(dir_labels, dir_data, 1,if_modify_size,if_normalized)
             
dir_labels = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\negative ROI\T2 SAG'
dir_data = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process'
train_neg_labels_T2s, val_neg_labels_T2s = read_neg_labels(dir_labels, dir_data, 2,if_modify_size,if_normalized)


train_neg_labels_T1s = crop_labels_into_inputs(train_neg_labels_T1s, input_size)
val_neg_labels_T1s = crop_labels_into_inputs(val_neg_labels_T1s, input_size)

train_neg_labels_T2s = crop_labels_into_inputs(train_neg_labels_T2s, input_size)
val_neg_labels_T2s = crop_labels_into_inputs(val_neg_labels_T2s, input_size)


def get_neg_labels(dir_labels, dir_data, type_data=1, N_per_im = 7, if_modify_size = True, if_normalized = True):
    '''
    randomly read negative labels from images, but should avoid having positive labels
    
    input:
    dir_labels: the directory where labels are restored 
    dir_data: the directory where corresponding data are restored
    type_data: 1(respectively 2) symbols T1 data(respectively T2 data)
    
    output:
    train,test,val data where each element is a ensemble of negative labels in one image
    
    '''    
    size = modify_label_size if  if_modify_size else label_size
    
    train_inds = np.arange(1,12)
    val_inds = np.arange(12,16)
    
    train_labels = []
    val_labels = []

    for root, dirnames, filenames in os.walk(dir_labels):
        for filename in filenames:
            # read positive indices to avoid adding them into the negative labels
            ind_brebis = filename[:-6]
            ind_jour = filename[-5]
            rois_dict = read_roi_zip(os.path.join(dir_labels,filename))
            axis_ens = np.array(rois_dict_to_axis(rois_dict))
            
            type_data = 'T2_TSE_SAG' if type_data == 2 else 'T1_TSE_SAG'
            dir_a_data = os.path.join(os.path.join(os.path.join(dir_data,'Brebis'+ind_brebis),ind_jour),type_data)
            for f_root, _, f_data in os.walk(dir_a_data):
                im = io.imread(os.path.join(f_root,f_data[0]), as_gray=True)
                if if_normalized:
                    im = (im-im.mean())/im.std()
                break
            
            j = 0
            while j < N_per_im:
                if_chosen = True
                cx = np.random.randint(im.shape[1] - size[1] + 1)
                cy = np.random.randint(im.shape[0] - size[0] + 1)
                
                for x1, y1, _, _ in axis_ens:
                    if abs(cx-x1)<size[1]/2 and abs(cy-y1)<size[0]/2:
                        if_chosen = False
                        break
                
                if if_chosen:
                    if int(ind_brebis) in train_inds:
                        train_labels.append(im[cy:cy+size[1], cx:cx+size[0]])
                    elif int(ind_brebis) in val_inds:
                        val_labels.append(im[cy:cy+size[1], cx:cx+size[0]])
                    j += 1
                    
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    return train_labels, val_labels

def get_neg_neiborhood_labels(dir_labels, dir_data, dist, type_data=1, if_modify_size = True, if_normalized = True):
    '''
    randomly read negative labels from the neiborhood of ROIs
    
    '''
    size = modify_label_size if  if_modify_size else label_size
    
    train_inds = np.arange(1,12)
    val_inds = np.arange(12,16)
    
    train_labels = []
    val_labels = []
    for root, dirnames, filenames in os.walk(dir_labels):
        for filename in filenames:
            # read positive indices to avoid adding them into the negative labels
            ind_brebis = filename[:-6]
            ind_jour = filename[-5]
            rois_dict = read_roi_zip(os.path.join(dir_labels,filename))
            axis_ens = np.array(rois_dict_to_axis(rois_dict))
            
            type_data = 'T2_TSE_SAG' if type_data == 2 else 'T1_TSE_SAG'
            dir_a_data = os.path.join(os.path.join(os.path.join(dir_data,'Brebis'+ind_brebis),ind_jour),type_data)
            for f_root, _, f_data in os.walk(dir_a_data):
                im = io.imread(os.path.join(f_root,f_data[0]), as_gray=True)
                if if_normalized:
                    im = (im-im.mean())/im.std()
                break

            for x1, y1, w, h in axis_ens:
                d1 = np.round(w/2).astype(np.int)
                d2 = np.round(h/2).astype(np.int)
                
                coors = []
                
                for i in range(-d1-dist,d1+dist+1,2):
                    coors.append((x1+i,y1-d2-dist))
                    coors.append((x1+i,y1+d2+dist+1))
                
                for j in range(-d2-dist,d2+dist+1,2):
                    coors.append((x1-d1-dist,y1))
                    coors.append((x1+d1+dist+1,y1))
                
                for cx,cy in coors:
                    if 0 <= cy < 512 and 0 <= cx < 512 and 0 <= cy+size[1] < 512 and 0 <= cx+size[0] < 512:
                        if int(ind_brebis) in train_inds:
                            train_labels.append(im[cy:cy+size[1], cx:cx+size[0]])
                        elif int(ind_brebis) in val_inds:
                            val_labels.append(im[cy:cy+size[1], cx:cx+size[0]])
                
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    return train_labels, val_labels

def cut_size(ims, size):
    size_ori = ims.shape[1:]
    dw, dh = (size_ori[0] - size[0])//2, (size_ori[1] - size[1])//2
    return ims[:,dw:size_ori[0]-dw, dh:size_ori[1]-dh]

dist = label_size[0]//3

# 注意：negative labels由三部分组成
# T1
dir_labels = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\positive ROI\T1 SAG'
dir_data = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process'
# random neg labels
train_neg_labels_T1, val_neg_labels_T1 = get_neg_labels(dir_labels, dir_data, 1, 3,if_modify_size,if_normalized)
# random neg neighborhood labels and reducection size
train_neg_neighborhood_labels_T1, val_neg_neighborhood_labels_T1 = get_neg_neiborhood_labels(dir_labels, dir_data, dist, 1, if_modify_size, if_normalized)
# concatenation      
train_neg_labels_T1 = np.concatenate((train_neg_labels_T1,train_neg_neighborhood_labels_T1), axis=0)
val_neg_labels_T1 = np.concatenate((val_neg_labels_T1,val_neg_neighborhood_labels_T1), axis=0)
# cut_size from 42x42 to 36x36
train_neg_labels_T1 = cut_size(train_neg_labels_T1, input_size)
val_neg_labels_T1 = cut_size(val_neg_labels_T1, input_size)
# concatenation
train_neg_labels_T1 = np.concatenate((train_neg_labels_T1,train_neg_labels_T1s), axis=0)
val_neg_labels_T1 = np.concatenate((val_neg_labels_T1,val_neg_labels_T1s), axis=0)
# reduce size
train_neg_labels_T1 = train_neg_labels_T1[random.sample(list(np.arange(train_neg_labels_T1.shape[0])),train_pos_labels_T1.shape[0])]
val_neg_labels_T1 = val_neg_labels_T1[random.sample(list(np.arange(val_neg_labels_T1.shape[0])),val_pos_labels_T1.shape[0])]

# T2
dir_labels = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\positive ROI\T2 SAG'
dir_data = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process'
# random neg labels
train_neg_labels_T2, val_neg_labels_T2 = get_neg_labels(dir_labels, dir_data, 2, 7,if_modify_size,if_normalized)
# random neg neighborhood labels and reducection size
train_neg_neighborhood_labels_T2, val_neg_neighborhood_labels_T2 = get_neg_neiborhood_labels(dir_labels, dir_data, dist, 2, if_modify_size, if_normalized)
# concatenation
train_neg_labels_T2 = np.concatenate((train_neg_labels_T2,train_neg_neighborhood_labels_T2), axis=0)
val_neg_labels_T2 = np.concatenate((val_neg_labels_T2,val_neg_neighborhood_labels_T2), axis=0)
# cut size from 42x42 to 36x36
train_neg_labels_T2 = cut_size(train_neg_labels_T2, input_size)
val_neg_labels_T2 = cut_size(val_neg_labels_T2, input_size)
# concatenation
train_neg_labels_T2 = np.concatenate((train_neg_labels_T2,train_neg_labels_T2s), axis=0)
val_neg_labels_T2 = np.concatenate((val_neg_labels_T2,val_neg_labels_T2s), axis=0)
# reduce size
train_neg_labels_T2 = train_neg_labels_T2[random.sample(list(np.arange(train_neg_labels_T2.shape[0])),train_pos_labels_T2.shape[0])]
val_neg_labels_T2 = val_neg_labels_T2[random.sample(list(np.arange(val_neg_labels_T2.shape[0])),val_pos_labels_T2.shape[0])]

# x: concatenation and format casting
x_trainT1 = np.concatenate((train_pos_labels_T1,train_neg_labels_T1), axis = 0).astype(np.float32)
x_valT1 = np.concatenate((val_pos_labels_T1,val_neg_labels_T1), axis = 0).astype(np.float32)
x_trainT2 = np.concatenate((train_pos_labels_T2,train_neg_labels_T2), axis = 0).astype(np.float32)
x_valT2 = np.concatenate((val_pos_labels_T2,val_neg_labels_T2), axis = 0).astype(np.float32)

# y: 
y_trainT1 = np.concatenate((np.ones(train_pos_labels_T1.shape[0]),np.zeros(train_neg_labels_T1.shape[0])), axis = 0).astype(np.int64)
y_valT1 = np.concatenate((np.ones(val_pos_labels_T1.shape[0]),np.zeros(val_neg_labels_T1.shape[0])), axis = 0).astype(np.int64)
y_trainT2 = np.concatenate((np.ones(train_pos_labels_T2.shape[0]),np.zeros(train_neg_labels_T2.shape[0])), axis = 0).astype(np.int64)
y_valT2 = np.concatenate((np.ones(val_pos_labels_T2.shape[0]),np.zeros(val_neg_labels_T2.shape[0])), axis = 0).astype(np.int64)

# print
print('train data: ')
print(x_trainT1.shape, x_trainT1.dtype)
print(x_trainT2.shape, x_trainT2.dtype)
print(y_trainT1.shape, y_trainT1.dtype)
print(y_trainT2.shape, y_trainT2.dtype)

print('\nvalidation data: ')
print(x_valT1.shape, x_valT1.dtype)
print(x_valT2.shape, x_valT2.dtype)
print(y_valT1.shape, y_valT1.dtype)
print(y_valT2.shape, y_valT2.dtype)


