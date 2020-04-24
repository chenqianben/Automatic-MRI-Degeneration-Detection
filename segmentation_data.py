import os
import shutil   # copy file

import skimage.io as io
from skimage.color import gray2rgb
from skimage.transform import resize,rescale
from read_roi import read_roi_file, read_roi_zip

import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor 

from utils import input_size, img_height, img_width
from utils import normalize, read_data



def rois_dict_to_axis(rois_dict):
    axis_ens_centre = []
    axis_ens1 = []
    axis_ens2 = []
    for i, key in enumerate(rois_dict.keys()):
        if i==0: # middle
            x_centre = rois_dict[key]['left']
            y_centre = rois_dict[key]['top']
            w_centre = rois_dict[key]['width']
            h_centre = rois_dict[key]['height']
            axis_ens_centre.append([y_centre,x_centre,w_centre,h_centre])
        if i==1: # left or right
            x1 = rois_dict[key]['left']
            y1 = rois_dict[key]['top']
            w1 = rois_dict[key]['width']
            h1 = rois_dict[key]['height']
            axis_ens1.append([y1,x1,w1,h1])
        if i==2: # left or right
            x2 = rois_dict[key]['left']
            y2 = rois_dict[key]['top']
            w2 = rois_dict[key]['width']
            h2 = rois_dict[key]['height']
            axis_ens2.append([y2,x2,w2,h2])
    return axis_ens_centre,axis_ens1,axis_ens2


# In[15]:


def make_labels(axis_ens, img_height, img_width, tol=0):
    im = np.zeros((img_height, img_width))
    b = axis_ens[0][3]//2                        # 上下半径
    a = axis_ens[0][2]//2                        # 左右半径
    y = axis_ens[0][0] + b                       # 纵坐标 
    x = axis_ens[0][1] + a                       # 横坐标

    # rescale
    rescale_size = img_height/input_size[0]
    b = round(b*rescale_size)
    a = round(a*rescale_size)
    y = round(y*rescale_size)
    x = round(x*rescale_size)
    
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if (y-i)**2/b**2+(x-j)**2/a**2 < 1 + tol:
                im[i,j]=1
    return im


# # X_train and X_test

# In[16]:


dir_labels = r'.\segmentation_data'
dir_data = r'.\ROI detected'

def read_data(dir_labels, dir_data):
    X_train = []
    confs = []
    nums = []
    for root_l, dirnames_l, filenames_l in os.walk(dir_labels):
        for filename_l in filenames_l:
            num = filename_l[0:-4]
            for root_d, dirnames_d, filenames_d in os.walk(dir_data):
                for filename_d in filenames_d:
                    if num == filename_d[0:-5]:
                        #print(os.path.join(dir_data,filename_d))
                        im = io.imread(os.path.join(dir_data, filename_d))
                        confs.append([im.mean(), im.std()])
                        X_train.append(im)
                        nums.append([num])
    X_train = np.array(X_train)
    return X_train, confs, nums

X_train, confs_seg, nums = read_data(dir_labels, dir_data)

# rescale to <rescale_size> times bigger， rescale的时候要暂时将channels维度放在最后一维上
X_train = resize(X_train.transpose(1,2,0), (img_height, img_width)).transpose(2,0,1)  
X_train = normalize(X_train)        # 先normalization变成0到1


X_train_1 = X_train.copy()
X_train_2 = X_train.copy()


dir_labels = r'./segmentation_data'
def read_labels(dir_labels, type_labels, img_height, img_width, tol=0):
    Y_train_1 = []
    Y_train_2 = []
    nums = []
    for root, dirnames, filenames in os.walk(dir_labels):
        for filename in filenames:
            num = filename[0:-4]
            rois = read_roi_zip(os.path.join(dir_labels,filename))
            axis_ens_centre, axis_ens1, axis_ens2 = rois_dict_to_axis(rois)
            if type_labels == 1:
                im = make_labels(axis_ens_centre, img_height, img_width, tol)
                Y_train_1.append(im)
            
            if type_labels == 2:
                im1 = make_labels(axis_ens1, img_height, img_width, tol)
                im2 = make_labels(axis_ens2, img_height, img_width, tol)
                im = im1+im2
                Y_train_2.append(im)
            nums.append(num)
    Y_train_1 = np.array(Y_train_1)
    Y_train_2 = np.array(Y_train_2)
    return Y_train_1, Y_train_2, nums

tol = 0.1
Y_train_1, _, nums = read_labels(dir_labels, 1, img_height, img_width, tol)            
_, Y_train_2, nums = read_labels(dir_labels, 2, img_height, img_width, tol)            



from skimage.color import gray2rgb
X_show = gray2rgb(X_train_1).astype(np.float)
X_show[:,:,:,0] = X_show[:,:,:,0] + 255*Y_train_1       # centre is red
X_show[:,:,:,1] = X_show[:,:,:,1] + 255*Y_train_2       # sides are green
X_show = (X_show-X_show.min()) / (X_show.min()+X_show.max())
X_show = X_show/X_show.max()


dir_labels = r'.\segmentation_data'
dir_data = r'.\ROI detected'
def read_data_test(dir_labels,dir_data):
    
    num = []
    X_test = []
    
    for root_l, dirnames_l, filenames_l in os.walk(dir_labels):
        for filename_l in filenames_l:
            num.append(filename_l[0:-4])
            
    for root_d, dirnames_d, filenames_d in os.walk(dir_data):
        for filename_d in filenames_d:
            if filename_d[0:-5] not in num:
#                 print(filename_d[0:-5])
                Im = io.imread(os.path.join(dir_data,filename_d))
                im = (Im-Im.min())/(Im.max()-Im.min()) # normalized
                X_test.append(im)
    X_test = np.array(X_test)
    return X_test

X_test_unlabelled = read_data_test(dir_labels,dir_data)      
X_test_unlabelled = resize(X_test_unlabelled.transpose(1,2,0), (img_height, img_width)).transpose(2,0,1)   
X_test_unlabelled = normalize(X_test_unlabelled)        # 先normalization变成0到1

# 增加维度
X_train_1 = X_train_1[:,:,:,np.newaxis].astype(np.float32)
Y_train_1 = Y_train_1[:,:,:,np.newaxis].astype(np.float32)

X_train_2 = X_train_2[:,:,:,np.newaxis].astype(np.float32)
Y_train_2 = Y_train_2[:,:,:,np.newaxis].astype(np.float32)

X_test_unlabelled = X_test_unlabelled[:,:,:,np.newaxis].astype(np.float32)
