#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import shutil   # copy file

from skimage.transform import resize
import skimage.io as io
import pydicom  # process dcm files

import numpy as np
import matplotlib.pyplot as plt

from preprocess import *


# In[2]:


entry = 'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_brut'
exit = 'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation'

# create the IRM_process dir

def create_dirs(entry,exit):
    path1 = os.path.join(exit,'IRM_process')
    if not os.path.exists(path1):
        os.mkdir(path1)
    for nb_sheep,root_sheep in enumerate(os.listdir(entry)):
        #print(nb_sheep,root_sheep)

        # create the sheep dir in the IRM_process dir
        sheep_name = 'Brebis' + str(nb_sheep+1)
        path2 = os.path.join(path1,sheep_name)
        if not os.path.exists(path2):
            os.mkdir(path2)

        for nb_temps,root_temps in enumerate(os.listdir(os.path.join(entry,root_sheep))):
            #print(nb_temps,root_temps)

            # create the time dir in the sheep_dir
            time_name = str(nb_temps+1)
            path3 = os.path.join(path2,time_name)
            if not os.path.exists(path3):
                os.mkdir(path3)

            path = os.path.join(entry,os.path.join(root_sheep,root_temps))
            
            for root, dirnames, filenames in os.walk(path):
                # 1. consider the T1 SAG images
                if root.find('T1_TSE_SAG') is not -1:
                    # create the T1_TSE_SAG dir in the time dir
                    path4 = os.path.join(path3,'T1_TSE_SAG')
                    if not os.path.exists(path4):
                        os.mkdir(path4)
                    # copy some T1 images into the destination dir 
                    length = len(filenames)       
                    nb_chosen = int(length/2)
                    for f in filenames:
                        if f.endswith(str(nb_chosen)+'.dcm'):
                            shutil.copy2(os.path.join(root,f), path4)
                            break               # 读取一张就要break，有些文件夹可能有很多张，只有一张有用，
                                                # 其他张都是大小或者位置或者拍摄有问题的
 
                # 2. consider the T2 SAG images
                if root.find('T2_TSE_SAG') is not -1:
                    # create the T2_TSE_SAG dir in the time dir
                    path4 = os.path.join(path3,'T2_TSE_SAG')
                    if not os.path.exists(path4):
                        os.mkdir(path4)
                    # copy some T1 images into the destination dir 
                    length = len(filenames)       
                    nb_chosen = int(length/2)
                    for f in filenames:
                        if f.endswith(str(nb_chosen)+'.dcm'):
                            shutil.copy2(os.path.join(root,f), path4)
                            break
                
                # 3. consider the T1 images
                if root.find('T1_Images') is not -1:
                    # create the T1_images dir in the time dir
                    path4 = os.path.join(path3,'T1_images')
                    if not os.path.exists(path4):
                        os.mkdir(path4)
                    # copy some T1 images into the destination dir 
                    length = len(filenames)           
                    nb_chosen = int(length/2)
                    for f in filenames:
                        if f.endswith(str(nb_chosen)+'.dcm'):
                            shutil.copy2(os.path.join(root,f), path4)
                            break
                                
                # 4. consider the T2 images
                if root.find('T2_Images') is not -1:
                    # create the T2_images dir in the time dir
                    path4 = os.path.join(path3,'T2_images')
                    if not os.path.exists(path4):
                        os.mkdir(path4)
                    # copy some T2 images into the destination dir 
                    length = len(filenames)       
                    nb_chosen = int(length/2)
                    for f in filenames:
                        if f.endswith(str(nb_chosen)+'.dcm'):
                            shutil.copy2(os.path.join(root,f), path4)
                            break
                                
                # 5. consider the T2 star images
                if root.find('T2Star_Images') is not -1:
                    # create the T2_images dir in the time dir
                    path4 = os.path.join(path3,'T2Star_Images')
                    if not os.path.exists(path4):
                        os.mkdir(path4)
                    # copy some T2 images into the destination dir 
                    length = len(filenames)       
                    nb_chosen = int(length/2)
                    for f in filenames:
                        if f.endswith(str(nb_chosen)+'.dcm'):
                            shutil.copy2(os.path.join(root,f), path4)
                            break
                        
create_dirs(entry,exit)


# In[3]:


# read training data of dcm files
def read_trainT1_data(if_normalized=True):
    DIR = 'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process'
    train_data = []
    for root, dirnames, filenames in os.walk(DIR):
        if root.find('T1_TSE_SAG') is not -1:
            for filename in filenames:
                f = os.path.join(root, filename)
                ds = pydicom.dcmread(f)      # dcm format         
                im = ds.pixel_array          # array 这里train data是（512，512） dtype = int16
                if if_normalized:
                    im = (im-im.mean())/im.std()
                train_data.append(im)
    train_data = np.array(train_data)
    return train_data

def read_trainT2_data(if_normalized=True):
    DIR = 'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process'
    train_data = []
    for root, dirnames, filenames in os.walk(DIR):
        if root.find('T2_TSE_SAG') is not -1:
            for filename in filenames:
                f = os.path.join(root, filename)
                ds = pydicom.dcmread(f)      # dcm format         
                im = ds.pixel_array          # array 这里train data是（512，512） dtype = int16
                if if_normalized:
                    im = (im-im.mean())/im.std()
                train_data.append(im)
    train_data = np.array(train_data)
    return train_data

trainT1_data = read_trainT1_data()
trainT2_data = read_trainT2_data()
print(trainT1_data.shape)                  # (73,512,512)   dtype = float64，15只*5 = 75张，其中两只羊(13,14)只有四天
print(trainT2_data.shape)
#train_data = train_data.transpose(1,2,0)


# In[4]:


# find differences between train data and test data
train_file_T1 = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process\Brebis2\1\T1_TSE_SAG\IM-0002-0003.dcm'
train_file_T2 = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process\Brebis2\1\T2_TSE_SAG\IM-0001-0003.dcm'
test_file_T1 = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process\Brebis2\1\T1_images\IM-0008-0012.dcm'
test_file_T2 = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process\Brebis2\1\T2_images\IM-0005-0005.dcm'
test_star_file_T2 = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\IRM_process\Brebis2\1\T2Star_Images\IM-0010-0005.dcm'

train_ex_T1 = io.imread(train_file_T1)
train_ex_T2 = io.imread(train_file_T2)
test_ex_T1 = io.imread(test_file_T1)
test_ex_T2 = io.imread(test_file_T2)
test_star_ex_T2 = io.imread(test_star_file_T2)

print(train_ex_T1.shape)
print(train_ex_T2.shape)
print(test_ex_T1.shape)
print(test_ex_T2.shape)
print(test_star_ex_T2.shape)

plt.figure(figsize=(20,20))
plt.subplot(221)    
plt.imshow(train_ex_T1,cmap='gray')     
plt.title('T1_TSE_SAG')
plt.subplot(222)
plt.imshow(test_ex_T1,cmap='gray')     
plt.title('T1_images')
plt.subplot(223)    
plt.imshow(train_ex_T2,cmap='gray')     
plt.title('T2_TSE_SAG')
plt.subplot(224)
plt.imshow(test_ex_T2,cmap='gray')    
plt.title('T2_Images')
plt.show()


# In[5]:


# plot some train data and preprocess results
show_examples = 10
plt.figure(figsize=(24,80))

counter = 1
for i in range(show_examples):
    im = trainT1_data[np.random.randint(trainT1_data.shape[0])]
    
    # prepocess
    edges = canny(im, sigma=10)
    im_gmm = gmm(im,n=3) 
    
    plt.subplot(show_examples,3,counter)
    plt.imshow(im,cmap='gray')
    plt.axis('off')
    counter += 1
    
    plt.subplot(show_examples,3,counter)
    plt.imshow(edges,cmap='gray')
    plt.axis('off')
    counter += 1
    
    plt.subplot(show_examples,3,counter)
    plt.imshow(im_gmm,cmap='gray')
    plt.axis('off')
    counter += 1
    
plt.show()


# In[ ]:


# 1.spatial filtering
im_gaussian = gaussian(im,sigma=3)                    # sigma越大越平滑
im_median = median(im, disk(3))

plt.figure(figsize=(30,10))
plt.subplot(131)
plt.imshow(im,cmap='gray')
plt.subplot(132)
plt.imshow(im_gaussian,cmap='gray')
plt.subplot(133)
plt.imshow(im_median,cmap='gray')


# In[ ]:


# 2.edges and contours
edges = canny(im, sigma=15)                             # edge image detected by using canny method ,sigma 越大，则edges越少
#contours = find_contours(edges,0.8)

plt.figure(figsize=(20,20))
plt.subplot(221)
plt.imshow(im,cmap='gray')
plt.subplot(222)
plt.imshow(edges,cmap='gray')
#plt.subplot(223)
#plt.imshow(contours)
plt.show()


# In[ ]:


# 3.clustering
gmm_label_pred1 = gmm(im_gaussian,3)
gmm_label_pred2 = gmm(im,3)
#kmeans_label_pred = kmeans(im,3)

plt.figure(figsize=(30,10))
plt.subplot(131)
plt.imshow(im,cmap='gray')
plt.subplot(132)
plt.imshow(gmm_label_pred1,cmap='gray')
plt.subplot(133)
plt.imshow(gmm_label_pred2,cmap='gray')
plt.show()


# In[ ]:




