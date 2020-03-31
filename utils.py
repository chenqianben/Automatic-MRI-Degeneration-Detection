#!/usr/bin/env python
# coding: utf-8

from skimage.feature import hog
from skimage.exposure import adjust_gamma
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans 
from skimage.filters import gaussian, median, laplace
from skimage.draw import ellipse_perimeter

from math import ceil, floor
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
import pydicom

from read_roi import read_roi_file, read_roi_zip

label_size = (45,45)
modify_label_size = (42,42)
input_size = (36,36)
img_height, img_width = 128, 128

def read_data(root_path, if_normalized=True):
    '''read five types of images from the parent folder, if_normalized points to the T1 SAG images'''
    ims_T1s = []
    ims_T1 = []
    ims_T2s = []
    ims_T2 = []
    ims_T2st = []
    pos = []
    axis_ens_T1 = []
    axis_ens_T2 = []
    axis_ens_T2star = []
    
    for root, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if root.find('T1_TSE_SAG') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2) + 1
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T1s = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T1s = (im_T1s-im_T1s.mean())/im_T1s.std()
                    ims_T1s.append(im_T1s)
                    pos.append(os.path.abspath(os.path.dirname(root)))
                    break
                    
            if root.find('T1_Images') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2)
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T1 = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T1 = (im_T1-im_T1.mean())/im_T1.std()
                    ims_T1.append(im_T1)
                    break
                if f.endswith('.roi'):
                    ds = read_roi_file(f)
                    for i, key in enumerate(ds.keys()):
                        xs = ds[key]['x']
                        ys = ds[key]['y']
                        break
                    axis_T1 = np.array([xs,ys]).T
                    axis_ens_T1.append(axis_T1)
                    
            if root.find('T2_TSE_SAG') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2) + 1
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T2s = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T2s = (im_T2s-im_T2s.mean())/im_T2s.std()
                    ims_T2s.append(im_T2s)
                    break
                    
            if root.find('T2_Images') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2) + 1
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T2 = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T2 = (im_T2-im_T2.mean())/im_T2.std()
                    ims_T2.append(im_T2)
                    break
                if f.endswith('.roi'):
                    ds = read_roi_file(f)
                    for i, key in enumerate(ds.keys()):
                        xs = ds[key]['x']
                        ys = ds[key]['y']
                        break
                    axis_T2 = np.array([xs,ys]).T
                    axis_ens_T2.append(axis_T2)
                    
            if root.find('T2Star_Images') is not -1:
                f = os.path.join(root,filename)
                nb_chosen = int(len(filenames)/2) + 1
                if f.endswith(str(nb_chosen)+'.dcm'):
                    ds = pydicom.dcmread(f)            # dcm format 
                    im_T2st = ds.pixel_array            # array 这里train data是（512，512） dtype = int16
                    if if_normalized:
                        im_T2st = (im_T2st-im_T2st.mean())/im_T2st.std()
                    ims_T2st.append(im_T2st)
                    break
                if f.endswith('.roi'):
                    ds = read_roi_file(f)
                    for i, key in enumerate(ds.keys()):
                        xs = ds[key]['x']
                        ys = ds[key]['y']
                        break
                    axis_T2star = np.array([xs,ys]).T
                    axis_ens_T2star.append(axis_T2star)
                    
    return np.array(ims_T1s), np.array(ims_T1), np.array(ims_T2s), np.array(ims_T2), np.array(ims_T2st), pos, np.array(axis_ens_T1), np.array(axis_ens_T2), np.array(axis_ens_T2star)

def rois_dict_to_axis(rois_dict):
    axis_ens = []
    for i, key in enumerate(rois_dict.keys()):
        x = rois_dict[key]['left']
        y = rois_dict[key]['top']
        w = rois_dict[key]['width']
        h = rois_dict[key]['height']
        axis_ens.append([x,y,w,h])
    return axis_ens

def get_rois(im, rects, if_modify_size):
    im_rois = []
    for i,(x,y,w,h) in enumerate(rects):
        if if_modify_size:
            l = ceil((label_size[1]-modify_label_size[1])/2)
            r = floor((label_size[1]-modify_label_size[1])/2)
            u = floor((label_size[0]-modify_label_size[0])/2)
            d = ceil((label_size[0]-modify_label_size[0])/2)
            
            x = x + l
            y = y + u
            w = w - l - r
            h = h - u - d
        im_rois.append(im[y:y+h,x:x+w])
    return np.array(im_rois)

def reduce_size_from_indice(axis_ens, size):
    size_ori = axis_ens[0][2:4]
    axis_ens_new = []
    for i,(x,y,w,h) in enumerate(axis_ens):
        l = ceil((label_size[1]-modify_label_size[1])/2)
        r = floor((label_size[1]-modify_label_size[1])/2)
        u = floor((label_size[0]-modify_label_size[0])/2)
        d = ceil((label_size[0]-modify_label_size[0])/2)
        
        x = x + l
        y = y + u
        w = w - l - r
        h = h - u - d
        
        axis_ens_new.append([x,y,size[0],size[1]])
    return axis_ens_new

def read_rois_and_axis_data(dir_rois):
    rois_ens = []
    rois_axis_ens = []
    for root, dirnames, filenames in os.walk(dir_rois):       
        if not len(dirnames):
            rois = []
            rois_axis = []
            for filename in filenames:
                if filename.endswith(('.png', '.jpg', '.jpeg','.JPG', '.tif', 'tiff', '.gif')):
                    roi = io.imread(os.path.join(root, filename))
                    rois.append(roi)
                if filename.endswith(('.npy')):
                    roi_axis = np.load(os.path.join(root, filename))
                    rois_axis.append(roi_axis)
            rois_ens.append(np.array(rois))
            rois_axis_ens.append(np.array(rois_axis).squeeze())
    return rois_ens, rois_axis_ens


def find_rect(im, orientations=4, pixels_per_cell=(512,2), cells_per_block=(1,1)):
    # do hog to detect and separate the spine
    im_gau = gaussian(im, sigma = 5)
    fd, hog_im = hog(im_gau, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=True)
    hog_im = adjust_gamma(hog_im,gamma=0.5)
    
    precision = pixels_per_cell[1]
    p = [(2*i+1) for i in range(512//precision)]
    centres = np.array([c*0.5*precision for c in p]).astype(int)
    centres_max  = [np.max(hog_im[512//2,centres[i]]) for i in range(512//precision)]
    th = np.mean(centres_max)
    centres_use = [centres[i] for i in range(len(centres_max)) if centres_max[i]>th]
    
    # detect ananomy: there may be some noise in the left of the picture that is noted in the centres_use
    for centre in centres_use:
        if centre + precision not in centres_use and centre - precision not in centres_use:
            centres_use.remove(centre)
            
    # detect ananomy: there may be some noise just in the left of the spine that affected the mask
    while np.max(centres_use) - np.min(centres_use) > 128:
        centres_use.pop(0)
    
    # obtain ROI
    centre_r, centre_l = np.max(centres_use), np.min(centres_use)
    centre_m = int((centre_r + centre_l)/2)
    im_copy = np.zeros_like(im)
    
    # make some tolerance
    centre_l, centre_m = centre_l - int(0.3 * (centre_m - centre_l)), centre_m + int(0.3 * (centre_m - centre_l))
    im_copy[:,centre_l:centre_m] = im[:,centre_l:centre_m]

    return im_copy, centre_l, centre_m

def gmm(im, size, n):
    data = im.ravel()[:,np.newaxis]

    model = GaussianMixture(n_components = n)
    model.fit(data)
    label_pred = model.predict(data)
    label_pred = label_pred.reshape(size)
    return label_pred

def kmeans(im, size, n):
    data = im.ravel()[:,np.newaxis]

    model = KMeans(n_clusters = n)
    model.fit(data)
    label_pred = model.labels_
    label_pred = label_pred.reshape(size)
    return label_pred

def normalize(ims): # normalize to 0~1
    ims_out = []
    for im in ims:
        ims_out.append((im-im.min())/(im.max()-im.min()))
    return np.array(ims_out)

def plot_axis(axis, shape):
    x, y, w, h = axis
    if shape == 'ellipse':
        X = [x,x,x+w,x+w,x]
        Y = [y,y+h,y+h,y,y]
        c = np.array([X,Y]).T
        plt.plot(c[:, 0], c[:, 1], '--r', lw=2)
    elif shape == 'rect' or shape == 'rectangle':
        xc = x + round(w/2)
        yc = y + round(h/2)
        a = round(w/2)
        b = round(h/2)
        
        cy, cx = ellipse_perimeter(yc, xc, a, b)
        init = np.array([cx, cy]).T  
        plt.plot(init[:, 0], init[:, 1], '--r', lw=2)
    else:
        print('shape should be ellipse or rectangle')