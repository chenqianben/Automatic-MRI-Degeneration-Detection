#!/usr/bin/env python
# coding: utf-8
from skimage.filters import gaussian, median, laplace
from skimage.morphology import disk
from skimage.feature import canny, hog
#from skimage.measure import find_contours
from skimage.exposure import adjust_gamma
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage.transform import hough_line, hough_line_peaks
from data import trainT1_data, trainT2_data

import numpy as np
import matplotlib.pyplot as plt

print(trainT1_data.shape)
print(trainT2_data.shape)

def kmeans(im,n):
    data = im.ravel()[:,np.newaxis]

    model = KMeans(n_clusters = n)
    model.fit(data)
    label_pred = model.labels_
    label_pred = label_pred.reshape(512,512)
    return label_pred
def gmm(im,n):
    data = im.ravel()[:,np.newaxis]

    model = GaussianMixture(n_components = n)
    model.fit(data)
    label_pred = model.predict(data)
    label_pred = label_pred.reshape(512,512)
    return label_pred

# data pre-processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

nb = np.random.randint(trainT1_data.shape[0])
im = trainT1_data[nb]

def find_roi1(im, precision):
    # preprocess 1: do the binary separation to get the mask
    med = median(im,disk(5))
    im_gmm_med = gmm(med,n=2)
    if np.mean(im_gmm_med) > 0.5:
        im_gmm_med = np.where(im_gmm_med==0,1,0)
    mask = im_gmm_med*im
    
    # preprocess 2: do hog to detect and separate the spine
    fd, hog_im = hog(mask, orientations=4, pixels_per_cell=(512, precision),
                        cells_per_block=(2, 2), visualize=True)
    hog_im = adjust_gamma(hog_im,gamma=0.5)
    
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
    centre_l, centre_m = centre_l - int(0.1 * (centre_m - centre_l)), centre_m + int(0.1 * (centre_m - centre_l))
    im_copy[:,centre_l:centre_m] = mask[:,centre_l:centre_m]

    return im_copy
precision = 8    # nb of pixcels per cell horizontally, the lower the more precise
im_roi1 = find_roi1(im, precision)

def find_roi2(im, precision):
    fd, hog_im = hog(im, orientations=8, pixels_per_cell=(precision, precision),
                        cells_per_block=(4, 4), visualize=True)
    return hog_im
precision = 2    # nb of pixcels per cell horizontally, the lower the more precise
im_roi2 = find_roi2(im_roi1, precision)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(im_roi1,cmap='gray')
plt.subplot(122)
plt.imshow(im_roi2,cmap='gray')
plt.show()

# TODO 试验记录
# TODO 从去噪声 模糊化效果来看，median保留了更多的特征，比gaussian要好

# plot some train data and preprocess results
if __name__ == '__main__':
    data_examples = 5
    type_examples = 5
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 60)
    plt.figure(figsize=(8*type_examples,8*data_examples))

    counter = 1
    for i in range(data_examples):
        nb = np.random.randint(trainT1_data.shape[0])
        im = trainT1_data[nb]

        # prepocess
        # 1
        #smo = gaussian(im,sigma=5)
        med = median(im,disk(5))
        #lap = laplace(med,ksize=3)
        
        # 2
        im_gmm_med = gmm(med,n=2)
        if np.mean(im_gmm_med) > 0.5:
            im_gmm_med = np.where(im_gmm_med==0,1,0)
        
        # 3
        im_roi = im_gmm_med*im
        
        # 4
        edg = canny(im_roi, sigma=10)
        h, theta, d = hough_line(edg, theta=tested_angles)
        
        plt.subplot(data_examples,type_examples,counter)
        plt.imshow(im,cmap='gray')
        origin = np.array((0, edg.shape[1]))
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            if abs(angle) < np.pi/18 and 100 < abs(dist) < 500:
                y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                plt.plot(origin, (y0, y1), '-r')
        plt.xlim((0, im.shape[1]))
        plt.ylim((im.shape[0], 0))
        plt.title('No.' + str(nb), fontsize=20)
        plt.axis('off')
        counter += 1

        plt.subplot(data_examples,type_examples,counter)
        plt.imshow(med,cmap='gray')
        plt.axis('off')
        counter += 1

        plt.subplot(data_examples,type_examples,counter)
        plt.imshow(im_gmm_med,cmap='gray')
        plt.axis('off')
        counter += 1
        
        plt.subplot(data_examples,type_examples,counter)
        plt.imshow(im_roi,cmap='gray')
        plt.axis('off')
        counter += 1
        
        plt.subplot(data_examples,type_examples,counter)
        plt.imshow(edg,cmap='gray')
        plt.axis('off')
        counter += 1

    plt.show()







