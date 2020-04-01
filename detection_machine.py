from read_roi import read_roi_file, read_roi_zip
import skimage.io as io
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
import random
import os
from datetime import datetime

from sklearn.mixture import GaussianMixture
from sklearn import metrics # Calinski-Harabasz Index评估的聚类分数
# evaluation using sklearn
from sklearn.metrics import accuracy_score,recall_score, precision_score
from sklearn.metrics import confusion_matrix

from skimage.filters import gaussian, median, laplace
from skimage.morphology import disk, erosion
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse, resize
from skimage.feature import canny, hog
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.segmentation import active_contour
from skimage.transform import hough_line, hough_line_peaks
from skimage import io

from utils import label_size, modify_label_size, input_size
from utils import find_rect, gmm, kmeans, rois_dict_to_axis, get_rois, reduce_size_from_indice


class Detection():
    def __init__(self, model, ims, input_size):
        self.model = model
        self.ims = ims
        self.N = len(self.ims)
        self.input_size = input_size
        
    def operate(self, rate):
        print(80 * "=")
        print('OPERATING')
        print(80 * "=")
        start=datetime.now()
        self.rate = rate
        self.indices_pred_ens = []
        self.cluster_model_ens = []
        self.inds_anomal = [] 
        self.center_indices_ens = []
        
        for i, im in enumerate(self.ims):
            # reduce the image size to a rectangle
            precision = 2    
            im_rect, l, r = find_rect(im, orientations=4, pixels_per_cell=(512,precision),
                                    cells_per_block=(2,2))
            im_rect = im_rect[:,l:r]
            
            # input data to the model
            rois_im, indices = self.find_roi(im_rect, self.input_size, strides = (2,2))
            indices = np.array(indices)
            indices[:,1] += l
            indices = indices[:,:2]
            
            # predctions
            results = self.model(rois_im[:,:,:,np.newaxis])
            #preds = np.argmax(results, axis=1)
            
            # get positive indices
            pos_indices, neg_indices = self.predction_from_results(results, indices, self.rate)
            if len(pos_indices) is not 0:
                pos_indices_point = (np.array(pos_indices))[:,:2].reshape(-1,2)       
            
                # chose the best n by using calinski_harabasz_score and silhouette_score
                indices_pred, cluster_model = self.clustering(pos_indices_point)
                self.indices_pred_ens.append(indices_pred)
                self.cluster_model_ens.append(cluster_model)
            else:
                print('Encounter a problem detecting image of indice',i)
                self.inds_anomal.append(i)
                self.indices_pred_ens.append(None)
                self.cluster_model_ens.append(None)
                
            # get center clusters
            center_indices = np.around(self.cluster_model_ens[i].means_).astype(np.int64)
            # 去掉最下面那一个roi
            ind = np.argmax(center_indices,0)[0]
            center_indices = np.delete(center_indices, obj = ind, axis = 0)
            self.center_indices_ens.append(center_indices)

        print('time consuming: ', datetime.now()-start)
        
    def draw_figure(self, inds):
        print(80 * "=")
        print('DRAWING FIGURE')
        print(80 * "=")
        inds = list(inds)
        plt.figure(figsize = (10,10*len(inds)))
        colormap = plt.cm.gist_ncar   
        colors = [colormap(i) for i in np.linspace(0, 1, 4)] 
        for i, ind in enumerate(inds):
            if ind not in self.inds_anomal:
                im = self.ims[ind]
                indices_pred = self.indices_pred_ens[ind]
                cluster_model = self.cluster_model_ens[ind]
                center_indices = np.around(cluster_model.means_).astype(np.int64)
                gt = self.ground_truth(im, self.pos[i], center_indices)
                
                plt.subplot(len(inds),1,i+1)
                plt.imshow(im,cmap='gray')
                plt.axis('off')
                plt.title(ind)
                [self.plot_roi_from_indice(im, self.input_size, center_indices[i], colors[2]) for i in range(len(center_indices)) if gt[i]==1] 
                [self.plot_roi_from_indice(im, self.input_size, center_indices[i], colors[1]) for i in range(len(center_indices)) if gt[i]==0]
            else:
                print('Image of indice', ind ,'took out')
        plt.show()
        
    def evaluation(self):
        print(80 * "=")
        print('EVALUATION')
        print(80 * "=")
        pos = self.pos
        TPs = []
        FPs = []
        for i,im in enumerate(self.ims):
            # 找到ground truth
            gt = self.ground_truth(im, self.pos[i], self.center_indices_ens[i])
            # 考虑到evaluation 的 indice全部都预测为正，因此只需要计算TP和FP
            TPs.append(np.sum(gt))
            FPs.append(gt.shape[0] - np.sum(gt))
        
        # 查准率, 指的是在所有预测正的里面正确的比例
        print('Model evaluation: TP: ', np.array(TPs).sum().astype(np.int), ' | FP: ', np.array(FPs).sum().astype(np.int), 
            ' | precision score: ',np.array(TPs).sum()/(np.array(FPs).sum()+np.array(TPs).sum()))
    
    def read_and_save_rois(self, dir_rois=False):
        print(80 * "=")
        print('READING AND SAVING ROI')
        print(80 * "=")
        if dir_rois and not os.path.exists(dir_rois):
            os.makedirs(dir_rois)
        rois_ens = []
        for i, center_indices in enumerate(self.center_indices_ens):
            if dir_rois and not os.path.exists(os.path.join(dir_rois,str(i))):
                os.makedirs(os.path.join(dir_rois,str(i)))
            rois = []
            for j, (y, x) in enumerate(center_indices):
                x = int(x)
                y = int(y)
                roi = self.ims[i][y:y+self.input_size[0], x:x+self.input_size[1]]
                rois.append(roi)
                if dir_rois:
                    io.imsave(os.path.join(os.path.join(dir_rois, str(i)), str(j)+'.tiff'), roi.astype(np.float32))# from float64 to float32 saved as tif
            rois_ens.append(rois)
        if dir_rois:
            print('Rois figures saved to {} successfully'.format(dir_rois))
        return rois_ens
               
    def read_and_save_rois_axis(self, dir_rois=False):
        print(80 * "=")
        print('READING AND SAVING ROI AXIS')
        print(80 * "=")
        if dir_rois and not os.path.exists(dir_rois):
            os.makedirs(dir_rois)
        for i, center_indices in enumerate(self.center_indices_ens):
            if dir_rois and not os.path.exists(os.path.join(dir_rois,str(i))):
                os.makedirs(os.path.join(dir_rois,str(i)))
            np.save(os.path.join(os.path.join(dir_rois, str(i)),'rois_axis.npy'), np.array(self.center_indices_ens[i]))
        if dir_rois:
            print('Rois axis npy file saved to {} successfully'.format(dir_rois))
        return self.center_indices_ens
    
    def ground_truth(self, im, pos, indices, tol = 5):
        dir_labels = r'D:\课件\ECN第二年DATASIM\Projet\projet mapping segmentation\positive ROI'
        dir_labels = os.path.join(dir_labels,'T'+str(pos[0])+' SAG')
        dir_labels = os.path.join(dir_labels,str(pos[1])+'_'+str(pos[2]))
        rois_dict = read_roi_zip(dir_labels + '.zip')
        axis_ens = np.array(rois_dict_to_axis(rois_dict))
        #axis_ens = reduce_size_from_indice(axis_ens, modify_label_size)

        gt_true = np.zeros(indices.shape[0],)  # 第一列是gt,第二列是preds
        gt = []
        for (x,y,_,_) in axis_ens:
            x += ceil((label_size[1]-modify_label_size[1])/2)
            y += floor((label_size[0]-modify_label_size[0])/2)
            gt.append([y,x])
        gt = np.array(gt)

        tol += modify_label_size[0] - input_size[0]
        for i,(test_y,test_x) in enumerate(indices):
            for (gt_y,gt_x) in gt:
                if (test_y-gt_y)< tol and (test_x-gt_x)< tol:
                    gt_true[i] = 1
                    continue

        return gt_true

    def plot_roi_from_indice(self, ims, input_size, indice, if_label = False):
        """Plot roi from one indice"""
        h, w = input_size
        x, y = indice[1], indice[0] 
        X = [x,x,x+w,x+w,x]
        Y = [y,y+h,y+h,y,y]
        c = np.array([X,Y]).T
        if if_label == False:
            plt.plot(c[:, 0], c[:, 1], '-b', lw=2)
        else:
            plt.plot(c[:, 0], c[:, 1], '-b', lw=2, color = if_label) 
        
    def from_im_to_indices(self,im, input_size, strides):
        '''
        change from an image to patches indices as the input of CNN network
        input: 
            input_size: input_size[0]*input_size[1] of input of CNN network 
            strides: stride[0]*stride[1] of the moving steps 
        output:
            patches indices of an image of form: (left top width height)
        '''
        indices = []

        h, w = im.shape
        for i in range(0, h-input_size[0]+1, strides[0]):
            for j in range(0, w-input_size[1]+1, strides[1]):
                indices.append([i, j, input_size[0], input_size[1]])
        return indices
    
    def from_indices_to_patches(self, im, indices):
        patches = []
        for (i, j, h, w) in indices:
            patches.append(im[i:i+h, j:j+w])
        return patches
    
    def find_roi(self, im, input_size, strides):
        '''find the rois of an image by using CNN model'''
        indices = self.from_im_to_indices(im, input_size, strides)
        patches = self.from_indices_to_patches(im, indices)
        rois_im = np.array(patches).astype(np.float32)
        return rois_im, indices
            
    def predction_from_results(self, results, indices, rate):
        pos_indices = []
        neg_indices = []
        for ind, (y1,y2) in enumerate(results): 
            if y2 > rate:
                pos_indices.append(indices[ind])
            else:
                neg_indices.append(indices[ind])
        return pos_indices, neg_indices
    
    def clustering(self, pos_indices_point):
        '''chose the best n by using calinski_harabasz_score and silhouette_score'''        
        scores = np.zeros((5,2))
        for k in range(6,11):
            cluster_model = GaussianMixture(n_components = k)
            indices_pred = cluster_model.fit_predict(pos_indices_point)
            #print(pos_indices_point)
            #print(indices_pred)
            scores[k-6,0] = metrics.calinski_harabasz_score(pos_indices_point, indices_pred)
            scores[k-6,1] = metrics.silhouette_score(pos_indices_point, indices_pred)
        scores[:,0] = (scores[:,0] - scores[:,0].mean())/(scores[:,0].std())
        scores[:,1] = (scores[:,1] - scores[:,1].mean())/(scores[:,1].std())
        scores = np.sum(scores, axis=1)
        k_chosen = scores.argmax() + 6
        
        cluster_model = GaussianMixture(n_components = k_chosen)
        indices_pred = cluster_model.fit_predict(pos_indices_point)
        #score = metrics.calinski_harabasz_score(pos_indices_point, indices_pred)
        return indices_pred, cluster_model
    
    def extract_rois_from_image(self, im, input_size, indices):
        rois = []
        for x,y in indices:
            rois.append(im[x:x+input_size[1],y:y+input_size[0]])
        return rois

    def from_normalization_to_origin(self, rois, conf):
        return (np.array(rois) * conf[1] + conf[0]).astype(np.int16)
