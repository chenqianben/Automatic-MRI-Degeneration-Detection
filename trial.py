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

# 1.spatial filtering
if __name__ == '__main__':
    nb = np.random.randint(trainT1_data.shape[0])
    im = trainT1_data[nb]
    im_gaussian = gaussian(im,sigma=3)                    # sigma越大越平滑
    im_median = median(im, disk(3))

    plt.figure(figsize=(30,10))
    plt.subplot(131)
    plt.imshow(im,cmap='gray')
    plt.subplot(132)
    plt.imshow(im_gaussian,cmap='gray')
    plt.subplot(133)
    plt.imshow(im_median,cmap='gray')

# 2.edges and contours
if __name__ == '__main__':
    nb = np.random.randint(trainT1_data.shape[0])
    im = trainT1_data[nb]
    edges = canny(im, sigma=15)                             # edge image detected by using canny method ,sigma 越大，则edges越少
    #contours = find_contours(edges,0.8)

    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.imshow(im,cmap='gray')
    plt.subplot(222)
    plt.imshow(edges,cmap='gray')
    #plt.subplot(223)
    #plt.imshow(contours)
    plt.show()

# 3.clustering
if __name__ == '__main__':
    #gmm_label_pred1 = gmm(im_gaussian,2)
    gmm_label_pred2 = gmm(im,2)
    #kmeans_label_pred = kmeans(im,3)

    plt.figure(figsize=(30,10))
    plt.subplot(131)
    plt.imshow(im,cmap='gray')
    plt.subplot(132)
    #plt.imshow(gmm_label_pred1,cmap='gray')
    plt.subplot(133)
    plt.imshow(gmm_label_pred2,cmap='gray')
    plt.show()

# 4. Hough Line
# 发现问题： 1.有一些比较暗的图片，直线找不出来，还有一些脊柱有一点弯曲，直线也找不出来 2.方法不稳定，每次找同一张图片的结果不一定一样
if __name__ == '__main__':
    nb = np.random.randint(trainT1_data.shape[0])
    im = trainT1_data[nb]
    med = median(im,disk(5))
    im_gmm_med = gmm(med,n=2)
    if np.mean(im_gmm_med) > 0.5:
        im_gmm_med = np.where(im_gmm_med==0,1,0)
    im_roi = im_gmm_med*im
    edg = canny(im_roi, sigma=10)

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 60)
    h, theta, d = hough_line(edg, theta=tested_angles)

    plt.figure(figsize=(5,5))
    plt.imshow(im,cmap='gray')
    origin = np.array((0, edg.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        if abs(angle) < np.pi/18 and 100 < abs(dist) < 500:
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            plt.plot(origin, (y0, y1), '-r')
    plt.xlim((0, im.shape[1]))
    plt.ylim((im.shape[0], 0))
    plt.title('No.' + str(nb))
    plt.show()

# 5. hog 很棒的操作
if __name__ == '__main__':
    nb = np.random.randint(trainT1_data.shape[0])
    im = trainT1_data[nb]
    med = median(im,disk(5))
    im_gmm_med = gmm(med,n=2)
    if np.mean(im_gmm_med) > 0.5:
        im_gmm_med = np.where(im_gmm_med==0,1,0)
    im_roi = im_gmm_med*im
    fd, hog_im = hog(im_roi, orientations=4, pixels_per_cell=(512, 64),
                        cells_per_block=(1, 1), visualize=True)
    #hog_im = adjust_gamma(hog_im,gamma=0.5)

    #print(fd.shape)
    #print(hog_image.shape)

    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(im,cmap='gray')
    plt.subplot(132)
    plt.imshow(hog_im,cmap='gray')
    plt.subplot(133)
    plt.imshow(im,cmap='gray')
    plt.imshow(hog_im,cmap='gray')
    plt.show()

# 6. segmentation of ROI
#from scipy import ndimage as ndi

# plot some train data and preprocess results
if __name__ == '__main__':
	data_examples = 5
	type_examples = 5
	tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 60)
	plt.figure(figsize=(8 * type_examples, 8 * data_examples))

	counter = 1
	for i in range(data_examples):
		nb = np.random.randint(trainT1_data.shape[0])
		im = trainT1_data[nb]

		# prepocess
		# 1
		# smo = gaussian(im,sigma=5)
		med = median(im, disk(5))
		# lap = laplace(med,ksize=3)

		# 2
		im_gmm_med = gmm(med, n=2)
		if np.mean(im_gmm_med) > 0.5:
			im_gmm_med = np.where(im_gmm_med == 0, 1, 0)

		# 3
		im_roi = im_gmm_med * im

		# 4
		edg = canny(im_roi, sigma=10)
		h, theta, d = hough_line(edg, theta=tested_angles)

		plt.subplot(data_examples, type_examples, counter)
		plt.imshow(im, cmap='gray')
		for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
			if abs(angle) < np.pi / 18 and 100 < abs(dist) < 500:
				y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
				plt.plot(origin, (y0, y1), '-r')
		plt.xlim((0, im.shape[1]))
		plt.ylim((im.shape[0], 0))
		plt.title('No.' + str(nb), fontsize=20)
		plt.axis('off')
		counter += 1

		plt.subplot(data_examples, type_examples, counter)
		plt.imshow(med, cmap='gray')
		plt.axis('off')
		counter += 1

		plt.subplot(data_examples, type_examples, counter)
		plt.imshow(im_gmm_med, cmap='gray')
		plt.axis('off')
		counter += 1

		plt.subplot(data_examples, type_examples, counter)
		plt.imshow(im_roi, cmap='gray')
		plt.axis('off')
		counter += 1

		plt.subplot(data_examples, type_examples, counter)
		plt.imshow(edg, cmap='gray')
		plt.axis('off')
		counter += 1

	plt.show()