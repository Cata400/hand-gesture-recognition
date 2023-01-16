import cv2
import optuna
import os
import random
import skimage

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler


def read_image_cv2(path):
    """
    Read image using OpenCV then convert it to RGB format.

    Parameters
    ----------
    path : string
        Path to the image.

    Returns
    -------
    img : 3D array-like
        RGB Image in HWC format.
    """
    img = cv2.imread(path)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def show_image(img, title="", min=0, max=255, gray=False):
    """
    Plot image.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be plotted, in HW or HWC format.
    title : string
        Title of the plot.
    min : int, optional
        Minimum value for the plot, by default 0.
    max : int, optional
        Maximum value for the plot, by default 255.
    gray : bool, optional
        True if the image is grayscale, False otherwise, by default False.
    """
    plt. figure()
    if gray:
        plt.imshow(img, cmap='gray', vmin=min, vmax=max)
        plt.colorbar()
    else:
        plt.imshow(img, vmin=min, vmax=max)

    plt.title(title)
    plt.show()
    
    
def get_image_paths(dataset_path, no_classes):
    """
    Get the paths of all images in the dataset.

    Parameters
    ----------
    dataset_path : string
        Path to the dataset.
    no_classes : int
        Number of classes in the dataset.

    Returns
    -------
    file_list : list
        List of lists containing the paths of all images in the dataset.
    """
    file_list = []
    for i in range(no_classes):
        files = []
        for file in os.listdir(os.path.join(dataset_path, str(i))):
            files.append(os.path.relpath(os.path.join(dataset_path, str(i), file)))
        file_list.append(files) 
    
    return file_list


def train_test_split(file_list, seed, train_size=0.7):
    """
    Split the dataset into train and test.

    Parameters
    ----------
    file_list : list
        List of lists containing the paths of all images in the dataset.
    seed : int
        Seed for the random generator.
    train_size : float, optional
        Percentage of the dataset to be used for training, by default 0.7.

    Returns
    -------
    train_files : list
        List of lists containing the paths of all images in the train set.
    test_files : list
        List of lists containing the paths of all images in the test set.
    """
    no_classes = len(file_list)
    no_images = len(file_list[0])
    no_train_images = int(no_images * train_size)
    
    ### Random indeces for train-test split
    random.seed(seed)
    indeces = random.sample(range(0, no_images), no_train_images)
    
    ### Split the data into train and test
    train_files = []
    test_files = []
    for i in range(no_classes):
        temp = []
        for j in range(no_images):
            if j in indeces:
                temp.append(file_list[i][j])
        train_files.append(temp)
        test_files.append(sorted(list(set(file_list[i]) - set(temp))))
        
    return train_files, test_files


def resize_factor_cv2(img, factor=0.2):
    """
    Resize image using OpenCV.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be resized, in HW or HWC format.
    factor : int
        Factor by which the image will be resized.

    Returns
    -------
    img : 2D or 3D array-like
        Resized image, in HW or HWC format.
    """
    img = cv2.resize(img, (int(img.shape[1] * factor), int(img.shape[0] * factor)))
    
    return img


def resize_cv2(img, shape):
    """
    Resize image using OpenCV.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be resized, in HW or HWC format.
    shape : tuple
        Shape of the resized image.

    Returns
    -------
    img : 2D or 3D array-like
        Resized image, in HW or HWC format.
    """
    img = cv2.resize(img, shape)
    
    return img


def blur_cv2(img, kernel_size):
    """
    Blur image using OpenCV.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be blurred, in HW or HWC format.
    kernel_size : int
        Size of the kernel.

    Returns
    -------
    img : 2D or 3D array-like
        Blurred image, in HW or HWC format.
    """
    img = cv2.blur(img, (kernel_size, kernel_size))
    
    return img


def compute_color_histogram_cv2(img, plot=False):
    """
    Compute color histogram using OpenCV.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be used for computing the histogram, in HW or HWC format.
    plot : bool, optional
        True for plotting the histogram, False instead, by default False

    Returns
    -------
    histogram : 2D array-like
        Color histogram of shape (channels, colors).
    """
    histogram = []
    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist(images=[img], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
        histogram.append(hist)
        if plot:
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        
    if plot:
        plt.show()
    histogram = np.squeeze(np.array(histogram))
    return histogram


def similiar_color(img, color, threshhold):
    """
    Check if a pixel has a similar color to the given color.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be used for checking the color, in HW or HWC format.
    color : 1D array-like
        Color to be compared with the pixels.
    threshhold : int
        Maximum distance between the colors.

    Returns
    -------
    mask : 2D array-like
        Mask of the pixels with similar colors.
    """
    mask = 255 * np.ones(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if abs(img[i, j, 0] - color[0]) < threshhold and abs(img[i, j, 1] - color[1]) < threshhold and abs(img[i, j, 2] - color[2]) < threshhold:
                mask[i, j] = [0] * img.shape[2]
    return mask
    

def manual_segmentation(img, channel=0, threshold=100):
    """
    Segment an image using a manual threshold.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be segmented, in HW or HWC format.
    channel : int, optional
        Channel to be used for segmentation, by default 0
    threshold : int, optional
        Threshold for the segmentation, by default 100

    Returns
    -------
    mask : 2D array-like
        Mask of the segmented image.
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[img[:, :, channel] > threshold] = 255
    
    return mask


def get_sobel_gradients_cv2(img, scale=1, delta=0, ddepth=cv2.CV_16S):
    """
    Compute the gradients of an image using a Sobel filter in OpenCV.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be used for computing the gradients, in HW or HWC format.
    scale : int, optional
        Scale factor for the computed derivatives, by default 1
    delta : int, optional
        Value added to the results of cv2.Sobel() prior to return, by default 0
    ddepth : _type_, optional
        Depth of the output image, by default cv2.CV_16S to avoid overflow (16-bit signed integer).

    Returns
    -------
    grad : 2D array-like
        Gradients of the image.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    
    grad_x = cv2.Sobel(img_gray, ddepth, dx=1, dy=0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img_gray, ddepth, dx=0, dy=1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return grad


def crop_image(img, rect):
    """
    Crop an image using a rectangle.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be cropped, in HW or HWC format.
    rect : 1D array-like
        Rectangle to be used for cropping, in (x, y, w, h) format.

    Returns
    -------
    img : 2D or 3D array-like
        Cropped image, in HW or HWC format.
    """
    if len(img.shape) == 2:
        return img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    else:
        return img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :]


def get_contour_cv2(img, img_orig):
    """
    Get the contour of an image using OpenCV.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be used for computing the contour, in HW or HWC format.
    img_orig : 2D or 3D array-like
        Original image, in HW or HWC format.

    Returns
    -------
    img : 2D or 3D array-like
        Cropped contour image with the contour, in HW or HWC format.
    img_orig : 2D or 3D array-like
        Cropped contour image with the contour, in HW or HWC format.
    """
    contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_rect = np.array([0, 0, 0, 0])
    for contour in contours:
        temp_rect = cv2.boundingRect(contour) # (x, y, w, h)
        if temp_rect[2] * temp_rect[3] > max_area:
            max_area = temp_rect[2] * temp_rect[3]
            max_rect = temp_rect
    
    return crop_image(img, max_rect), crop_image(img_orig, max_rect)


def salt_pepper_noise(img, prob):
    """
    Add salt and pepper noise to an image.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be used for adding the noise, in HW or HWC format.
    prob : float
        Probability of the salt or pepper noise.

    Returns
    -------
    output : 2D or 3D array-like
        Image with the salt and pepper noise, in HW or HWC format.
    """
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rnd = random.random()
            if rnd < prob:
                output[i, j] = 0
            elif rnd > thres:
                output[i, j] = 255
            else:
                output[i, j] = img[i, j]
                
    return output


def median_filter(img, kernel_size):
    """
    Apply a median filter to an image.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be used for applying the median filter, in HW or HWC format.
    kernel_size : int
        Size of the kernel.

    Returns
    -------
    output : 2D or 3D array-like
        Image with the median filter, in HW or HWC format.
    """
    output = np.zeros(img.shape, np.uint8)
    
    for i in range(kernel_size // 2, img.shape[0] - kernel_size // 2):
        for j in range(kernel_size // 2, img.shape[1] - kernel_size // 2):
            output[i, j] = np.median(img[i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1])
                
    return output


def compute_hog_cv2(img, winSize, blockSize, blockStride, cellSize, nbins, deriv_aperture=1, win_sigma=4, histogram_norm_type=0, L2_hys_threshold=0.2, gamma_correction=0, nlevels=64, winStride=(8, 8), padding=(8, 8), locations=((10, 20),)):
    """
    Compute the Histogram of Oriented Gradients (HOG) of an image using OpenCV.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be used for computing the HOG, in HW or HWC format.
    winSize : tuple
        Size of the detection window.
    blockSize : tuple
        Size of the block data structure.
    blockStride : tuple
        Stride of the block in the window.
    cellSize : tuple
        Size of the cell data structure.
    nbins : int
        Number of bins.
    deriv_aperture : int, optional
        Aperture size for the Sobel operator, by default 1
    win_sigma : float, optional
        Gaussian smoothing window parameter, by default 4
    histogram_norm_type : int, optional
        Histogram normalization type, by default 0
    L2_hys_threshold : float, optional
        L2-Hys normalization method shrinkage, by default 0.2
    gamma_correction : int, optional
        Flag to specify whether the gamma correction preprocessing is required or not, by default 0
    nlevels : int, optional
        Maximum number of detection window increases, by default 64
    winStride : tuple, optional
        Window stride, by default (8, 8)
    padding : tuple, optional
        Padding, by default (8, 8)
    locations : tuple, optional
        Locations of the detection window, by default ((10, 20),)

    Returns
    -------
    hog : 2D array-like
        Histogram of Oriented Gradients (HOG) of the image.
    """
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, 
                            nbins, deriv_aperture, win_sigma, histogram_norm_type, 
                            L2_hys_threshold, gamma_correction, nlevels)
    hist = hog.compute(img, winStride, padding, locations)
    
    return hist


def compute_hog_scikit(img, nbins, pixels_per_cell, cells_per_block, axis=None):
    """
    Compute the Histogram of Oriented Gradients (HOG) of an image using OpenCV.

    Parameters
    ----------
    img : 2D or 3D array-like
        Image to be used for computing the HOG, in HW or HWC format.
    nbins : int
        Number of bins.
    pixels_per_cell : tuple
        Size of the cell data structure.
    cells_per_block : tuple
        Size of the block data structure.

    Returns
    -------
    hog : 2D array-like
        Histogram of Oriented Gradients (HOG) of the image.
    """
    hist = skimage.feature.hog(img, orientations=nbins, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True, channel_axis=axis)
    return hist


def svm_classifier(train_data, train_labels, test_data, test_labels, kernel='rbf', C=1, gamma='scale', seed=42):
    """
    Training a Support Vector Machine (SVM) classifier.

    Parameters
    ----------
    train_data : list of 2D or 3D array-like
        Training data.
    train_labels : list
        Training labels.
    test_data : list of 2d or 3D array-like
        Test data.
    test_labels : list
        Test labels.
    kernel : str, optional
        kernel function, by default 'rbf'.
    C : int, optional
        Cost of SVM, by default 1.
    gamma : str, optional
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid', by default 'scale'.
    seed : int, optional
        Random state seed, by default 42.
    """
    clf = SVC(C=C, kernel=kernel, gamma=gamma, random_state=seed)
    scaler = StandardScaler()
    
    train_data_scaled = scaler.fit_transform(train_data)
    clf.fit(train_data_scaled, train_labels)
    
    train_acc = clf.score(train_data_scaled, train_labels)
    print("Train accuracy: ", train_acc * 100, "%")

    test_data_scaled = scaler.transform(test_data)
    test_acc = clf.score(test_data_scaled, test_labels)
    print("Test accuracy: ", test_acc * 100, "%")
    
    
def knn_classifier(train_data, train_labels, test_data, test_labels, n_neighbors=2, weights='distance'):
    """
    Training a K-Nearest Neighbors (KNN) classifier.

    Parameters
    ----------
    train_data : list of 2D or 3D array-like
        Training data.
    train_labels : list
        Training labels.
    test_data : list of 2d or 3D array-like.
        Test data.
    test_labels : list
        Test labels.
    n_neighbors : int, optional
        Number of neighbors for KNN, by default 2.
    weights : str, optional
        Weight function used in prediction, by default 'distance'.
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    scaler = StandardScaler()

    train_data_scaled = scaler.fit_transform(train_data)
    clf.fit(train_data_scaled, train_labels)
    
    train_acc = clf.score(train_data_scaled, train_labels)
    print("Train accuracy: ", train_acc * 100, "%")

    test_data_scaled = scaler.transform(test_data)
    test_acc = clf.score(test_data_scaled, test_labels)
    print("Test accuracy: ", test_acc * 100, "%")
    