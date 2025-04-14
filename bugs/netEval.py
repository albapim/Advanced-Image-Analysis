# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:30:09 2023

@author: Pawel Pieta, papi@dtu.dk
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import skimage.io as io
import cv2
import time
from sklearn.model_selection import train_test_split

from bugs.sampleNetwork import netFunc

def one_hot_encode(targets, num_classes=None):
    """
    One-hot encode the target values.

    Args:
        targets (numpy.ndarray): Array of target values (integers).
        num_classes (int): Total number of classes. If None, inferred from the targets.

    Returns:
        numpy.ndarray: One-hot encoded matrix of shape (num_samples, num_classes).
    """
    if num_classes is None:
        num_classes = np.max(targets) + 1  # Infer number of classes from the targets
    one_hot = np.zeros((len(targets), num_classes))
    one_hot[np.arange(len(targets)), targets] = 1
    return one_hot

def resize_images_to_vectors(images):
    num_images, height, width = images.shape
    return images.reshape(num_images, height * width)

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = io.imread(os.path.join(folder, filename))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = img_gray / 255.0  # Normalize pixel values to [0, 1]
        images.append(img_gray)

    return np.array(images)


def netEval(netFunc, dataPath, targetPath):
    '''
    Evaluates the accuracy of a classification model using provided dataset.
    
    Parameters
    ----------
    netFunc : function
        Function of the network that takes an image and outputs a predicted class
    dataPath: string
        Path to a folder with data
    targetPath: string 
        Path to a file with target labels (either .txt or .npy)
    
    Returns
    -------
    accuracy: float
        Accuracy of the network on provided dataset
    execTime:
        Network prediction execution time
    '''
    
    assert callable(netFunc), "The first argument is not callable, it should be a network function."
    assert os.path.exists(dataPath), f"Provided path: {dataPath} does not exist."
    assert os.path.exists(targetPath), f"Provided path: {targetPath} does not exist."
    
    ext = os.path.splitext(targetPath)[-1]
    assert ext == '.txt' or ext == '.npy', f"Target path extension file {ext} is not supported, use .txt or .npy"
    
    targets = np.loadtxt(targetPath, dtype=int)
    one_hot_targets = one_hot_encode(targets)

    X_images = load_images_from_folder(dataPath)
    X_vectors = resize_images_to_vectors(X_images)

    # Split the dataset into training and testing sets
    # take only test set
    _, X_test, _, y_test = train_test_split(X_vectors, one_hot_targets, test_size=0.2, random_state=42)
    X_test = X_test.T
    y_test = np.argmax(y_test.T, axis=0)

    # Execute network
    t0 = time.time()
    predList = netFunc(X_test)
    t1 = time.time()
    
    # Calculate accuracy and execution time
    accuracy = np.sum(np.equal(predList,y_test))/len(y_test)
    execTime = t1-t0    
    
    return accuracy, execTime
        
    

if __name__ == "__main__":
    
    targetPath = '/home/amapy/Desktop/advanced_image_analysis_2025/Advanced-Image-Analysis/bugs/data/week9_BugNIST2D_train/train_targets.txt'
    dataPath = '/home/amapy/Desktop/advanced_image_analysis_2025/Advanced-Image-Analysis/bugs/data/week9_BugNIST2D_train/train/' 
    
    accuracy, execTime = netEval(netFunc, dataPath, targetPath)
    
    print(f"Achieved accuracy: {np.round(accuracy,4)}")
    print(f"Network execution time: {np.round(execTime,4)}s")