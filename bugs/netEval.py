# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:30:09 2023

@author: Pawel Pieta, papi@dtu.dk
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import skimage.io
import time

from bugs.sampleNetwork import netFunc

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
    
    if ext == '.txt':
        with open(targetPath) as f:
            targetList = np.array(f.readlines()).astype(int)
    else:
        targetList = np.load(targetPath)
    
    
    # Read in the images
    imgList = glob.glob(dataPath+'/*.png')
    assert imgList, f"No .png images found in folder {targetPath}"
    imgsArr = np.array([skimage.io.imread(fname) for fname in imgList])
    
    # Execute network
    t0 = time.time()
    predList = netFunc(imgsArr)
    t1 = time.time()
    
    # Calculate accuracy and execution time
    accuracy = np.sum(np.equal(predList,targetList))/len(targetList)
    execTime = t1-t0
    
    return accuracy, execTime
        
    

if __name__ == "__main__":
    
    targetPath = '/home/amapy/Desktop/advanced_image_analysis_2025/Advanced-Image-Analysis/bugs/data/week9_BugNIST2D_train/train_targets.txt'
    dataPath = '/home/amapy/Desktop/advanced_image_analysis_2025/Advanced-Image-Analysis/bugs/data/week9_BugNIST2D_train/train' 
    
    accuracy, execTime = netEval(netFunc, dataPath, targetPath)
    
    print(f"Achieved accuracy: {np.round(accuracy,4)}")
    print(f"Network execution time: {np.round(execTime,4)}s")