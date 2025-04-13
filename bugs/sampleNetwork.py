# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:31:20 2023

@author: Pawel Pieta, papi@dtu.dk
"""
import numpy as np

def netFunc(images):
    '''
    Loads a feed forward neural network, and predicts the labels of each image in an array.

    Parameters
    ----------
    images : numpy array
        An array with grayscale images loaded with "skimage.io.imread(fname,as_gray=True)",
        float datatype with pixel values between 0 and 1. The array has \n
        dimension N x H x W x C where N is the number of images, H is the height of the images, \n
        W is the width of the images and C is the number of channels in each image.

    Returns
    -------
    class_indices : Iterable[int]
        An iterable (e.g., a list) of the class indices representing the predictions of the network. The valid indices range from 0 to 11.

    '''
    
    # Specify here the main path to where you keep your model weights, (in case we need to modify it)
    # try to make the path relative to this code file location, 
    # e.g "../weights/v1/" instead of "C:/AdvImg/week10/weights/v1"
    MODEL_PATH = "../weights/weights_3.npy"
    
            
    # Change this to return a list of indices predicted by your model
    class_indices = np.random.randint(0,12, size=images.shape[0])

    return class_indices


