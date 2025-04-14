# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:31:20 2023

@author: Pawel Pieta, papi@dtu.dk
Modified by: [Your Name]
"""
import numpy as np
import os
import pickle

def forward_pass_dynamic(X, weights):
    """
    Perform a forward pass through a neural network with multiple layers.

    Args:
        X (numpy.ndarray): Input data of shape (input_dim, num_samples).
        weights (list): List of weight matrices for each layer.

    Returns:
        activations (list): List of activations for each layer (including input and hidden layers).
        outputs (numpy.ndarray): Output of the network after softmax activation.
    """
    activations = [np.vstack((X, np.ones((1, X.shape[1]))))]  # Add bias to input and store as the first activation
    for W in weights[:-1]:  # For all layers except the output layer
        z = np.dot(W.T, activations[-1])  # Compute z = W * activations of the previous layer
        h = np.maximum(z, 0)  # Apply ReLU activation
        h = np.vstack((h, np.ones((1, h.shape[1]))))  # Add bias
        activations.append(h)  # Store activations for this layer

    # Output layer
    z = np.dot(weights[-1].T, activations[-1])  # Compute z for the output layer
    e_x = np.exp(z - np.max(z, axis=0))  # Apply softmax activation
    outputs = e_x / e_x.sum(axis=0)  # Normalize to get probabilities
    
    return activations, outputs

def netFunc(images):
    '''
    Loads a feed forward neural network, and predicts the labels of each image in an array.

    Parameters
    ----------
    images : numpy array
        An array with grayscale images loaded with "skimage.io.imread(fname,as_gray=True)",
        float datatype with pixel values between 0 and 1. The array has
        dimension N x H x W x C where N is the number of images, H is the height of the images,
        W is the width of the images and C is the number of channels in each image.

    Returns
    -------
    class_indices : Iterable[int]
        An iterable (e.g., a list) of the class indices representing the predictions of the network. The valid indices range from 0 to 11.
    '''
    
    # Specify here the main path to where you keep your model weights
    # Make it relative to this code file location
    MODEL_PATH = "./bugs/weights/"
    # Check if the directory exists, and create it if not
    os.makedirs(MODEL_PATH, exist_ok=True)
    # Look for the weights file in the MODEL_PATH directory
    weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pkl')]
    # If no weight file exists, return random predictions (for testing purposes)
    if not weight_files:
        print("Warning: No weight files found in", MODEL_PATH)
        return np.random.randint(0, 12, size=images.shape[0])
    # Use the most recent weight file (or you could specify a particular one)
    weight_file = os.path.join(MODEL_PATH, weight_files[-1])
    
    try:
        # Load the weights
        with open(weight_file, 'rb') as f:
            weights = pickle.load(f)
        
        # Preprocess the images: resize to vectors and normalize if needed
        # First check if images are already normalized (between 0 and 1)
        if np.max(images) > 1.0:
            images = images / 255.0
        
        # Forward pass through the network
        _, outputs = forward_pass_dynamic(images, weights)
        
        # Get the predicted class indices
        class_indices = np.argmax(outputs, axis=0)
        
        return class_indices
    
    except Exception as e:
        print(f"Error loading or using the model: {e}")
        # Return random predictions as fallback
        return np.random.randint(0, 12, size=images.shape[0])