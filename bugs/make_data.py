
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os

def make_data(example_nr, n_pts = 200, noise = 1):
    '''Make data for the neural network. The data is read from a png file in the 
    cases folder, which must be placed together with your code.
    
    Parameters:
    example_nr : int
        1-7
    n_pts : int
        Number of points in each of the two classes
    noise : float
        Standard deviation of the Gaussian noise

    Returns:
    X : ndarray
        2 x n_pts array of points
    T : ndarray
        2 x n_pts array of boolean values
    x_grid : ndarray
        2 x n_pts array of points in regular grid for visualization
    dim : tuple of int
        Dimensions of the grid
    
    Example:
    example_nr = 1
    n_pts = 2000
    noise = 2
    X, T, x_grid, dim = make_data(example_nr, n_pts, noise)

    fig, ax = plt.subplots()
    ax.plot(X[0,T[0]], X[1,T[0]], '.r', alpha=0.3)
    ax.plot(X[0,T[1]], X[1,T[1]], '.g', alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_box_aspect(1)

    Authors: Vedrana Andersen Dahl and Anders Bjorholm Dahl - 20/3-2024
    vand@dtu.dk, abda@dtu.dk

    '''

    in_dir = 'cases/'
    file_names = sorted(os.listdir(in_dir))
    file_names = [f for f in file_names if f.endswith('.png')]

    im = skimage.io.imread(in_dir + file_names[example_nr-1])
    
    [r_white, c_white] = np.where(im == 255)
    [r_gray, c_gray] = np.where(im == 127)
    n_white = np.minimum(r_white.shape[0], n_pts)
    n_gray = np.minimum(r_gray.shape[0], n_pts)

    rid_white = np.random.permutation(r_white.shape[0])
    rid_gray = np.random.permutation(r_gray.shape[0])
    pts_white = np.array([c_white[rid_white[:n_white]], r_white[rid_white[:n_white]]])
    pts_gray = np.array([c_gray[rid_gray[:n_gray]], r_gray[rid_gray[:n_gray]]])

    X = np.hstack((pts_white, pts_gray))/5 + np.random.randn(2, n_white+n_gray)*noise
    T = np.zeros((2, n_white+n_gray), dtype=bool)
    T[0,:n_white] = True
    T[1,n_white:] = True

    dim = (100, 100)
    QX, QY = np.meshgrid(range(0, dim[0]), range(0, dim[1]))
    x_grid = np.vstack((np.ravel(QX), np.ravel(QY)))

    return X, T, x_grid, dim

if __name__ == "__main__":
    example_nr = 1
    n_pts = 2000
    noise = 3
    X, T, x_grid, dim = make_data(example_nr, n_pts, noise)

    fig, ax = plt.subplots()
    ax.plot(X[0,T[0]], X[1,T[0]], '.r', alpha=0.3)
    ax.plot(X[0,T[1]], X[1,T[1]], '.g', alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_box_aspect(1)

