import os
from skimage.util import view_as_windows
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_corner(img, dims=(64, 64)):
    corner = img[:, -dims[0]:, -dims[1]:]
    return corner

def get_windows(img, dims=(64, 64), steps=32):
    img = np.rollaxis(img, -1)
    img = np.rollaxis(img, -1)
    windows = view_as_windows(img, (64, 64, 3), step=steps).squeeze()
    windows = np.moveaxis(windows, -1, 2)
    return windows

def get_bottom(img, dims=(64, 64), steps=32, dtype=np.float32):
    size = dims[0]
    channels = img.shape[0]
    length = (img.shape[2]-(dims[0]-steps))//steps
    bottoms = np.empty((length,channels,64,64), dtype=dtype)
    for x in range(length):
        i = img[:, -size:, x*steps:(x*steps)+64]
        bottoms[x, :, :, :] = i
    #bottoms = bottoms.astype('uint8')
    return bottoms

def get_side(img, dims=(64, 64), steps=32, dtype=np.float32):
    size = dims[1]
    channels = img.shape[0]
    length = (img.shape[1]-(dims[1]-steps))//steps
    sides = np.empty((length,channels,64,64), dtype=dtype)
    for x in range(length):
        i = img[:, x*steps:(x*steps)+64, -size:]
        sides[x, :, :, :] = i
    #sides = sides.astype('uint8')
    return sides

if __name__ == '__main__':
    file = 'monopoly64.png'
    img = cv2.imread(file, 1)
    img = np.rollaxis(img, -1)
    print('Image Shape:', img.shape)
    print('\nThe board:')
    plt.imshow(np.moveaxis(img, 0, -1))
    plt.show()
    ###############################################################################
    print('\nMain Windows:')
    windows = get_windows(img)
    print(windows.shape)
    dims=windows.shape[0]
    
    fig, axs = plt.subplots(windows.shape[0], windows.shape[1], sharex='col', sharey='row',
                            gridspec_kw={'hspace':.02, 'wspace':.02}, figsize=(16, 16))
    count = 0
    for n in range(dims):
        for m in range(dims):
            axs[n, m].imshow(np.moveaxis(windows[n, m, :, :], 0, -1))
            count+=1
    plt.show()
    
    ###############################################################################
    print('\nSide Windows:')
    side_windows = get_side(img)
    #side_windows = np.expand_dims(side_windows, 1)
    print('Side windows', side_windows.shape)
    fig, axs = plt.subplots(1, side_windows.shape[0],
                            sharex='col', sharey='row',
                            gridspec_kw={'hspace':.02, 'wspace':.02}, 
                            figsize=(16, 16))
    count = 0
    for n in range(5):
        i = np.moveaxis(side_windows.squeeze()[n, :, :, :], 0, -1)
        axs[n].imshow(i)
        count+=1
    plt.show()
    
    ###############################################################################
    print('\nBottom Windows:')
    bottom_windows = get_bottom(img)
    #bottom_windows = np.expand_dims(bottom_windows, 1)
    fig, axs = plt.subplots(1, bottom_windows.shape[0], 
                            sharex='col', sharey='row',
                            gridspec_kw={'hspace':.02, 'wspace':.02}, 
                            figsize=(16, 16))
    count = 0
    for n in range(5):
        i = np.moveaxis(bottom_windows.squeeze()[n, :, :, :], 0, -1)
        axs[n].imshow(i)
        count+=1
    plt.show()
    
    ###############################################################################
    print('\nThe Corner Window:')
    corner_window = get_corner(img)
    plt.imshow(np.moveaxis(corner_window, 0, -1))
    plt.show()












