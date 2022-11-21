import numpy as np
import cv2 as cv


def calculate_fv(mags, dirs, bins):
    """
    Calculating feature vector for the cell
    :param mags:
    Magnitudes of gradient for the cell
    :param dirs:
    Directions of gradient for the cell
    :param bins:
    Number of bins of the histogram (feature vector dimension)
    :return:
    Feature vector
    """
    angles = np.linspace(0, 180, bins, endpoint=False)
    delta = angles[1] - angles[0]
    hist = np.zeros(bins)

    for i in range(bins):
        # Getting the indexes of angles in dirs matrix inside the current bin
        idx = None
        if i != bins - 1:
            idx = np.where((dirs>=angles[i])*(dirs<angles[i+1]))  # YOUR CODE HERE
        else:
            idx = np.where((dirs>=angles[i])) 

        props = (dirs[idx] - angles[i]) / delta
        # Calculate the feature vector for the cell
        if i != bins - 1:
            hist[i] += np.sum((1-props)*mags[idx])  # YOUR CODE HERE
            hist[i + 1] += np.sum(props*mags[idx]) # YOUR CODE HERE
        else:
            hist[i] += np.sum((1-props)*mags[idx])  # YOUR CODE HERE
            hist[0] += np.sum(props*mags[idx])  # YOUR CODE HERE
        # Can you optimize this code?
    return hist


def HOG(img, img_size=(64, 128), cell_size=(8, 8), block_size=(2, 2), orientations=9):
    """
    Returns the HOG feature vector of the given image
    :param img: 
    The image path
    :param img_size: 
    The resized img size
    type: tuple of size (int, int), default (64, 128)
    :param cell_size: 
    Size of cell in pixels
    type: tuple of size (int, int), default (8,8)
    :param block_size:
    Number of cells in each block
    type: tuple of size (int,int), default (2,2)
    :param orientations: 
    Number of bins for each cell gradient histogram
    type: int, default 9
    :return:
    The final feature vector for the image
    """
    # Read image using cv2
    image = cv.imread(img)
    # Resizing it into give shape
    image = cv.resize(image, img_size, interpolation=cv.INTER_AREA)
    # Making it grayscale, i.e one channel
    # Try to make HOG feature vector on multichannel image
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Now calculate gx and gy
    # Hint use gradient function from numpy

    # YOUR CODE STARTS HERE
    gx = np.gradient(image, axis=1)
    gy = np.gradient(image, axis=0)
    # YOUR CODE ENDS HERE

    # Calculating magnitudes and directions
    # Hint np.arctan() returns a number between [-pi/2; pi/2], but we need between [0; 180]

    # YOUR CODE STARTS HERE

    g_mags = np.sqrt(gx**2 + gy**2)
    g_dirs = np.arctan(gy/(gx+1e-10))*180/np.pi + 180*(np.arctan(gy/(gx+1e-10))<0)

    # YOUR CODE ENDS HERE

    x_block, x_stride = block_size[0] * cell_size[0], cell_size[0]
    y_block, y_stride = block_size[1] * cell_size[1], cell_size[1]

    fv = []  # Final feature vector

    # Loop for blocks
    for i in range(0, img_size[1] - y_stride * (block_size[1] - 1), y_stride):
        for j in range(0, img_size[0] - x_stride * (block_size[0] - 1), x_stride):
            block_v = []
            cell_v = []
            # Loop for cell
            for _ in range(block_size[0]):
                for _a_ in range(block_size[1]):
                    # Taking directions and magnitudes cell matrices
                    g_dir_c = g_dirs[i:i+y_stride, j:j+x_stride]  
                    g_mags_c = g_mags[i:i+y_stride, j:j+x_stride]  
                    i = i + y_stride
                    j = j + x_stride
                    cell_v.extend(calculate_fv(g_mags_c, g_dir_c, orientations))
            # Can you optimize this algorithm?
            block_v = np.array(cell_v).ravel()
            # Normalize block_v
            block_v = block_v/np.sqrt(np.sum(block_v**2)+1e-12)  # YOUR CODE HERE
            fv.extend(block_v)
    return np.array(fv).ravel(), gx, gy


fv = HOG('hog-histogram-1.png', img_size=(400, 456))

