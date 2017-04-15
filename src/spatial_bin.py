import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            color_space_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            color_space_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            color_space_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            color_space_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            color_space_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        color_space_image = np.copy(img)

    # Use cv2.resize().ravel() to create the feature vector
    small_img = cv2.resize(color_space_image, size)
    features = small_img.ravel()
    # Return the feature vector
    return features

def test():
    # Read in an image
    image = mpimg.imread('../test_images/test2.jpg')
    feature_vec = bin_spatial(image, color_space='HSV', size=(32, 32))
    
    # Plot features
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')
    plt.show()

#test()
