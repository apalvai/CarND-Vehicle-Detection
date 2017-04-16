import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from sklearn.preprocessing import StandardScaler

from color_hist import color_hist
from spatial_bin import bin_spatial
from hog import get_hog_features

def convert_color(image, cspace='RGB'):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            color_space_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            color_space_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            color_space_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            color_space_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            color_space_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        color_space_image = np.copy(image)
    
    return color_space_image

def extract_features_in_image(image, cspace='RGB', spatial_size=(32, 32),
                              hist_bins=32, hist_range=(0, 256),
                              orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    
    # apply color conversion if other than 'RGB'
    color_space_image = convert_color(image, cspace=cspace)
    
    # Apply bin_spatial() to get spatial color features
    col_features = bin_spatial(color_space_image, color_space=cspace, size=spatial_size)

    # Apply color_hist() to get color histogram features
    channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features = color_hist(color_space_image, nbins=hist_bins, bins_range=hist_range)
    
    # Extract HOG features
    if hog_channel == 'ALL':
        hog_features = []
        # get hog features for all channels in the image
        for channel in range(color_space_image.shape[2]):
            hog_features_channel, hog_image_channel = get_hog_features(color_space_image[:, :, channel], orient,
                                                                       pix_per_cell, cell_per_block,
                                                                       vis=True, feature_vec=False)
            hog_features.extend(hog_features_channel)
    else:
        hog_features, hog_image = get_hog_features(color_space_image[:, :, hog_channel], orient,
                                                   pix_per_cell, cell_per_block,
                                                   vis=True, feature_vec=False)
    # process the hog_features 
    hog_features = np.ravel(hog_features)

    # combine spatial colour and histogram features
    # print('col_features: ', col_features.shape)
    # print('hist_features: ', hist_features.shape)
    # print('hog_features: ', hog_features.shape)
    feature_list = np.concatenate((col_features, hist_features, hog_features))
    return feature_list

def extract_features(image_paths, cspace='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256),
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    for image_path in image_paths:
        feature_list = []
        # Read in each one by one
        image = mpimg.imread(image_path)
        feature_list = extract_features_in_image(image, cspace=cspace, spatial_size=spatial_size,
                                                 hist_bins=hist_bins, hist_range=hist_range,
                                                 orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        # print('feature_list: ', feature_list.shape)
        
        # Append the new feature vector to the features list
        features.append(feature_list)
    # return all features
    return features

def extract_features_from(path, cspace='HSV', spatial_size=(32, 32),
                          hist_bins=32, hist_range=(0, 256),
                          orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    
    t=time.time()
    image_names = glob.glob(path)
    features = extract_features(image_names, cspace=cspace, spatial_size=spatial_size,
                                hist_bins=hist_bins, hist_range=hist_range,
                                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to extract features', len(features))
    return features

def vehicle_images_path():
    return '../vehicles/*/*.png'

def non_vehicle_images_path():
    return '../non-vehicles/*/*.png'

def extract_vehicle_features(cspace='HSV', spatial_size=(32, 32),
                             hist_bins=32, hist_range=(0, 256),
                             orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    print('extracting vehicle features...')
    images_path = vehicle_images_path()
    vehicle_features = extract_features_from(images_path, cspace=cspace, spatial_size=spatial_size,
                                             hist_bins=hist_bins, hist_range=hist_range,
                                             orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    print('found vehicle features: ', len(vehicle_features))
    return vehicle_features

def extract_non_vehicle_features(cspace='HSV', spatial_size=(32, 32),
                                 hist_bins=32, hist_range=(0, 256),
                                 orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    print('extracting non-vehicle features...')
    images_path = non_vehicle_images_path()
    non_vehicle_features = extract_features_from(images_path, cspace=cspace, spatial_size=spatial_size,
                                                 hist_bins=hist_bins, hist_range=hist_range,
                                                 orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    print('non-vehicle features: ', len(non_vehicle_features))
    return non_vehicle_features

def normalize_features(feature_list):
    print('normalizing features...')
    # Create an array stack, NOTE: StandardScaler() expects np.float64
    X = np.vstack(feature_list).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    print('total features count: ', len(scaled_X))
    # return normalized features
    return scaled_X, X, X_scaler

def test():
    # extract vehicle and non_vehicle features
    vehicle_features = extract_vehicle_features()
    non_vehicle_features = extract_non_vehicle_features()
    
    if len(vehicle_features) > 0:
        # Normalize features
        combined_features = [vehicle_features, non_vehicle_features]
        scaled_X, X, X_scaler = normalize_features((vehicle_features, non_vehicle_features))
        
        images_path = vehicle_images_path()
        image_names = glob.glob(images_path)
        
        vehicle_ind = np.random.randint(0, len(image_names))
        print('vehicle_ind: ', vehicle_ind)
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(image_names[vehicle_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[vehicle_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[vehicle_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.show()

#test()
