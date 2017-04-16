import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from features import extract_vehicle_features, extract_non_vehicle_features, normalize_features

def get_features_and_labels():
    color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    
    # extract vehicle and non_vehicle features
    vehicle_features = extract_vehicle_features(cspace=color_space, spatial_size=spatial_size,
                                                hist_bins=hist_bins, hist_range=(0, 256),
                                                orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
                                                
    non_vehicle_features = extract_non_vehicle_features(cspace=color_space, spatial_size=spatial_size,
                                                        hist_bins=hist_bins, hist_range=(0, 256),
                                                        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    
    # Normalize features
    combined_features = [vehicle_features, non_vehicle_features]
    scaled_X, X = normalize_features((vehicle_features, non_vehicle_features))
    
    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
    
    return scaled_X, y

def get_train_and_test_data(features, y):
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=rand_state)
    
    return X_train, X_test, y_train, y_test

def train_linear_SVC_classifer(svc, X_train, X_test, y_train, y_test):
    
    # Use a linear SVC
    if svc == None:
        print('creating Linear SVC...')
        svc = LinearSVC()
    
    # Check the training time for the SVC
    print('training LinearSVC...')
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return svc

def test():
    # Retrieve features and labels
    features, y = get_features_and_labels()
    svc = None
    for epoch in range (0, 2):
        X_train, X_test, y_train, y_test = get_train_and_test_data(features, y)
        svc = train_linear_SVC_classifer(svc, X_train, X_test, y_train, y_test)

test()

