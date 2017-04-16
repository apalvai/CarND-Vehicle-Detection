import numpy as np
import time
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from features import extract_vehicle_features, extract_non_vehicle_features, normalize_features, extract_features_in_image
from helpers import slide_window, draw_boxes

def get_features_and_labels():
    color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11  # HOG orientations
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
    scaled_X, X, X_scaler = normalize_features((vehicle_features, non_vehicle_features))
    
    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
    
    return scaled_X, y, X_scaler

def get_train_and_test_data(features, y):
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=rand_state)
    
    return X_train, X_test, y_train, y_test

def train_linear_SVC_classifer(svc, X_train, X_test, y_train, y_test):
    print('X_train: ', X_train.shape)
    print('X_test: ', X_test.shape)
    # Use a linear SVC
    if svc == None:
        print('creating Linear SVC...')
        svc = LinearSVC()
        
#        parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10], 'gamma':[0.1, 1, 10]}
#        svr = svm.SVC()
#        svc = GridSearchCV(svr, parameters)

    # Check the training time for the SVC
    print('training LinearSVC...')
    t=time.time()
    svc.fit(X_train, y_train)
#    print('best params: ', svc.best_params_)
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

def search_windows(img, windows, clf, scaler,
                   color_space='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
                   orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using extract_features_in_image()
        features = extract_features_in_image(test_img, cspace=color_space, spatial_size=spatial_size,
                                             hist_bins=hist_bins, hist_range=hist_range,
                                             orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        #5) Scale extracted features to be fed to classifier
        features = np.array(features).astype(np.float64)
        test_features = scaler.transform(features.reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    
    #8) Return windows for positive detections
    return on_windows

def test():
    # Retrieve features and labels
    features, y, X_scaler = get_features_and_labels()
    
    # create and train a classifier
    svc = None
    for epoch in range (0, 1):
        X_train, X_test, y_train, y_test = get_train_and_test_data(features, y)
        svc = train_linear_SVC_classifer(svc, X_train, X_test, y_train, y_test)
    
    print('svc', svc)
    # Examine the performance of classifier with a test image
    image = mpimg.imread('../test_images/test1.jpg')
    
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255

    # Size of the test image is (1280, 720)
    image_shape = image.shape
    y_start_stop = [320, image_shape[0]]
    print('y_start_stop: ', y_start_stop)
    
    # windows to be explored in the image
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=(64, 64), xy_overlap=(0.5, 0.5))
    print ('windows count: ', len(windows))
    # determine the windows where a vehicle is detected
    print('searching image for vehicles...')
    hot_windows = search_windows(image, windows, svc, X_scaler,
                                 color_space='HLS', spatial_size=(16, 16), hist_bins=16, hist_range=(0, 256),
                                 orient=11, pix_per_cell=8, cell_per_block=2, hog_channel='ALL')
    print ('hot_windows count: ', len(hot_windows))

    # draw the boxes for detected hot_windows
    draw_image = np.copy(image)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()

test()

