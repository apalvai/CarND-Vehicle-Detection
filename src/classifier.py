import numpy as np
import time
import cv2
import pickle

import os.path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from features import get_features_and_labels, extract_features_in_image, convert_color
from helpers import slide_window, draw_boxes
from color_hist import color_hist
from spatial_bin import bin_spatial
from hog import get_hog_features

def get_train_and_test_data(features, y):
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=rand_state)
    
    return X_train, X_test, y_train, y_test

def train_linear_SVC_classifer(clf, X_train, X_test, y_train, y_test):
    print('X_train: ', X_train.shape)
    print('X_test: ', X_test.shape)
    # Use a linear SVC
    if clf == None:
        # print('creating Linear SVC...')
        svc = LinearSVC(C=0.5)
        svc.fit(X_train, y_train)
        
        clf = CalibratedClassifierCV(svc, cv=2, method='isotonic')
        
        # print('creating SVC and searching for best params...')
        # parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1], 'gamma':[0.1, 1]}
        # svr = svm.SVC()
        # svc = GridSearchCV(svr, parameters)

    # Check the training time for the SVC
    print('training...')
    t=time.time()
    clf.fit(X_train, y_train)
    # print('best params: ', svc.best_params_)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return clf

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

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_vehicles_using_hog_sub_sampling(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins):
    
    windows = []
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # Get color features
            spatial_features = bin_spatial(subimg, color_space=color_space, size=spatial_size)
            channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features = color_hist(subimg, nbins=hist_bins, bins_range=(0, 256))
            
            if contains_invalid_data(spatial_features) or contains_invalid_data(hist_features) or contains_invalid_data(hog_features):
                print('found invalid data. skipping this window.')
                continue
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction_prob = svc.predict_proba(test_features)
            car_prediction_prob = test_prediction_prob[0][1]
            
            if car_prediction_prob >= 0.95:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                windows.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))

    return windows

def contains_invalid_data(data):
    return (np.any(np.isnan(data))) or (np.all(np.isfinite(data)) == False)

def detect_vehicles_using_sliding_window(image, features, y, X_scaler, svc, should_train_classifier=False):
    # Image feature extraction parameters
    color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    hist_range = (0, 256)
    
    # load the saved features
    # features, y, X_scaler = load_training_data()
    if features is None or y is None or X_scaler is None or should_train_classifier == True:
        # Retrieve features and labels
        features, y, X_scaler = get_features_and_labels(color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel)
        # save the features
        save_training_data([features, y, X_scaler])
    
    # load classifier
    # svc = load_classifier()
    if svc is None or should_train_classifier == True:
        # create and train a classifier
        for epoch in range (0, 7):
            X_train, X_test, y_train, y_test = get_train_and_test_data(features, y)
            svc = train_linear_SVC_classifer(svc, X_train, X_test, y_train, y_test)
            save_classifier(svc)
    
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255
    
    # Size of the test image is (1280, 720)
    image_shape = image.shape
    y_start_stop = [320, image_shape[0]]
    print('y_start_stop: ', y_start_stop)
    
    # windows to be explored in the image
    window_sizes = [64, 80, 96, 112, 128, 144, 160]
    windows = []
    for window_size in window_sizes:
        windows_per_size = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=(window_size, window_size), xy_overlap=(0.5, 0.5))
        windows.extend(windows_per_size)
    
    print ('windows count: ', len(windows))
    # determine the windows where a vehicle is detected
    print('searching image for vehicles...')
    hot_windows = search_windows(image, windows, svc, X_scaler,
                                 color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                                 orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    print ('hot_windows count: ', len(hot_windows))
    return hot_windows

def detect_vehicles_using_hog_sub_sampling(image, features, y, X_scaler, svc, should_train_classifier=False):
    # Image feature extraction parameters
    color_space = 'HLS'
    spatial_size = (16, 16)
    hist_bins = 16
    hist_range = (0, 256)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    
    # load the saved features
    # features, y, X_scaler = load_training_data()
    if features is None or y is None or X_scaler is None or should_train_classifier == True:
        # Retrieve features and labels
        features, y, X_scaler = get_features_and_labels(color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel)
        # save the features
        save_training_data([features, y, X_scaler])
        
    # load classifier
    # svc = load_classifier()
    if svc is None or should_train_classifier == True:
        # create and train a classifier
        for epoch in range (0, 7):
            X_train, X_test, y_train, y_test = get_train_and_test_data(features, y)
            svc = train_linear_SVC_classifer(svc, X_train, X_test, y_train, y_test)
            save_classifier(svc)
    
    image_shape = image.shape
    ystart = np.int(image_shape[0]/2)
    ystop = np.int(image_shape[0] - 60)
    scales = [0.75, 1.5, 2.25, 3.0, 3.75]
    
    hot_windows = []
    for scale in scales:
        hot_windows_per_scale = find_vehicles_using_hog_sub_sampling(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins)
        # print('found ', len(hot_windows_per_scale), ' hot windows for scale ', scale)
        hot_windows.extend(hot_windows_per_scale)
    
    # print('found hot windows: ', len(hot_windows))
    return hot_windows

def save_training_data(training_data):
    # training_data consists of features, labels and scaler(normalization) saved as training_data.pickle
    print('saving training data...')
    with open('training_data.pickle', 'wb') as f:
        pickle.dump(training_data, f)
    print('saved training data')

def load_training_data():
    # training_data consists of features, labels and scaler(normalization) saved as training_data.pickle
    print('loading training data...')
    if file_exists('training_data.pickle'):
        with open('training_data.pickle', 'rb') as f:
            features, y, X_scaler = pickle.load(f)
        print('loaded training data')
        return features, y, X_scaler
    else:
        return None, None, None

def save_classifier(clf):
    print('saving classifier...')
    with open('classifier.pickle', 'wb') as f:
        pickle.dump(clf, f)
    print('saved classifier')

def load_classifier():
    print('loading classifier...')
    if file_exists('classifier.pickle'):
        with open('classifier.pickle', 'rb') as f:
            clf = pickle.load(f)
        print('loaded classifier')
        return clf
    else:
        return None

def file_exists(file_path):
    return os.path.exists(file_path)

def test():
    # Examine the performance of classifier with a test image
    image = mpimg.imread('../test_images/test1.jpg')
    
    # load training data and classifier
    features, y, X_scaler = load_training_data()
    svc = load_classifier()
    
    # detect vehciles using sliding window
    # hot_windows = detect_vehicles_using_sliding_window(image, features, y, X_scaler, svc)
    
    # detect vehicles using sliding window and hog sub-sampling
    hot_windows = detect_vehicles_using_hog_sub_sampling(image, features, y, X_scaler, svc)
    
    # draw hot_windows on the image
    test_img = draw_boxes(image, hot_windows, (0, 0, 255), 6)
    
    # plot the image with windows
    plt.imshow(test_img)
    plt.show()

#test()
