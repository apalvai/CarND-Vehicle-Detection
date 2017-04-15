import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from features import extract_vehicle_features, extract_non_vehicle_features, normalize_features

def get_features_and_labels():
    # extract vehicle and non_vehicle features
    vehicle_features = extract_vehicle_features()
    non_vehicle_features = extract_non_vehicle_features()
    
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
    for epoch in range (0, 5):
        X_train, X_test, y_train, y_test = get_train_and_test_data(features, y)
        svc = train_linear_SVC_classifer(svc, X_train, X_test, y_train, y_test)

test()

