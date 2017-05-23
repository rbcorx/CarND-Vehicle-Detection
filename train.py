import time

import glob

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from utils import convert_color, extract_features


file = "model.pk"
data = {}

def save(item, key, commit=False):
    print ("saving: {}".format(key))
    # import ipdb
    # ipdb.set_trace()
    data[key] = item
    if commit:
        pickle.dump(data, open(file, 'wb'))


def get_clf():
    # TODO gridsearch
    params = {
        "kernel": ["rbf"],
        # "C": [1.0, 10.0, 100, 1000],
        # "gamma": ["auto", 1.0, 10.0],
    }
    svr = svm.SVC()

    return GridSearchCV(svr, params)


def get_features_scaler():
    scaler = StandardScaler()
    def scale(features):
        scaler.fit(features)
        save(scaler, "scaler")
        return scaler.transform(features)

    return scale


def load_data(root, paths, f_patterns, lim=99999, y=True):
    files = []
    [[files.extend(glob.glob("{}{}{}".format(root, path, f_pattern))) for path in paths] for f_pattern in f_patterns]
    files = files[:lim]

    # load images and convert colorspace to YCrCb
    # TODO try changing to BGR2LUV
    labels = np.ones(len(files)) if y else np.zeros(len(files))

    return (files, labels)


def trim_features():
    """trim features using decision trees to balance features"""
    pass

def train(x_features, y):
    random_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=random_state)

    clf = get_clf()
    t1 = time.time()
    clf.fit(x_train, y_train)
    print("time taken for training: {}".format(time.time() - t1))
    print("test accuracy of classifier: {}".format(clf.score(x_test, y_test)))

    print("best params: {}".format(clf.best_params_))

    return clf


def test():
    vehicle_root = "vehicles/"
    vehicle_paths = ["KITTI_extracted/"]
    non_vehicle_root = "non-vehicles/"
    non_vehicle_paths = ["Extras/"]
    file_patterns = ["*.png"]

    limit = 50

    vehicles_x, vehicles_y = load_data(vehicle_root, vehicle_paths, file_patterns, lim=limit, y=True)
    non_vehicles_x, non_vehicles_y = load_data(non_vehicle_root, non_vehicle_paths, file_patterns, lim=limit, y=False)

    x = []
    y = []
    x.extend(vehicles_x)
    x.extend(non_vehicles_x)

    y.extend(vehicles_y)
    y.extend(non_vehicles_y)

    # np.vstack((vehicles_x, non_vehicles_x)).astype(np.float64), np.hstack((vehicles_y, non_vehicles_y))

    combi = list(zip(x, y))
    np.random.shuffle(combi)
    x, y = list(zip(*combi))

    print("loading files...")
    t1 = time.time()
    images = [convert_color(cv2.imread(file)) for file in x]
    print ("time taken = {}".format(time.time() - t1))

    x = np.array(images, np.float64)
    y = np.array(y)

    settings = {
        'orient': 16,
        'pix_per_cell': 8,
        'cell_per_block': 2,
        'hog_channel': 'ALL',
        'spatial_size': (8, 8),
        'hist_bins': 32,
        'hist_range': (0, 256),
    }

    features = extract_features(x, **settings)

    scaler = get_features_scaler()

    scaled_features = scaler(features)

    print(len(features[0]))

    clf = train(scaled_features, y)

    save(clf, "clf", True)




if __name__ == "__main__":
    # test()

    # Read in cars and notcars
    images = glob.glob('*vehicles/*/*')
    cars = []
    notcars = []
    for image in images:
        if 'non' in image:
            notcars.append(image)
        else:
            cars.append(image)
    ## Uncomment if you need to reduce the sample size
    #sample_size = 500
    #cars = cars[0:sample_size]
    #notcars = notcars[0:sample_size]
    print(len(cars))
    print(len(notcars))

    # Define parameters for feature extraction
    color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 8  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    car_features = extract_features(cars, cspace=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel)
    print ('Car samples: ', len(car_features))
    notcar_features = extract_features(notcars, cspace=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel)
    print ('Notcar samples: ', len(notcar_features))
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    X_scaler = StandardScaler().fit(X) # Fit a per-column scaler
    scaled_X = X_scaler.transform(X) # Apply the scaler to X

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)))) # Define the labels vector

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)

    print('Using:',orient,'orientations', pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    svc = svm.LinearSVC(loss='hinge') # Use a linear SVC
    t=time.time() # Check the training time for the SVC
    svc.fit(X_train, y_train) # Train the classifier
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4)) # Check the score of the SVC

    save(svc, "clf", True)




