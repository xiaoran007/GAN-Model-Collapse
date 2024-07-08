from sklearn.model_selection import train_test_split
import numpy as np


def split_dataset_random(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)

    return X_train, y_train, X_test, y_test


def GANs_two_class_real_data(X_train, y_train):
    X_real = []
    y_train = y_train.ravel()
    for i in range(len(y_train)):
        if int(y_train[i]) == 1:
            X_real.append(X_train[i])
    X_real = np.array(X_real)
    y_real = np.ones((X_real.shape[0],))
    return X_real, y_real


def GANs_two_class_majority_data(X_train, y_train):
    X_majority = []
    y_train = y_train.ravel()
    for i in range(len(y_train)):
        if int(y_train[i]) == 0:
            X_majority.append(X_train[i])
    X_majority = np.array(X_majority)
    y_majority = np.zeros((X_majority.shape[0],))
    return X_majority, y_majority


def get_minor_major(y):
    minor, major = 0, 0
    for i in y:
        if i == 1:
            minor += 1
        else:
            major += 1
    return minor, major


