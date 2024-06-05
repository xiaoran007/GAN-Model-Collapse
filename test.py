from GAN import get_default_device, get_generator
from sklearn.model_selection import train_test_split
import numpy as np
import DatasetsLoader


def split_dataset_random(X, y):
    # Split dataset into 7:1:2 for training : validation : testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=None, stratify=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test


def GANs_two_class_real_data(X_train, y_train):  # Defining the real data for GANs
    X_real = []
    y_train = y_train.ravel()
    for i in range(len(y_train)):
        if int(y_train[i]) == 1:
            X_real.append(X_train[i])
    X_real = np.array(X_real)
    y_real = np.ones((X_real.shape[0],))
    return X_real, y_real


dataset = DatasetsLoader.Dataset(dataset_name="PredictTerm")
X, y = dataset.GetDataset()
X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_random(X, y)
lr = 0.0002
epochs = 150
batch_size = 128

print(f"data size: {len(y_train)}")
X_real, y_real = GANs_two_class_real_data(X_train, y_train)
print(f"real data size: {len(y_real)}")
generator_G = get_generator(X_train, X_real, y_real, get_default_device(force_to_cpu=False, force_skip_mps=True), lr, epochs, batch_size, 1, 0)
