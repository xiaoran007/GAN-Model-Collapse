import pandas as pd
from imblearn.over_sampling import SMOTE
from DatasetsLoader import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
import numpy as np

def split_dataset_random(X, y):
    # Split dataset into 7:1:2 for training : validation : testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)

    return X_train, y_train, X_test, y_test


dataset = Dataset("CreditCard")
X, y = dataset.GetDataset()
X_train, y_train, X_test, y_test = split_dataset_random(X, y)
X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)

minor, major = 0, 0
for i in y_train:
    if i == 1:
        minor += 1
    else:
        major += 1
print(f"minor: {minor}, major: {major}")

minor, major = 0, 0
for i in y_train_SMOTE:
    if i == 1:
        minor += 1
    else:
        major += 1
print(f"minor: {minor}, major: {major}")


print(X_train.shape)
print(X_train_SMOTE.shape)

def GANs_two_class_real_data(X_train, y_train):  # Defining the real data for GANs
    X_real = []
    y_train = y_train.ravel()
    for i in range(len(y_train)):
        if int(y_train[i]) == 1:
            X_real.append(X_train[i])
    X_real = np.array(X_real)
    y_real = np.ones((X_real.shape[0],))
    return X_real, y_real

X_real, y_real = GANs_two_class_real_data(X_train, y_train)
X_train_SMOTE_gen = X_train_SMOTE[X_train.shape[0]:]
X_train_SMOTE_sel = resample(X_train_SMOTE_gen, n_samples=X_real.shape[0])

print(type(X_train_SMOTE_sel))

columns = [f'feature{i+1}' for i in range(X_real.shape[1])]
original_data = pd.DataFrame(X_real, columns=columns)
generated_data = pd.DataFrame(X_train_SMOTE_sel, columns=columns)

features = original_data.columns
n_features = len(features)

# Create a figure with subplots in a grid with 3 columns
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols  # Calculate the number of rows needed

plt.figure(figsize=(20, n_rows * 5))

for i, feature in enumerate(features):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.hist(original_data[feature], bins=30, alpha=0.7, label='Original', color='blue')
    plt.hist(generated_data[feature], bins=30, alpha=0.7, label='Generated', color='orange')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.show()
