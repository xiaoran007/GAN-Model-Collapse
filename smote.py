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


dataset = Dataset("SouthGermanCredit")
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
# X_train_SMOTE_sel = resample(X_train_SMOTE_gen, n_samples=X_real.shape[0])
X_train_SMOTE_sel = X_train_SMOTE_gen[np.random.choice(X_train_SMOTE_gen.shape[0], size=X_real.shape[0], replace=False)]

print(type(X_train_SMOTE_sel))

total_x = np.concatenate((X_real, X_train_SMOTE_sel), axis=0)

tsne = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=50)
X_embedded = tsne.fit_transform(total_x)
# X_embedded = tsne.fit_transform(X_real)
#
# x_gen = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=50).fit_transform(X_train_SMOTE_sel)
#
print(trustworthiness(total_x, X_embedded, n_neighbors=5))
# print(trustworthiness(X_train_SMOTE_sel, x_gen, n_neighbors=5))
#

num_real_samples = X_real.shape[0]

# Plotting
plt.figure(figsize=(10, 8))

# Plot real data
plt.scatter(X_embedded[:num_real_samples, 0], X_embedded[:num_real_samples, 1], label='Original Data', alpha=0.6, c='blue')

# Plot generated data
plt.scatter(X_embedded[num_real_samples:, 0], X_embedded[num_real_samples:, 1], label='Generated Data', alpha=0.6, c='red')

# Add legend
plt.legend()

# plt.figure(figsize=(10, 8))
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], label='Original Data', alpha=0.6, c='blue')
# plt.scatter(x_gen[:, 0], x_gen[:, 1], label='Generated Data', alpha=0.6, c='red')
# plt.legend()
plt.title("t-SNE visualization of Original and Generated Data")
plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.show()

