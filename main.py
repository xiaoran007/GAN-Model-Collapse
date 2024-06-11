from GAN import get_default_device, get_generator, Generator, Discriminator
from sklearn.model_selection import train_test_split
import numpy as np
import DatasetsLoader
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness


def split_dataset_random(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y)

    return X_train, y_train, X_test, y_test


def GANs_two_class_real_data(X_train, y_train):  # Defining the real data for GANs
    X_real = []
    y_train = y_train.ravel()
    for i in range(len(y_train)):
        if int(y_train[i]) == 1:
            X_real.append(X_train[i])
    X_real = np.array(X_real)
    y_real = np.ones((X_real.shape[0],))
    return X_real, y_real


def load_checkpoint(name):
    generator = Generator(X_train.shape[1], X_train.shape[1], 128)
    discriminator = Discriminator(X_train.shape[1], 128)
    checkpoint = torch.load(f"checkpoint/checkpoint_{name}.pth")
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    return generator, discriminator


def get_minor_major(y):
    minor, major = 0, 0
    for i in y:
        if i == 1:
            minor += 1
        else:
            major += 1
    return minor, major



dataset = DatasetsLoader.Dataset(dataset_name="PredictTerm")
X, y = dataset.GetDataset()
X_train, y_train, X_test, y_test = split_dataset_random(X, y)
lr = 0.0002
epochs = 100
batch_size = 128
device = get_default_device(force_to_cpu=False, force_skip_mps=True)

print(f"data size: {len(y_train)}")
X_real, y_real = GANs_two_class_real_data(X_train, y_train)
print(f"real data size: {len(y_real)}")
# generator_G = get_generator(X_train, X_real, y_real, device, lr, epochs, batch_size, 1, 0)

# generator, discriminator = load_checkpoint(epochs-1)
generator, discriminator = load_checkpoint(50)
generator.eval()

minor, major = get_minor_major(y_train)
need_gen = major - minor
GANs_noise = torch.randn((X_real.shape[0]), (X_real.shape[1]), device=device)
output = generator(GANs_noise.float().to(device)).cpu().detach().numpy()
print(output)
print(len(output))


X = X_real
y = y_real

tsne = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=100)
X_embedded = tsne.fit_transform(X)

x_gen = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=100).fit_transform(output)

print(trustworthiness(X, X_embedded, n_neighbors=5))
print(trustworthiness(output, x_gen, n_neighbors=5))

plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], label='Original Data', alpha=0.6, c='blue')
plt.scatter(x_gen[:, 0], x_gen[:, 1], label='Generated Data', alpha=0.6, c='red')
plt.legend()
plt.title("t-SNE visualization of Original and Generated Data")
plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.show()
