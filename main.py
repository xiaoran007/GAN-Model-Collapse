from GAN import get_default_device, get_generator, Generator, Discriminator
import DatasetsLoader
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from myutil import split_dataset_random, GANs_two_class_real_data, ModelNotFitException
import json
import os
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from ctgan import CTGAN
import numpy as np
from PIL import Image


class GANTsne:
    def __init__(self, dataset_name, device):
        self.dataset_name = dataset_name
        self.lr = 0.0002
        self.epochs = 150
        self.batch_size = 128
        self.X_train = None
        self.y_train = None
        self.X_real = None
        self.y_real = None
        self._load_dataset()
        self.device = device

    def _load_dataset(self):
        dataset = DatasetsLoader.Dataset(dataset_name=self.dataset_name)
        X, y = dataset.GetDataset()
        X_train, y_train, _, _ = split_dataset_random(X, y)
        self.X_train = X_train
        self.y_train = y_train
        self._set_real_data()

    def _set_real_data(self):
        self.X_real, self.y_real = GANs_two_class_real_data(self.X_train, self.y_train)

    def _load_checkpoint(self, epoch):
        generator = Generator(self.X_train.shape[1], self.X_train.shape[1], 128)
        discriminator = Discriminator(self.X_train.shape[1], 128)
        checkpoint = torch.load(f"checkpoint/{self.dataset_name}/checkpoint_{epoch}.pth")
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        return generator.to(self.device), discriminator.to(self.device)

    def _pre_trained(self):
        try:
            with open(f"checkpoint/{self.dataset_name}/checkpoints.json", "r") as f:
                config = json.load(f)
            if self.epochs == config["epochs"] and self.batch_size == config["batch_size"] and self.lr == config["lr"] and f"{self.device}" == config["device"]:
                reuse = True
            else:
                reuse = False
        except Exception as e:
            reuse = False

        return reuse

    def fit(self):
        if not self._pre_trained():
            get_generator(self.X_train, self.X_real, self.y_real, self.device, self.lr, self.epochs, self.batch_size, 1, 0, dataset_name=self.dataset_name)
            with open(f"checkpoint/{self.dataset_name}/checkpoints.json", "w") as f:
                config = {"epochs": self.epochs, "lr": self.lr, "batch_size": self.batch_size, "device": f"{self.device}"}
                json.dump(config, f)
        else:
            print("Found pre-trained model, skipping training")

    def draw_and_save(self, epoch):
        assert epoch in range(self.epochs)
        if os.path.exists(f"./result/{self.dataset_name}"):
            pass
        else:
            os.mkdir(f"./result/{self.dataset_name}")
        generator, _ = self._load_checkpoint(epoch)
        generator.eval()

        GANs_noise = torch.randn((self.X_real.shape[0]), (self.X_real.shape[1]), device=self.device)
        output = generator(GANs_noise.float().to(self.device)).cpu().detach().numpy()

        if self.X_real.shape[0] <= 150:
            perplexity = self.X_real.shape[0] / 2
        else:
            perplexity = 100

        real_data = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=perplexity).fit_transform(self.X_real)
        generated_data = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=perplexity).fit_transform(output)

        print(trustworthiness(self.X_real, real_data, n_neighbors=5))
        print(trustworthiness(output, generated_data, n_neighbors=5))

        plt.figure(figsize=(10, 8))
        plt.scatter(real_data[:, 0], real_data[:, 1], label='Original Data', alpha=0.6, c='blue')
        plt.scatter(generated_data[:, 0], generated_data[:, 1], label='Generated Data', alpha=0.6, c='red')
        plt.legend()
        plt.title(f"t-SNE visualization of {self.dataset_name}-GAN {epoch}")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.savefig(f"./result/{self.dataset_name}/t-SNE_{epoch}.png")
        # plt.show()


class SMOTETsne:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.X_train = None
        self.y_train = None
        self.X_real = None
        self.y_real = None
        self._load_dataset()

    def _load_dataset(self):
        dataset = DatasetsLoader.Dataset(dataset_name=self.dataset_name)
        X, y = dataset.GetDataset()
        X_train, y_train, _, _ = split_dataset_random(X, y)
        self.X_train = X_train
        self.y_train = y_train
        self._set_real_data()

    def _set_real_data(self):
        self.X_real, self.y_real = GANs_two_class_real_data(self.X_train, self.y_train)

    def draw_and_save(self):
        X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(self.X_train, self.y_train)
        X_train_SMOTE_gen = X_train_SMOTE[self.X_train.shape[0]:]
        X_train_SMOTE_sel = resample(X_train_SMOTE_gen, n_samples=self.X_real.shape[0])

        if self.X_real.shape[0] <= 150:
            perplexity = self.X_real.shape[0] / 2
        else:
            perplexity = 100

        real_data = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=perplexity).fit_transform(self.X_real)
        generated_data = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=perplexity).fit_transform(X_train_SMOTE_sel)

        print(trustworthiness(self.X_real, real_data, n_neighbors=5))
        print(trustworthiness(X_train_SMOTE_sel, generated_data, n_neighbors=5))

        plt.figure(figsize=(10, 8))
        plt.scatter(real_data[:, 0], real_data[:, 1], label='Original Data', alpha=0.6, c='blue')
        plt.scatter(generated_data[:, 0], generated_data[:, 1], label='Generated Data', alpha=0.6, c='red')
        plt.legend()
        plt.title(f"t-SNE visualization of {self.dataset_name}-SMOTE")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.savefig(f"./result/{self.dataset_name}/t-SNE_SMOTE.png")
        # plt.show()


class CTGANTsne(SMOTETsne):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.CTGAN = None

    def fit(self):
        self.CTGAN = CTGAN(batch_size=500, epochs=100, cuda=True, verbose=True)
        X_train_fraud = self.X_train[self.y_train == 1]
        self.CTGAN.fit(X_train_fraud)

    def draw_and_save(self):
        if self.CTGAN is None:
            raise ModelNotFitException

        # num_samples = np.count_nonzero(self.y_train == 0) - np.count_nonzero(self.y_train == 1)
        num_samples = np.count_nonzero(self.y_train == 1)
        synthetic_data_x = self.CTGAN.sample(num_samples)
        synthetic_data_y = np.ones(num_samples)
        X_train_ctgan = np.concatenate([synthetic_data_x, self.X_train], axis=0)
        y_train_ctgan = np.concatenate([synthetic_data_y, self.y_train], axis=0)

        if self.X_real.shape[0] <= 150:
            perplexity = self.X_real.shape[0] / 2
        else:
            perplexity = 100

        real_data = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=perplexity).fit_transform(self.X_real)
        generated_data = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=perplexity).fit_transform(X_train_ctgan)

        print(trustworthiness(self.X_real, real_data, n_neighbors=5))
        print(trustworthiness(X_train_ctgan, generated_data, n_neighbors=5))

        plt.figure(figsize=(10, 8))
        plt.scatter(real_data[:, 0], real_data[:, 1], label='Original Data', alpha=0.6, c='blue')
        plt.scatter(generated_data[:, 0], generated_data[:, 1], label='Generated Data', alpha=0.6, c='red')
        plt.legend()
        plt.title(f"t-SNE visualization of {self.dataset_name}-CTGAN")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.savefig(f"./result/{self.dataset_name}/t-SNE_CTGAN.png")
        # plt.show()


class Imgs:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def draw_and_save(self):
        image1 = Image.open(f"./result/{self.dataset_name}/t-SNE_10.png")
        image2 = Image.open(f"./result/{self.dataset_name}/t-SNE_50.png")
        image3 = Image.open(f"./result/{self.dataset_name}/t-SNE_100.png")
        image4 = Image.open(f"./result/{self.dataset_name}/t-SNE_149.png")
        image5 = Image.open(f"./result/{self.dataset_name}/t-SNE_CTGAN.png")
        image6 = Image.open(f"./result/{self.dataset_name}/t-SNE_SMOTE.png")

        width, height = image1.size

        combined_image = Image.new('RGB', (width * 3, height * 2))

        combined_image.paste(image1, (0, 0))
        combined_image.paste(image2, (width, 0))
        combined_image.paste(image3, (width * 2, 0))
        combined_image.paste(image4, (0, height))
        combined_image.paste(image5, (width, height))
        combined_image.paste(image6, (width * 2, height))

        combined_image.save(f"./result/{self.dataset_name}/combined.png")


if __name__ == "__main__":
    # for i in DatasetsLoader.Datasets_list:
    for i in ['SouthGermanCredit']:
        print("Start training for dataset: ", i)

        gan_obj = GANTsne(dataset_name=i, device=get_default_device(force_skip_mps=False))
        gan_obj.fit()
        gan_obj.draw_and_save(epoch=10)
        gan_obj.draw_and_save(epoch=50)
        gan_obj.draw_and_save(epoch=100)
        gan_obj.draw_and_save(epoch=149)

        smote_obj = SMOTETsne(dataset_name=i)
        smote_obj.draw_and_save()

        ctgan_obj = CTGANTsne(dataset_name=i)
        ctgan_obj.fit()
        ctgan_obj.draw_and_save()

        imgs_obj = Imgs(dataset_name=i)
        imgs_obj.draw_and_save()

