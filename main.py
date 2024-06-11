from GAN import get_default_device, get_generator, Generator, Discriminator
import DatasetsLoader
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from myutil import split_dataset_random, GANs_two_class_real_data
import json
import os


class Tsne:
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
        return generator, discriminator

    def _pre_trained(self):
        try:
            with open(f"checkpoint/{self.dataset_name}/checkpoints.json", "r") as f:
                config = json.load(f)
            if self.epochs == config["epochs"] and self.batch_size == config["batch_size"] and self.lr == config["lr"]:
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
                config = {"epochs": self.epochs, "lr": self.lr, "batch_size": self.batch_size}
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
        plt.title(f"t-SNE visualization of {self.dataset_name}-{epoch}")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.savefig(f"./result/{self.dataset_name}/t-SNE_{epoch}.png")
        plt.show()


if __name__ == "__main__":
    for i in DatasetsLoader.Datasets_list:
        obj = Tsne(dataset_name=i, device=get_default_device(force_skip_mps=True))
        obj.fit()
        obj.draw_and_save(epoch=10)
        obj.draw_and_save(epoch=50)
        obj.draw_and_save(epoch=100)
        obj.draw_and_save(epoch=149)

