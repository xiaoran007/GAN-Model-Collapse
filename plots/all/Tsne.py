from plots import GANTsne, SMOTETsne, CTGANTsne
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
from ctgan import CTGAN
import numpy as np
from PIL import Image


class GANTsneAll(GANTsne):
    def __init__(self, dataset_name, device, debug=False):
        super().__init__(dataset_name, device)
        self.debug = debug

    def draw_and_save(self, epoch):
        assert epoch in range(self.epochs)
        if os.path.exists(f"./result/{self.dataset_name}"):
            pass
        else:
            os.mkdir(f"./result/{self.dataset_name}")
        generator, _ = self._load_checkpoint(epoch)
        generator.eval()

        n_samples = np.count_nonzero(self.y_train == 0) - np.count_nonzero(self.y_train == 1)

        GANs_noise = torch.randn(n_samples, (self.X_real.shape[1]), device=self.device)
        generated_data_x = generator(GANs_noise.float().to(self.device)).cpu().detach().numpy()
        generated_data_y = np.ones(n_samples)

        X_train_gen = np.concatenate([generated_data_x, self.X_train], axis=0)
        y_train_gen = np.concatenate([generated_data_y, self.y_train], axis=0)

        if self.X_real.shape[0] <= 75:
            perplexity = self.X_real.shape[0] / 2
        else:
            perplexity = 50

        X_embedded = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2, perplexity=perplexity).fit_transform(X_train_gen)

        print(trustworthiness(X_train_gen, X_embedded, n_neighbors=5))

        plt.figure(figsize=(10, 8))

        plt.scatter(X_embedded[y_train_gen == 0, 0], X_embedded[y_train_gen == 0, 1], color='blue', label='majority', alpha=0.6)

        plt.scatter(X_embedded[y_train_gen == 1, 0], X_embedded[y_train_gen == 1, 1], color='red', label='Minority', alpha=0.6)

        plt.legend()
        plt.title(f"t-SNE visualization of {self.dataset_name}-GAN {epoch+1}")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.savefig(f"./result/{self.dataset_name}/t-SNE-all_{epoch+1}.png")
        if self.debug:
            plt.show()


if __name__ == "__main__":
    os.chdir("../../")
    for i in ['CreditRisk']:
        print("Start training for dataset: ", i)

        gan_obj = GANTsneAll(dataset_name=i, device=get_default_device(force_skip_mps=False), debug=True)
        gan_obj.fit()
        gan_obj.draw_and_save(epoch=10)
        gan_obj.draw_and_save(epoch=50)
        gan_obj.draw_and_save(epoch=100)
        gan_obj.draw_and_save(epoch=149)



