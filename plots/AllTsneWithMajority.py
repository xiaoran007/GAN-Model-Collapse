from plots import GANTsne, SMOTETsne, CTGANTsne
import torch
from myutil import ModelNotFitException
import os
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt


class AllTsne:
    def __init__(self, dataset_name, device, epochs: list = None):
        if epochs is None:
            epochs = [9, 49, 99, 149]
        self.dataset_name = dataset_name
        self.device = device
        self.GANHelper = GANTsneHelper(self.dataset_name, self.device)
        self.SMOTEHelper = SMOTETsneHelper(self.dataset_name)
        self.CTGANHelper = CTGANTsneHelper(self.dataset_name)
        self._setHelper()
        self.epochs = epochs
        self.DataList = list()
        self.Real = self.GANHelper.X_real

    def _setHelper(self):
        self.GANHelper.fit()
        self.CTGANHelper.fit()

    # currently set only for epoch 9, 49, 99, 149
    def fit(self):
        real = self.Real
        gan_9 = self.GANHelper.generateData(epoch=9)
        gan_49 = self.GANHelper.generateData(epoch=49)
        gan_99 = self.GANHelper.generateData(epoch=99)
        gan_149 = self.GANHelper.generateData(epoch=149)
        smote = self.SMOTEHelper.generateData()
        ctgan = self.CTGANHelper.generateData()
        combined = np.vstack([real, gan_9, gan_49, gan_99, gan_149, smote, ctgan])

        if real.shape[0] <= 75:
            perplexity = real.shape[0] / 2
        else:
            perplexity = 50
        X_embedded = TSNE(n_components=2, random_state=42, verbose=1, angle=0.2,
                          perplexity=perplexity).fit_transform(combined)

        print(trustworthiness(combined, X_embedded, n_neighbors=5))

        end_real = len(real)
        end_gan9 = end_real + len(gan_9)
        end_gan49 = end_gan9 + len(gan_49)
        end_gan99 = end_gan49 + len(gan_99)
        end_gan149 = end_gan99 + len(gan_149)
        end_smote = end_gan149 + len(smote)
        end_ctgan = end_smote + len(ctgan)

        real_emb = X_embedded[:end_real]
        gan_9_emb = X_embedded[end_real:end_gan9]
        gan_49_emb = X_embedded[end_gan9:end_gan49]
        gan_99_emb = X_embedded[end_gan49:end_gan99]
        gan_149_emb = X_embedded[end_gan99:end_gan149]
        smote_emb = X_embedded[end_gan149:end_smote]
        ctgan_emb = X_embedded[end_smote:end_ctgan]

        self._plot(real_emb, gan_9_emb, "GAN", 9)
        self._plot(real_emb, gan_49_emb, "GAN", 49)
        self._plot(real_emb, gan_99_emb, "GAN", 99)
        self._plot(real_emb, gan_149_emb, "GAN", 149)
        self._plot(real_emb, smote_emb, "SMOTE")
        self._plot(real_emb, ctgan_emb, "CTGAN")

    def _plot(self, real, gen, model: str, epoch: int = 0):
        plt.figure(figsize=(10, 8))
        plt.scatter(real[:, 0], real[:, 1], label='Original Data',
                    alpha=0.6, c='blue')
        plt.scatter(gen[:, 0], gen[:, 1], label='Generated Data',
                    alpha=0.6, c='red')
        plt.legend()
        if model == "GAN":
            plt.title(f"t-SNE visualization of {self.dataset_name}-GAN {epoch + 1}")
            plt.xlabel("t-SNE feature 1")
            plt.ylabel("t-SNE feature 2")
            plt.savefig(f"./result/{self.dataset_name}/t-SNE_{epoch + 1}.png")
        elif model == "SMOTE":
            plt.legend()
            plt.title(f"t-SNE visualization of {self.dataset_name}-SMOTE")
            plt.xlabel("t-SNE feature 1")
            plt.ylabel("t-SNE feature 2")
            plt.savefig(f"./result/{self.dataset_name}/t-SNE_SMOTE.png")
        elif model == "CTGAN":
            plt.title(f"t-SNE visualization of {self.dataset_name}-CTGAN")
            plt.xlabel("t-SNE feature 1")
            plt.ylabel("t-SNE feature 2")
            plt.savefig(f"./result/{self.dataset_name}/t-SNE_CTGAN.png")


class GANTsneHelper(GANTsne):
    def __init__(self, dataset_name, device):
        super().__init__(dataset_name, device)

    def generateData(self, epoch: int):
        assert epoch in range(self.epochs)
        if os.path.exists(f"./result/{self.dataset_name}"):
            pass
        else:
            os.mkdir(f"./result/{self.dataset_name}")
        generator, _ = self._load_checkpoint(epoch)
        generator.eval()

        GANs_noise = torch.randn((self.X_real.shape[0]), (self.X_real.shape[1]), device=self.device)
        output = generator(GANs_noise.float().to(self.device)).cpu().detach().numpy()

        return output


class SMOTETsneHelper(SMOTETsne):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def generateData(self):
        X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(self.X_train, self.y_train)
        X_train_SMOTE_gen = X_train_SMOTE[self.X_train.shape[0]:]
        if len(X_train_SMOTE_gen) < len(self.X_real):
            X_train_SMOTE_sel = X_train_SMOTE_gen
        else:
            X_train_SMOTE_sel = X_train_SMOTE_gen[np.random.choice(X_train_SMOTE_gen.shape[0], size=self.X_real.shape[0], replace=False)]

        return X_train_SMOTE_sel


class CTGANTsneHelper(CTGANTsne):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def generateData(self):
        if self.CTGAN is None:
            raise ModelNotFitException

        # num_samples = np.count_nonzero(self.y_train == 0) - np.count_nonzero(self.y_train == 1)
        num_samples = self.X_real.shape[0]
        synthetic_data_x = self.CTGAN.sample(num_samples)
        X_train_ctgan = synthetic_data_x

        return X_train_ctgan
