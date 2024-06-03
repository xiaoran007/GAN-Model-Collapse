from torch import nn
import torch
from tqdm.auto import tqdm
from .backend import to_device, DeviceDataLoader, empty_cache
from torch.utils.data import TensorDataset, DataLoader
from .model import Generator, Discriminator


class TrainGenerator:
    def __init__(self, latent_data, opt_g, generator, discriminator, device, minority_class):
        self.latent_data = latent_data
        self.opt_g = opt_g
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.minority_class = minority_class

    def train(self):
        self.opt_g.zero_grad()

        fake_data = self.generator(self.latent_data)

        preds = self.discriminator(fake_data)
        if self.minority_class == 0:
            targets = torch.zeros_like(preds, device=self.device)
        elif self.minority_class == 1:
            targets = torch.ones_like(preds, device=self.device)
        else:
            print("Error. Invalid minority class.")
            exit(-1)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(preds, targets)

        loss.backward()
        self.opt_g.step()

        return loss.item()


class TrainDiscriminator:
    def __init__(self, real_data, latent_data, opt_d, generator, discriminator, device, minority_class, majority_class):
        self.real_data = real_data
        self.latent_data = latent_data
        self.opt_d = opt_d
        self.discriminator = discriminator
        self.generator = generator
        self.device = device
        self.minority_class = minority_class
        self.majority_class = majority_class

    def train(self):
        self.opt_d.zero_grad()

        real_preds = self.discriminator(self.real_data)
        if self.minority_class == 0:
            real_targets = torch.zeros_like(real_preds, device=self.device)
        elif self.minority_class == 1:
            real_targets = torch.ones_like(real_preds, device=self.device)
        else:
            print("Error. Invalid minority class.")
            exit(-1)

        criterion = nn.BCEWithLogitsLoss()
        real_loss = criterion(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        fake_data = self.generator(self.latent_data)

        fake_preds = self.discriminator(fake_data)
        if self.majority_class == 0:
            fake_targets = torch.zeros_like(fake_preds, device=self.device)
        elif self.majority_class == 1:
            fake_targets = torch.ones_like(fake_preds, device=self.device)
        else:
            print("Error. Invalid majority class.")
            exit(-1)

        fake_loss = criterion(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        loss = (real_loss + fake_loss) / 2
        loss.backward()
        self.opt_d.step()
        return loss.item(), real_score, fake_score


class Fit:
    def __init__(self, epochs, lr, discriminator, generator, train_dl, device, minority_class, majority_class):
        self.epochs = epochs
        self.lr = lr
        self.discriminator = discriminator
        self.generator = generator
        self.train_dl = train_dl
        self.device = device
        self.minority_class = minority_class
        self.majority_class = majority_class

    def fit(self):
        empty_cache(self.device)

        losses_g = list()
        losses_d = list()
        real_scores = list()
        fake_scores = list()

        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        for epoch in range(self.epochs):
            loss_g, loss_d, real_score, fake_score = 0, 0, 0, 0
            for real_data, _ in tqdm(self.train_dl):
                latent_data = torch.randn(real_data.shape[0], real_data.shape[1], device=self.device)
                loss_d, real_score, fake_score = TrainDiscriminator(real_data, latent_data, opt_d, self.generator,
                                                                    self.discriminator,
                                                                    self.device, self.minority_class,
                                                                    self.majority_class).train()

                loss_g = TrainGenerator(latent_data, opt_g, self.generator, self.discriminator, self.device,
                                        self.minority_class).train()

            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch + 1, self.epochs, loss_g, loss_d, real_score, fake_score))

        return losses_g, losses_d, real_scores, fake_scores


def get_generator(X_train, X_real, y_real, device, lr, epochs, batch_size, minority_class, majority_class):

    my_dataset = TensorDataset(torch.Tensor(X_real), torch.Tensor(y_real))

    train_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DeviceDataLoader(train_dataloader, device)

    gen = Generator(X_train.shape[1], X_train.shape[1], 128)
    disc = Discriminator(X_train.shape[1], 128)

    generator = to_device(gen.generator, device)
    discriminator = to_device(disc.discriminator, device)

    Fit(epochs, lr, discriminator, generator, train_dataloader, device, minority_class, majority_class).fit()

    return generator
