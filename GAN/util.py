from torch import nn
import torch


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




