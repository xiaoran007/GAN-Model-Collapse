from torch import nn


class Generator(nn.Module):

    def __init__(self, z_dim, im_dim, hidden_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self._get_generator_block(z_dim, hidden_dim),
            self._get_generator_block(hidden_dim, hidden_dim * 2),
            self._get_generator_block(hidden_dim * 2, hidden_dim * 4),
            self._get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.generator(noise)

    def get_generator(self):
        return self.generator

    @staticmethod
    def _get_generator_block(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )


class Discriminator(nn.Module):
    def __init__(self, im_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            self._get_discriminator_block(im_dim, hidden_dim * 4),
            self._get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self._get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.discriminator(image)

    def get_disc(self):
        return self.discriminator

    @staticmethod
    def _get_discriminator_block(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

