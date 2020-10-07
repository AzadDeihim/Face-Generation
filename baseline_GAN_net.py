import torch.nn as nn
import torch

########################################################################################################################
# Baseline GAN
########################################################################################################################
"""
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
"""


class BaselineGenerator(nn.Module):
    # TODO: check image heigh and image_width correct way around
    def __init__(self, config):
        super().__init__()

        self.num_latent = config.NUM_LATENT
        self.num_g_feat = config.NUM_G_FEAT
        # Image information
        self.num_channels = config.NUM_CHANNELS     # = 3
        self.image_height = config.IMAGE_SIZE[0]
        self.image_width = config.IMAGE_SIZE[1]

        def linear_block(in_feat, out_feat, normalise):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalise:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*linear_block(self.num_latent, 128, normalise=False),
                                   *linear_block(128, 256, normalise=True),
                                   *linear_block(256, 512, normalise=True),
                                   *linear_block(512, 1024, normalise=True),
                                   *linear_block(1024,
                                                 self.num_channels * self.image_height * self.image_width,
                                                 normalise=True),
                                   nn.Tanh()
                                   )

    def forward(self, x):
        """
        input is latent vector
        """
        # TODO: check dimensions
        x = x.squeeze()
        x = self.model(x)
        # reshape to image shape
        x = x.view(x.size(0), self.num_channels, self.image_height, self.image_width)
        return x


class BaselineDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Image information
        self.num_channels = config.NUM_CHANNELS     # = 3
        self.image_height = config.IMAGE_SIZE[0]
        self.image_width = config.IMAGE_SIZE[1]

        def linear_block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat),
                      nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.model = nn.Sequential(*linear_block(self.num_channels * self.image_height * self.image_width, 512),
                                   *linear_block(512, 256),
                                   nn.Linear(256, 1),
                                   nn.Sigmoid()
                                   )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x
