import torch.nn as nn
import torch

########################################################################################################################
# DCGAN
########################################################################################################################


# generator returns an 'image': object dimensionality batch_size x 3 x num_g_feat x num_d_feat
class DCGAN_Generator(nn.Module):
    def __init__(self, config):
        super(DCGAN_Generator, self).__init__()
        self.num_latent = config.NUM_LATENT
        self.num_g_feat = config.NUM_G_FEAT
        self.num_channels = config.NUM_CHANNELS

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.num_latent, self.num_g_feat * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.num_g_feat * 8),
            nn.ReLU(True),
            # state size. (num_g_feat * 8) x 4 x 4
            nn.ConvTranspose2d(self.num_g_feat * 8, self.num_g_feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_g_feat * 4),
            nn.ReLU(True),
            # state size. (num_g_feat * 4) x 8 x 8
            nn.ConvTranspose2d(self.num_g_feat * 4, self.num_g_feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_g_feat * 2),
            nn.ReLU(True),
            # state size. (num_g_feat * 2) x 16 x 16
            nn.ConvTranspose2d(self.num_g_feat * 2, self.num_g_feat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_g_feat),
            nn.ReLU(True),
            # state size. (num_g_feat) x 32 x 32
            nn.ConvTranspose2d(self.num_g_feat, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
 
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Discriminator takes an 'image': object dimensionality batch_size x 3 x H x W  
class DCGAN_Discriminator(nn.Module):
    def __init__(self, config):
        super(DCGAN_Discriminator, self).__init__()
        self.num_d_feat = config.NUM_D_FEAT
        self.num_channels = config.NUM_CHANNELS

        self.main = nn.Sequential(
            # input is (num_channels) x 64 x 64
            nn.Conv2d(self.num_channels, self.num_d_feat, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_d_feat) x 32 x 32
            nn.Conv2d(self.num_d_feat, self.num_d_feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_d_feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_d_feat * 2) x 16 x 16
            nn.Conv2d(self.num_d_feat * 2, self.num_d_feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_d_feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_d_feat * 4) x 8 x 8
            nn.Conv2d(self.num_d_feat * 4, self.num_d_feat * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_d_feat * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_d_feat * 8) x 4 x 4
            nn.Conv2d(self.num_d_feat * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        features = self.main(input)
        return features


# TODO: figure out how custom weight initialisation works
# Custom weights initialization called on netG and netD
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


########################################################################################################################
# Minibatch discrimination
########################################################################################################################
"""
https://gist.github.com/t-ae/732f78671643de97bbe2c46519972491

https://torchgan.readthedocs.io/en/latest/_modules/torchgan/layers/minibatchdiscrimination.html#MinibatchDiscrimination1d

1D
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims                  # TODO: "intermediate_features"
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        M = matrices.view(-1, self.out_features, self.kernel_dims).unsqueeze(0)      # 1xNxBxC

        # M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        # norm = torch.abs(M - M.permute(1, 0, 2, 3)).sum(3)  # NxNxB
        norm = (M - M.permute(1, 0, 2, 3)).abs().sum(3)
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        out = torch.cat([x, o_b], 1)
        return out










########################################################################################################################
# Spectral Normilisation DCGAN
########################################################################################################################
"""
https://github.com/niffler92/SNGAN/blob/master/spectralnorm.py
https://github.com/niffler92/SNGAN/blob/master/discriminator.py

https://github.com/isr-wang/SNGAN/blob/master/spectralnorm.py
https://github.com/isr-wang/SNGAN/blob/master/discriminator.py

https://torchgan.readthedocs.io/en/latest/_modules/torchgan/layers/spectralnorm.html#SpectralNorm2d
"""


class SpectralNorm(nn.Module):
    """
    Spectral normalization of weight with power iteration
    """
    def __init__(self, layer, args, niter=1):
        super().__init__()
        self.layer = layer
        self.niter = niter

        self.init_params(layer)

    @staticmethod
    def init_params(layer):
        """u, v, W_sn
        """
        w = layer.weight
        height = w.size(0)
        width = w.view(w.size(0), -1).shape[-1]

        layer.register_buffer('u', nn.Parameter(torch.randn(height, 1), requires_grad=False))
        layer.register_buffer('v', nn.Parameter(torch.randn(1, width), requires_grad=False))

    @staticmethod
    def update_params(layer, num_iter):
        u, v, w = layer.u, layer.v, layer.weight
        height = w.size(0)

        for i in range(num_iter):  # Power iteration
            v = w.view(height, -1).t() @ u
            v /= (v.norm(p=2) + 1e-12)
            u = w.view(height, -1) @ v
            u /= (u.norm(p=2) + 1e-12)

        # Spectral normalisation
        w.data /= (u.t() @ w.view(height, -1) @ v).data

    def forward(self, x):
        self.update_params(self.layer, self.niter)
        return self.layer(x)

# Discriminator takes an 'image': object dimensionality batch_size x 3 x H x W
class SNGAN_Discriminator(nn.Module):
    def __init__(self, config):
        super(SNGAN_Discriminator, self).__init__()
        self.num_d_feat = config.NUM_D_FEAT
        self.num_channels = config.NUM_CHANNELS

        self.main = nn.Sequential(
            # input is (num_channels) x 64 x 64
            nn.Conv2d(self.num_channels, self.num_d_feat, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_d_feat) x 32 x 32
            nn.Conv2d(self.num_d_feat, self.num_d_feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_d_feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_d_feat * 2) x 16 x 16
            nn.Conv2d(self.num_d_feat * 2, self.num_d_feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_d_feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_d_feat * 4) x 8 x 8
            nn.Conv2d(self.num_d_feat * 4, self.num_d_feat * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_d_feat * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_d_feat * 8) x 4 x 4
            nn.Conv2d(self.num_d_feat * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        features = self.main(input)
        return features






