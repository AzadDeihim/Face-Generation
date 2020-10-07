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
