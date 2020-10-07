import random
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.utils as vutils
from torch import optim
from torch.nn import BCELoss
import os

import DCGAN_net
from Celeba_dataset import CelebaDataset
from hyperparameters import Hyperparameters
from history import History

########################################################################################################################
# Set random seed
########################################################################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

########################################################################################################################
# Define hyperparameters
########################################################################################################################
config = Hyperparameters()

config.DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\ndevice = {0}\n'.format(config.DEVICE))

config.NUM_EPOCHS = 50
config.BATCH_SIZE = 128

config.IMAGE_SIZE = [64, 64]
# Number of channels in training images
config.NUM_CHANNELS = 3
# Size of z latent vector (i.e. size of generator input)
config.NUM_LATENT = 100
# Size of feature maps in generator
config.NUM_G_FEAT = 64
# Size of feature maps in discriminator
config.NUM_D_FEAT = 64

# Analytics
config.SAVE_NAME = 'test'
config.TENSORBOARD = False

########################################################################################################################
# Load dataset
########################################################################################################################

# Dataset
faces_dataset_params = {'dataset_dir': './Datasets/celeba',
                        'img_size': config.IMAGE_SIZE}
faces = CelebaDataset(**faces_dataset_params)

# Dataloader
dataloader_pars = {'batch_size': config.BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
dataloader = torch.utils.data.DataLoader(faces, **dataloader_pars)

########################################################################################################################
# Load models
########################################################################################################################

# Instantiate generator
netG_params = {'num_latent': config.NUM_LATENT, 'num_g_feat': config.NUM_G_FEAT, 'num_channels': config.NUM_CHANNELS}
netG = DCGAN_net.DCGAN_Generator(config).to(config.DEVICE)
# print('{}\n'.format(netG))

# Instantiate discriminator
netD_params = {'num_d_feat': config.NUM_D_FEAT, 'num_channels': config.NUM_CHANNELS}
netD = DCGAN_net.DCGAN_Discriminator(config).to(config.DEVICE)
# print('{}\n'.format(netD))

# Load weights
netG.apply(DCGAN_net.weights_init_normal)
netD.apply(DCGAN_net.weights_init_normal)

########################################################################################################################
# Training
########################################################################################################################

# Optimizers
optimizer_params = {'lr': 0.0002, 'weight_decay': 1e-3, 'betas': (0.5, 0.999)}
optimizerG = optim.Adam(netG.parameters(), **optimizer_params)
optimizerD = optim.Adam(netD.parameters(), **optimizer_params)

# Loss function
# loss = BCEWithLogitsLoss()
loss_function = BCELoss().to(config.DEVICE)

# Define real and fake labels.
REAL_LABEL = 1
FAKE_LABEL = 0

# Create batch of latent vectors that we will use to visualise the progression of the generator.
TEST_NOISE = torch.randn((64, config.NUM_LATENT, 1, 1)).to(config.DEVICE)

torch.save(TEST_NOISE, './Datasets/TEST_NOISE.pt')


