import random
from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.utils as vutils
from torch import optim
import os

import DCGAN_improved_net
from Celeba_dataset import CelebaDataset
from hyperparameters import Hyperparameters
from history import History
from loss_utils import HistoricalAveraging

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

config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
config.SAVE_NAME = 'improved_DCGAN_test'
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
netG = DCGAN_improved_net.DCGAN_Generator(config).to(config.DEVICE)
# print('{}\n'.format(netG))

# Instantiate discriminator
netD = DCGAN_improved_net.DCGAN_Discriminator(config).to(config.DEVICE)
# print('{}\n'.format(netD))

# Load weights
netG.apply(DCGAN_improved_net.weights_init_normal)
netD.apply(DCGAN_improved_net.weights_init_normal)

# Put models in train mode.
netG.train()
netD.train()

########################################################################################################################
# Training
########################################################################################################################

# Create analytics data storage object.
hist = History(record_window=10, tensorboard=config.TENSORBOARD, save_name=config.SAVE_NAME)

# Create batch of latent vectors that we will use to visualise the progression of the generator.
TEST_NOISE = torch.load(f='./Datasets/TEST_NOISE.pt').to(config.DEVICE)

# Test TEST_NOISE on untrained netG.
os.makedirs('./Results/{0}/test_images'.format(config.SAVE_NAME), exist_ok=True)
netG.eval()
with torch.no_grad():
    fake_batch = netG(TEST_NOISE).detach().cpu()
    fake_images = np.transpose(vutils.make_grid(fake_batch, padding=2, normalize=True, range=(-1, 1)), (1, 2, 0))
    plt.imsave(fname='./Results/{0}/test_images/fake_images_epoch{1}.png'.format(config.SAVE_NAME, 0),
               arr=fake_images.numpy())
netG.train()

# Optimizers --> two time-scale update rule used.
optimizerG_params = {'lr': 0.0001, 'weight_decay': 1e-3, 'betas': (0.5, 0.999)}
optimizerD_params = {'lr': 0.0004, 'weight_decay': 1e-3, 'betas': (0.5, 0.999)}
optimizerG = optim.Adam(netG.parameters(), **optimizerG_params)
optimizerD = optim.Adam(netD.parameters(), **optimizerD_params)

# Historical averaging classes.
netG_historical_averaging = HistoricalAveraging(reg_param=0.1)
netD_historical_averaging = HistoricalAveraging(reg_param=0.1)

########################################################################################################################
# Train loop
########################################################################################################################

# Save initial model.
os.makedirs('./Results/{0}/Models'.format(config.SAVE_NAME), exist_ok=True)
torch.save({'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            },
           './Results/{0}/Models/epoch{1}.pth'.format(config.SAVE_NAME, 0))

print("\nStarting Training Loop...")
start_timer = timer()
# For each epoch
for epoch in range(1, config.NUM_EPOCHS + 1):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, start=0):
        epoch_lossG_total = []
        epoch_lossD_total = []
        epoch_lossD_real = []
        epoch_lossD_fake = []
        epoch_D_x = []
        epoch_D_G_z1 = []
        epoch_D_G_z2 = []

        real_batch = data[0].to(config.DEVICE)
        b_size = real_batch.size(0)

        # One-sided label smoothing   # TODO: only smooth real_labels?
        real_labels = torch.empty(size=(b_size,)).uniform_(0.7, 1.2).to(config.DEVICE)
        fake_labels = torch.empty(size=(b_size,)).fill_(0).to(config.DEVICE)        # TODO: ".uniform_(0., 0.3)" for smoothing

        # Generate batch of latent vectors - standard normal distribution.
        noise = torch.randn((b_size, config.NUM_LATENT, 1, 1)).to(config.DEVICE)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        netD.zero_grad()
        # (1a) TRAIN netD WITH ALL-REAL BATCH
        output = netD(real_batch).view(-1)
        lossD_real = (-1. * real_labels * output.log()).mean()
        D_x = output.mean().item()

        # (1b) TRAIN netD WITH ALL-FAKE BATCH
        # Generate all-fake image batch with G
        fake_batch = netG(noise)
        # Classify all fake batch with D, but don't include gradients from forward prop through netG
        output = netD(fake_batch.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        lossD_fake = (-1. * (1. - fake_labels) * torch.log(1. - output)).mean()
        D_G_z1 = output.mean().item()

        reg_loss_term_netD = netD_historical_averaging(netD)
        lossD_total = lossD_real + lossD_fake + reg_loss_term_netD
        lossD_total.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ############################
        netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake_batch).view(-1)
        # Calculate G's loss based on this output
        lossG = (-1 * (1. - fake_labels) * output.log()).mean()
        D_G_z2 = output.mean().item()

        reg_loss_term_netG = netG_historical_averaging(netG)
        lossG_total = lossG + reg_loss_term_netG
        lossG_total.backward()
        optimizerG.step()

        # Record analytics to calculate mean for epoch.
        epoch_lossG_total.append(lossG.item())
        epoch_lossD_total.append(lossD_total.item())
        epoch_lossD_real.append(lossD_real.item())
        epoch_lossD_fake.append(lossD_fake.item())
        epoch_D_x.append(D_x)
        epoch_D_G_z1.append(D_G_z1)
        epoch_D_G_z2.append(D_G_z2)

        # Record iteration analytics.
        hist.save_iter_lossG(iter_lossG=lossG.item())
        hist.save_iter_lossD_total(iter_lossD_total=lossD_total.item())
        hist.save_iter_lossD_real(iter_lossD_real=lossD_real.item())
        hist.save_iter_lossD_fake(iter_lossD_fake=lossD_fake.item())
        hist.save_iter_D_x(iter_D_x=D_x)
        hist.save_iter_D_G_z1(iter_D_G_z1=D_G_z1)
        hist.save_iter_D_G_z2(iter_D_G_z2=D_G_z2)

        # Print training stats.
        if i % 50 == 0:
            print(
                '\rEpoch [{0}/{1}] [{2}/{3}]\tLoss_D: {4:.4f}\tLoss_G: {5:.4f}\tD(x): {6:.4f}==>0.5\tD(G(z)): [{7:.4f}|{8:.4f}]==>0.5\tTime taken: {9}s'.format(
                    epoch, config.NUM_EPOCHS, i, len(dataloader), lossD_total.item(), lossG.item(), D_x, D_G_z1, D_G_z2,
                    timedelta(seconds=int(timer() - start_timer))), end='')

    # Record epoch analytics.
    hist.save_mean_epoch_lossG(mean_epoch_lossG=np.mean(epoch_lossG_total), epoch=epoch)
    hist.save_mean_epoch_lossD_total(mean_epoch_lossD_total=np.mean(epoch_lossD_total), epoch=epoch)
    hist.save_mean_epoch_lossD_real(mean_epoch_lossD_real=np.mean(epoch_lossD_real), epoch=epoch)
    hist.save_mean_epoch_lossD_fake(mean_epoch_lossD_fake=np.mean(epoch_lossD_fake), epoch=epoch)
    hist.save_mean_epoch_D_x(mean_epoch_D_x=np.mean(epoch_D_x), epoch=epoch)
    hist.save_mean_epoch_D_G_z1(mean_epoch_D_G_z1=np.mean(epoch_D_G_z1), epoch=epoch)
    hist.save_mean_epoch_D_G_z2(mean_epoch_D_G_z2=np.mean(epoch_D_G_z2), epoch=epoch)

    # Evaluate generator's progress by saving netG's output on TEST_NOISE
    netG.eval()
    with torch.no_grad():
        fake_batch = netG(TEST_NOISE).detach().cpu()
        fake_images = np.transpose(vutils.make_grid(fake_batch, padding=2, normalize=True, range=(-1, 1)), (1, 2, 0))
        plt.imsave(fname='./Results/{0}/test_images/fake_images_epoch{1}.png'.format(config.SAVE_NAME, epoch),
                   arr=fake_images.numpy())
    netG.train()

    if epoch % 5 == 0 or epoch == config.NUM_EPOCHS:
        torch.save({'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    },
                   './Results/{0}/Models/epoch{1}.pth'.format(config.SAVE_NAME, epoch))

if config.TENSORBOARD:
    hist.writer.close()

hist.save_history()
