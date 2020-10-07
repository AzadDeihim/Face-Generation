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

config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\ndevice = {0}\n'.format(config.DEVICE))

config.NUM_EPOCHS = 100
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
config.TENSORBOARD = True

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
netG = DCGAN_net.Generator(config).to(config.DEVICE)
# print('{}\n'.format(netG))

# Instantiate discriminator
netD_params = {'num_d_feat': config.NUM_D_FEAT, 'num_channels': config.NUM_CHANNELS}
netD = DCGAN_net.Discriminator(config).to(config.DEVICE)
# print('{}\n'.format(netD))

# Load weights
netG.apply(DCGAN_net.weights_init)
netD.apply(DCGAN_net.weights_init)

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

# Create analytics data storage object.
hist = History(record_window=10, tensorboard=config.TENSORBOARD, save_name=config.SAVE_NAME)

# Put models in train mode.
netG.train()
netD.train()

########################################################################################################################
# Train loop
########################################################################################################################

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

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        # (1a) TRAIN WITH ALL-REAL BATCH
        netD.zero_grad()
        # Format batch
        real_batch = data[0].to(config.DEVICE)
        b_size = real_batch.size(0)
        labels = torch.empty(size=(b_size,)).fill_(REAL_LABEL).to(config.DEVICE)
        # Forward pass real batch through D
        output = netD(real_batch).view(-1)
        # Calculate loss on all-real batch
        lossD_real = loss_function(output, labels)
        # Calculate gradients for D in backward pass
        lossD_real.backward()
        D_x = output.mean().item()

        # (1b) TRAIN WITH ALL-FAKE BATCH
        # Generate batch of latent vectors
        noise = torch.randn((b_size, config.NUM_LATENT, 1, 1)).to(config.DEVICE)
        # Generate fake image batch with G
        fake_batch = netG(noise)
        labels.fill_(FAKE_LABEL)
        # Classify all fake batch with D
        # TODO: NB: .detach() required because don't what grads from forward pass through netG
        output = netD(fake_batch.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        lossD_fake = loss_function(output, labels)
        # Calculate the gradients for this batch
        lossD_fake.backward()

        # (1c) COMBINE LOSSES --> Add the gradients from the all-real and all-fake batches.
        lossD = lossD_real + lossD_fake
        # Update D
        optimizerD.step()
        D_G_z1 = output.mean().item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ############################
        netG.zero_grad()
        labels.fill_(REAL_LABEL)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake_batch).view(-1)
        # Calculate G's loss based on this output
        lossG = loss_function(output, labels)
        # Calculate gradients for G
        lossG.backward()
        # Update G
        optimizerG.step()
        D_G_z2 = output.mean().item()

        # Record analytics to calculate mean for epoch.
        epoch_lossG_total.append(lossG.item())
        epoch_lossD_total.append(lossD.item())
        epoch_lossD_real.append(lossD_real.item())
        epoch_lossD_fake.append(lossD_fake.item())
        epoch_D_x.append(D_x)
        epoch_D_G_z1.append(D_G_z1)
        epoch_D_G_z2.append(D_G_z2)

        # Record iteration analytics.
        hist.save_iter_lossG(iter_lossG=lossG.item())
        hist.save_iter_lossD_total(iter_lossD_total=lossD.item())
        hist.save_iter_lossD_real(iter_lossD_real=lossD_real.item())
        hist.save_iter_lossD_fake(iter_lossD_fake=lossD_fake.item())
        hist.save_iter_D_x(iter_D_x=D_x)
        hist.save_iter_D_G_z1(iter_D_G_z1=D_G_z1)
        hist.save_iter_D_G_z2(iter_D_G_z2=D_G_z2)

        # Print training stats.
        if i % 50 == 0:
            print(
                '\rEpoch [{0}/{1}] [{2}/{3}]\tLoss_D: {4:.4f}\tLoss_G: {5:.4f}\tD(x): {6:.4f}==>0.5\tD(G(z)): [{7:.4f}|{8:.4f}]==>0.5\tTime taken: {9}s'.format(
                    epoch, config.NUM_EPOCHS, i, len(dataloader), lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2,
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
    os.makedirs('./Results/{0}/test_images'.format(config.SAVE_NAME), exist_ok=True)
    netG.eval()
    with torch.no_grad():
        fake_batch = netG(TEST_NOISE).detach().cpu()
        fake_images = np.transpose(vutils.make_grid(fake_batch, padding=2, normalize=True), (1, 2, 0))
        plt.imsave(fname='./Results/{0}/Images/fake_images_epoch{1}.png'.format(config.SAVE_NAME, epoch),
                   arr=fake_images.numpy())
    netG.train()

    if epoch % 10 == 0 or epoch == config.NUM_EPOCHS:
        os.makedirs('./Results/{0}/Models'.format(config.SAVE_NAME), exist_ok=True)
        torch.save({'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    },
                   './Results/{0}/Models/epoch{1}.pth'.format(config.SAVE_NAME, epoch))

hist.save_history()
