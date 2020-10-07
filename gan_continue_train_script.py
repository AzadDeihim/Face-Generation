import random
import time
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision.utils as vutils
from torch import optim
from torch.nn import BCELoss

from GAN import gan_net1
from Celeba_dataset import CelebaDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device = {0}\n '.format(DEVICE))

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

########################################################################################################################
# Define variables
########################################################################################################################

TRIAL_NUM = 2
LOAD_EPOCH = 40
print('Loading: Trial{0}, Epoch{1}'.format(TRIAL_NUM, LOAD_EPOCH))

# Load Trial info
with open('./Results/Trial{0}/trial_info.pickle'.format(TRIAL_NUM), 'rb') as handle:
    trial_info = pickle.load(handle)
NUM_EPOCHS = 100
BATCH_SIZE = trial_info['BATCH_SIZE']
IMAGE_SIZE = trial_info['IMAGE_SIZE']
# Number of channels in training images
NUM_CHANNELS = trial_info['NUM_CHANNELS']
# Size of z latent vector (i.e. size of generator input)
NUM_LATENT = trial_info['NUM_LATENT']
# Size of feature maps in generator
NUM_G_FEAT = trial_info['NUM_G_FEAT']
# Size of feature maps in discriminator
NUM_D_FEAT = trial_info['NUM_D_FEAT']

trial_info['NUM_EPOCHS'] = NUM_EPOCHS
with open('./Results/Trial{0}/trial_info.pickle'.format(TRIAL_NUM), 'wb') as handle:
    pickle.dump(trial_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
########################################################################################################################
# Load models and Optimisers
########################################################################################################################
checkpoint = torch.load('./Results/Trial{0}/Models/epoch{1}.pth'.format(TRIAL_NUM, LOAD_EPOCH))

# Instantiate generator
netG_params = {'num_latent': NUM_LATENT, 'num_g_feat': NUM_G_FEAT, 'num_channels': NUM_CHANNELS}
netG = gan_net1.DCGAN_Generator(**netG_params)
netG.load_state_dict(checkpoint['netG_state_dict'])
netG.to(DEVICE)
# print('{}\n'.format(netG))

# Instantiate discriminator
netD_params = {'num_d_feat': NUM_D_FEAT, 'num_channels': NUM_CHANNELS}
netD = gan_net1.DCGAN_Discriminator(**netD_params)
netD.load_state_dict(checkpoint['netD_state_dict'])
netD.to(DEVICE)
# print('{}\n'.format(netD))


# Optimizers
optimizer_params = {'lr': 1e-3, 'weight_decay': 1e-3, 'betas': (0.5, 0.999)}
optimizerG = optim.Adam(netG.parameters(), **optimizer_params)
optimizerD = optim.Adam(netD.parameters(), **optimizer_params)

optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])


########################################################################################################################
# Load dataset
########################################################################################################################

# Dataset
faces_dataset_params = {'img_dir': './Datasets/celeba/img_align_celeba',
                        'img_size': IMAGE_SIZE}
faces = CelebaDataset(**faces_dataset_params)

# Dataloader
dataloader_pars = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
dataloader = torch.utils.data.DataLoader(faces, **dataloader_pars)

########################################################################################################################
# Training
########################################################################################################################

# Loss function
# loss = BCEWithLogitsLoss()
loss_function = BCELoss().to(DEVICE)

# Establish convention for real and fake labels during training
REAL_LABEL = 1
FAKE_LABEL = 0

# Create batch of latent vectors that we will use to visualise the progression of the generator
TEST_NOISE = torch.randn((64, NUM_LATENT, 1, 1)).to(DEVICE)

# Load analytics
with open('./Results/Trial{0}/analytics.pickle'.format(TRIAL_NUM), 'rb') as handle:
    analytics = pickle.load(handle)
for key in analytics.keys():
    analytics[key] = analytics[key][:LOAD_EPOCH]


# Put models in train mode
netG.train()
netD.train()

print("\nStarting Training Loop...")
timer_start = time.time()
# For each epoch
for epoch in range(LOAD_EPOCH, NUM_EPOCHS + 1):

    # For each batch in the dataloader
    for i, data in enumerate(dataloader, start=0):
        G_losses = []
        D_losses = []
        list_D_x = []
        list_D_G_z1 = []
        list_D_G_z2 = []

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_batch = data[0].to(DEVICE)
        b_size = real_batch.size(0)
        labels = torch.empty(size=(b_size,)).fill_(REAL_LABEL).to(DEVICE)
        # Forward pass real batch through D
        output = netD(real_batch).view(-1)
        # Calculate loss on all-real batch
        errD_real = loss_function(output, labels)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn((b_size, NUM_LATENT, 1, 1)).to(DEVICE)
        # Generate fake image batch with G
        fake_batch = netG(noise)
        labels.fill_(FAKE_LABEL)
        # Classify all fake batch with D
        # TODO: NB: .detach() required because don't what grads from forward pass through netG
        output = netD(fake_batch.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = loss_function(output, labels)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        ## Combine losses
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(REAL_LABEL)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake_batch).view(-1)
        # Calculate G's loss based on this output
        errG = loss_function(output, labels)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print(
                '\rEpoch [{0}/{1}] [{2}/{3}]\tLoss_D: {4:.4f}\tLoss_G: {5:.4f}\tD(x): {6:.4f}==>0.5\tD(G(z)): [{7:.4f}|{8:.4f}]==>0.5\tTime taken: {9:.2f}s'.format(
                    epoch, NUM_EPOCHS, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,
                    time.time() - timer_start), end='')
            timer_start = time.time()

        # Record analytics to calculate mean
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        list_D_x.append(D_x)
        list_D_G_z1.append(D_G_z1)
        list_D_G_z2.append(D_G_z2)

    analytics['Mean_D_losses'].append(np.mean(D_losses))
    analytics['Mean_G_losses'].append(np.mean(G_losses))
    analytics['Mean_D_x'].append(np.mean(list_D_x))
    analytics['Mean_D_G_z1'].append(np.mean(list_D_G_z1))
    analytics['Mean_D_G_z2'].append(np.mean(list_D_G_z2))

    # Check how the generator is doing by saving G's output on TEST_NOISE
    netG.eval()
    with torch.no_grad():
        fake_batch = netG(TEST_NOISE).detach().cpu()
        fake_images = np.transpose(vutils.make_grid(fake_batch, padding=2, normalize=True), (1, 2, 0))
        plt.imsave(fname='./Results/Trial{0}/Images/fake_images_epoch{1}.png'.format(TRIAL_NUM, epoch),
                   arr=fake_images.numpy())
    netG.train()

    if epoch % 20 == 0 or epoch == NUM_EPOCHS:
        torch.save({'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    },
                   './Results/Trial{0}/Models/epoch{1}.pth'.format(TRIAL_NUM, epoch))

    with open('./Results/Trial{0}/analytics.pickle'.format(TRIAL_NUM), 'wb') as handle:
        pickle.dump(analytics, handle, protocol=pickle.HIGHEST_PROTOCOL)
