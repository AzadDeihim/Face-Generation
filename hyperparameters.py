import torch
import pickle
import datetime


class Hyperparameters(object):
    """
    Class for storing hyperparameters.
    """

    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters.
        self.NUM_EPOCHS = 100
        self.BATCH_SIZE = 128

        # Image parameters.
        # TODO: change this to single number
        self.IMAGE_SIZE = [64, 64]
        # Number of channels in training images
        self.NUM_CHANNELS = 3

        # Generator parameters.
        # Size of z latent vector (i.e. size of generator input)
        self.NUM_LATENT = 100
        # Size of feature maps in generator
        self.NUM_G_FEAT = 64

        # Discriminator parameters.
        # Size of feature maps in discriminator
        self.NUM_D_FEAT = 64

        # Analytics
        self.SAVE_NAME = None
        self.TENSORBOARD = False

    def save_hyperparameters(self):
        print('Hyperparameters saved.')
        with open('./Results/{0}/hyperparameters_{1}.pickle'.format(self.SAVE_NAME,
                                                                    datetime.datetime.now().strftime("%d%m%Y-%H%M%S")),
                  'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


