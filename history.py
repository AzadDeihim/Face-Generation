from collections import deque
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from torch.utils.tensorboard import SummaryWriter


class History(object):
    """
    Class for storing analytics.
    """

    def __init__(self, record_window, tensorboard, save_name):
        self.record_window = record_window
        self.tensorboard = tensorboard
        self.save_name = save_name

        self.create_directories()

        if self.tensorboard:
            self.writer = SummaryWriter(log_dir='./logs/{0}/{1}'.format(self.save_name, datetime.datetime.now().strftime("%d%m%Y-%H%M%S")))

        self.mean_epoch_lossG = []
        self.iter_lossG = []

        self.mean_epoch_lossD_total = []
        self.mean_epoch_lossD_real = []
        self.mean_epoch_lossD_fake = []
        self.mean_epoch_D_x = []
        self.mean_epoch_D_G_z1 = []
        self.mean_epoch_D_G_z2 = []
        self.iter_lossD_total = []
        self.iter_lossD_real = []
        self.iter_lossD_fake = []
        self.iter_D_x = []
        self.iter_D_G_z1 = []
        self.iter_D_G_z2 = []

        self.mean_epoch_reg_loss_netD = []
        self.mean_epoch_reg_loss_netG = []
        self.mean_epoch_lossG_total = []
        self.iter_reg_loss_netD = []
        self.iter_reg_loss_netG = []
        self.iter_lossG_total = []

    def create_directories(self):
        os.makedirs('./Results/{0}'.format(self.save_name), exist_ok=True)

    def save_mean_epoch_lossG(self, mean_epoch_lossG, epoch):
        self.mean_epoch_lossG.append(mean_epoch_lossG)

        if self.tensorboard:
            self.writer.add_scalar('Generator/mean_epoch_lossG', scalar_value=mean_epoch_lossG, global_step=epoch)

    def save_mean_epoch_lossD_real(self, mean_epoch_lossD_real, epoch):
        self.mean_epoch_lossD_real.append(mean_epoch_lossD_real)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/mean_epoch_lossD_real', scalar_value=mean_epoch_lossD_real, global_step=epoch)

    def save_mean_epoch_lossD_fake(self, mean_epoch_lossD_fake, epoch):
        self.mean_epoch_lossD_fake.append(mean_epoch_lossD_fake)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/mean_epoch_lossD_fake', scalar_value=mean_epoch_lossD_fake, global_step=epoch)

    def save_mean_epoch_lossD_total(self, mean_epoch_lossD_total, epoch):
        self.mean_epoch_lossD_total.append(mean_epoch_lossD_total)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/mean_epoch_lossD_total', scalar_value=mean_epoch_lossD_total, global_step=epoch)

    def save_mean_epoch_D_x(self, mean_epoch_D_x, epoch):
        self.mean_epoch_D_x.append(mean_epoch_D_x)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/mean_epoch_D_x', scalar_value=mean_epoch_D_x, global_step=epoch)

    def save_mean_epoch_D_G_z1(self, mean_epoch_D_G_z1, epoch):
        self.mean_epoch_D_G_z1.append(mean_epoch_D_G_z1)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/mean_epoch_D_G_z1', scalar_value=mean_epoch_D_G_z1, global_step=epoch)

    def save_mean_epoch_D_G_z2(self, mean_epoch_D_G_z2, epoch):
        self.mean_epoch_D_G_z2.append(mean_epoch_D_G_z2)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/mean_epoch_D_G_z2', scalar_value=mean_epoch_D_G_z2, global_step=epoch)

    def save_iter_lossG(self, iter_lossG):
        self.iter_lossG.append(iter_lossG)

        if self.tensorboard:
            self.writer.add_scalar('Generator/iter_lossG', scalar_value=iter_lossG, global_step=len(self.iter_lossG))

    def save_iter_lossD_total(self, iter_lossD_total):
        self.iter_lossD_total.append(iter_lossD_total)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/iter_lossD_total', scalar_value=iter_lossD_total, global_step=len(self.iter_lossD_total))

    def save_iter_lossD_real(self, iter_lossD_real):
        self.iter_lossD_real.append(iter_lossD_real)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/iter_lossD_real', scalar_value=iter_lossD_real, global_step=len(self.iter_lossD_real))

    def save_iter_lossD_fake(self, iter_lossD_fake):
        self.iter_lossD_fake.append(iter_lossD_fake)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/iter_lossD_fake', scalar_value=iter_lossD_fake, global_step=len(self.iter_lossD_fake))

    def save_iter_D_x(self, iter_D_x):
        self.iter_D_x.append(iter_D_x)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/iter_D_x', scalar_value=iter_D_x, global_step=len(self.iter_D_x))

    def save_iter_D_G_z1(self, iter_D_G_z1):
        self.iter_D_G_z1.append(iter_D_G_z1)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/iter_D_G_z1', scalar_value=iter_D_G_z1, global_step=len(self.iter_D_G_z1))

    def save_iter_D_G_z2(self, iter_D_G_z2):
        self.iter_D_G_z2.append(iter_D_G_z2)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/iter_D_G_z2', scalar_value=iter_D_G_z2, global_step=len(self.iter_D_G_z2))

    def save_mean_epoch_reg_loss_netD(self, mean_epoch_reg_loss_netD, epoch):
        self.mean_epoch_reg_loss_netD.append(mean_epoch_reg_loss_netD)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/mean_epoch_reg_loss_netD', scalar_value=mean_epoch_reg_loss_netD,
                                   global_step=epoch)

    def save_mean_epoch_reg_loss_netG(self, mean_epoch_reg_loss_netG, epoch):
        self.mean_epoch_reg_loss_netG.append(mean_epoch_reg_loss_netG)

        if self.tensorboard:
            self.writer.add_scalar('Generator/mean_epoch_reg_loss_netG', scalar_value=mean_epoch_reg_loss_netG,
                                   global_step=epoch)

    def save_mean_epoch_lossG_total(self, mean_epoch_lossG_total, epoch):
        self.mean_epoch_lossG_total.append(mean_epoch_lossG_total)

        if self.tensorboard:
            self.writer.add_scalar('Generator/mean_epoch_lossG_total', scalar_value=mean_epoch_lossG_total,
                                   global_step=epoch)

    def save_iter_reg_loss_netD(self, iter_reg_loss_netD):
        if len(self.iter_reg_loss_netD):
            self.iter_reg_loss_netD.append(iter_reg_loss_netD.item())
        else:
            self.iter_reg_loss_netD.append(iter_reg_loss_netD)

        if self.tensorboard:
            self.writer.add_scalar('Discriminator/iter_reg_loss_netD', scalar_value=iter_reg_loss_netD, global_step=len(self.iter_reg_loss_netD))

    def save_iter_reg_loss_netG(self, iter_reg_loss_netG):
        if len(self.iter_reg_loss_netG):
            self.iter_reg_loss_netG.append(iter_reg_loss_netG.item())
        else:
            self.iter_reg_loss_netG.append(iter_reg_loss_netG)

        if self.tensorboard:
            self.writer.add_scalar('Generator/iter_reg_loss_netG', scalar_value=iter_reg_loss_netG, global_step=len(self.iter_reg_loss_netG))

    def save_iter_lossG_total(self, iter_lossG_total):
        self.iter_lossG_total.append(iter_lossG_total)

        if self.tensorboard:
            self.writer.add_scalar('Generator/iter_lossG_total', scalar_value=iter_lossG_total, global_step=len(self.iter_lossG_total))

    def save_history(self):
        with open('./Results/{0}/history_{1}.pickle'.format(self.save_name, datetime.datetime.now().strftime("%d%m%Y-%H%M%S")), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


