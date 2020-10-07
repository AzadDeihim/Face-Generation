import torch.nn as nn
import torch
from collections import deque

########################################################################################################################
# Historical Averaging Regularisation Term
########################################################################################################################
"""
https://torchgan.readthedocs.io/en/latest/modules/losses.html#historicalaveragegeneratorloss

https://torchgan.readthedocs.io/en/latest/_modules/torchgan/losses/historical.html#HistoricalAverageGeneratorLoss
"""

# TODO: change to "nn.Module" child class?
class HistoricalAveraging(nn.Module):
    """
    NB: separate instances required for netG and netD.
    """

    def __init__(self, reg_param=0.1):
        super().__init__()
        self.reg_param = reg_param
        self.iter_count = 0
        self.params_totals = []

    def forward(self, net):
        if self.iter_count == 0:
            for p in net.parameters():         # TODO: "for i, p in enumerate(netG.parameters()):"
                param = p.detach().clone()
                self.params_totals.append(param)
            loss_term = 0.             # (theta - theta)**2. = 0.
        else:
            loss_term = 0.
            for i, p in enumerate(net.parameters()):       # TODO: replace with "range(len(self.sum_parameters))"?
                loss_term += (p - (self.params_totals[i].detach() / self.iter_count)).sum().pow(2.)  # TODO: check ".pow(2.)"
                self.params_totals[i] += p.detach().clone()    # TODO: inplace operation?!
        loss_term *= self.reg_param      # TODO: inplace operation?
        self.iter_count += 1
        return loss_term
