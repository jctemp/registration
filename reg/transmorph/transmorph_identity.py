import torch
import torch.nn as nn


class TransMorphIdentity(nn.Module):
    def __init__(self, config):
        super(TransMorphIdentity, self).__init__()
        shape = (1, config.in_chans, *config.img_size)
        self.flow = torch.zeros(shape)

    def forward(self, x):
        return x, self.flow
