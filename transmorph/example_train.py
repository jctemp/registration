import os, sys
from torchvision import transforms
from data import datasets, trans
import numpy as np
import glob
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import losses
import utils

batch_size = 1
atlas_dir = "./IXI_data/atlas.pkl"
train_dir = "./IXI_data/Train/"
val_dir = "./IXI_data/Val/"
weights = [1, 1]  # loss
save_dir = f"TransMorph_ncc_{weights[0]}_diffusion_{weights[1]}"
if not os.path.exists("experiments/" + save_dir):
    os.makedirs("experiments/" + save_dir)
if not os.path.exists("logs/" + save_dir):
    os.makedirs("logs/" + save_dir)
lr = 0.0001
epoch_start = 0
max_epoch = 500

train_composed = transforms.Compose(
    [
        trans.RandomFlip(0),
        trans.NumpyType((np.float32, np.float32)),
    ]
)

val_composed = transforms.Compose(
    [
        trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
        trans.NumpyType((np.float32, np.int16)),
    ]
)
train_set = datasets.IXIBrainDataset(
    glob.glob(train_dir + "*.pkl")[:10], atlas_dir, transforms=train_composed
)
val_set = datasets.IXIBrainInferDataset(
    glob.glob(val_dir + "*.pkl")[:5], atlas_dir, transforms=val_composed
)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_set, batch_size=1, shuffle=False, num_workers=4, drop_last=True
)

criterions = [losses.NCC_vxm(), losses.Grad3d(penalty="l2")]


class TransMorphModel(pl.LightningModule):
    def __init__(self, model, stn):
        super().__init__()
        self.model = model
        self.stn = stn
        self.eval_dsc = utils.AverageMeter()

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        x_in = torch.cat([x, y], dim=1)
        output = self.model(x_in)
        loss = 0
        loss_vals = []
        for n, loss_fn in enumerate(criterions):
            curr_loss = loss_fn(output[n], y) * weights[n]
            loss_vals.append(curr_loss)
            loss += curr_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        x_seg = batch[2]
        y_seg = batch[3]
        x_in = torch.cat([x, y], dim=1)
        output = self.model(x_in)
        def_out = self.stn([x_seg.float(), output[1]])
        dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
        self.eval_dsc.update(dsc.item(), x.size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer
    

import models.TransMorph as TransMorph
from models.TransMorph import CONFIGS as CONFIGS_TM

config = CONFIGS_TM["TransMorph-Tiny"]
model = TransMorph.TransMorph(config)
reg_model = utils.register_model(config.img_size, 'nearest')

plmodel = TransMorphModel(model, reg_model)
trainer = pl.Trainer(max_epochs=max_epoch)

trainer.fit(plmodel, train_loader, val_loader)
