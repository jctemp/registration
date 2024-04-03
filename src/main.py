from argparse import ArgumentParser
import sys

from transmorph.models import TransMorph
from transmorph.models.TransMorph import CONFIGS as CONFIGS_TM

from lightning_model import TransMorphModel
from loss_functions import Grad3d
from stn import register_model
from dataset import LungDataModule

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

def training(args):
    
    print("Set model parameters") 
    config = CONFIGS_TM["TransMorph-Small"]
    tm_model = TransMorph.TransMorph(config)
    stn_model = register_model(config.img_size)
    optimizer = torch.optim.Adam
    lr = 1e-4
    criterion_image = (nn.MSELoss(), 1)
    criterion_flow = (Grad3d(penalty="l2"), 1)
   
    print("Prepare model") 
    model = TransMorphModel(
        tm_model = tm_model,
        stn_model = stn_model,
        optimizer = optimizer,
        lr = lr,
        criterion_image = criterion_image,
        criterion_flow = criterion_flow,
    )
    
    print("Set training parameters") 
    max_epoch = args.max_epochs
    accelerator = "gpu"
    devices = args.devices
    batch_size = 1
    
    wandb_logger = WandbLogger(project='lung-registration-transmorph')
    #wandb_logger.experiment.config["batch_size"] = batch_size
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_weights/',
        filename='TransMorph-Small-{epoch}-{val_loss:.2f}',
    )
    
    pl.seed_everything(42)
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        deterministic=False,
        benchmark=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    torch.set_float32_matmul_precision("medium")
    
    datamodule = LungDataModule(batch_size=batch_size, num_workers=4, pin_memory=True)
   
    print("Start training") 
    trainer.fit(model, datamodule=datamodule)
    print("Finish training") 
    trainer.test(model, datamodule=datamodule)
    wandb.finish()
    
    print(checkpoint_callback.best_model_path)

def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--devices", type=int, default=1)
    parser.add_argument("-e", "--max_epochs", type=int, default=100)
    args = parser.parse_args()

    training(args)

if __name__ == '__main__':
    sys.exit(main())
