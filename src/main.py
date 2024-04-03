import argparse
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

def train(args):
    config = CONFIGS_TM["TransMorph-Small"]
    tm_model = TransMorph.TransMorph(config)
    criterion_image = (nn.MSELoss(), 1)
    criterion_flow = (Grad3d(penalty="l2"), 1)
    optimizer = torch.optim.Adam
    lr = 1e-4
   
    model = TransMorphModel(
        tm_model = tm_model,
        optimizer = optimizer,
        lr = lr,
        criterion_image = criterion_image,
        criterion_flow = criterion_flow,
    )
    
    accelerator = "gpu" if args.devices > 0 else "auto"
    devices = args.devices if args.devices > 0 else "auto"
    max_epoch = args.max_epochs
    batch_size = 1
    
    wandb_logger = WandbLogger(project="lung-registration-transmorph")
    #wandb_logger.experiment.config["batch_size"] = batch_size
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="model_weights/",
        filename="TransMorph-Small-{epoch}-{val_loss:.2f}",
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
    wandb.finish()
    
    print(checkpoint_callback.best_model_path)

def test(args):
    config = CONFIGS_TM["TransMorph-Small"]
    tm_model = TransMorph.TransMorph(config)
    criterion_image = (nn.MSELoss(), 1)
    criterion_flow = (Grad3d(penalty="l2"), 1)
    optimizer = torch.optim.Adam
   
    model = TransMorphModel.load_from_checkpoint(
        args.path_to_ckpt, 
        strict=False, 
        tm_model = tm_model,
        criterion_image = criterion_image,
        criterion_flow = criterion_flow,
    )

    accelerator = "gpu" if args.devices > 0 else "auto"
    devices = args.devices if args.devices > 0 else "auto"
    batch_size = 1
    
    pl.seed_everything(42)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
    )
    
    datamodule = LungDataModule(batch_size=batch_size, num_workers=4, pin_memory=True)
    trainer.test(model, datamodule=datamodule)

def pred(args):
    pass

def main():
    parser = argparse.ArgumentParser(description="CLI for Training, Testing, and Prediction")
    subparsers = parser.add_subparsers(dest="command")  # "command" will store subcommand name
    
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--devices", type=int, default=1, help="The number of available CUDA devices")
    train_parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    
    test_parser = subparsers.add_parser("test", help="Test the model")
    test_parser.add_argument("--devices", type=int, default=1, help="The number of available CUDA devices")
    test_parser.add_argument("path_to_ckpt", type=str, help="Path to checkpoint file")
    
    pred_parser = subparsers.add_parser("pred", help="Make predictions")
    pred_parser.add_argument("--devices", type=int, default=1, help="The number of available CUDA devices")
    pred_parser.add_argument("path_to_ckpt", help="Path to checkpoint file")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    elif args.command == "pred":
        pass # pred(args)
    else:
        raise Exception("Error: invalid command")

if __name__ == "__main__":
    sys.exit(main())
