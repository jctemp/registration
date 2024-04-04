import argparse
import sys

from transmorph.models.TransMorph import TransMorph
from transmorph.models.TransMorph import CONFIGS as CONFIGS_TM

from model import TransMorphModel
from metrics import Grad3d
from dataset import LungDataModule

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
import torch
import torch.nn as nn


def train(args):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    # Hyperparameters
    config_name = "TransMorph-Lrn"
    criterion_image = (nn.MSELoss(), 1)
    criterion_flow = (Grad3d(penalty="l2"), 1)
    optimizer = torch.optim.Adam
    lr = 1e-4
    max_epoch = 100

    # Model
    tm_model = TransMorph(CONFIGS_TM[config_name])
    model = TransMorphModel(
        tm_model=tm_model,
        optimizer=optimizer,
        lr=lr,
        criterion_image=criterion_image,
        criterion_flow=criterion_flow,
    )

    # Trainer
    accelerator = "auto" if args.devices is None else "gpu"
    devices = "auto" if args.devices is None else args.devices
    batch_size = 1

    wandb_logger = WandbLogger(project="lung-registration-transmorph")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="model_weights/",
        filename=f"{config_name}-" + "{epoch}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        deterministic=False,
        benchmark=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
    )

    num_workers = 1 if args.num_workers is None else args.num_workers
    ckpt_path = None if args.path_to_ckpt is None else args.path_to_ckpt

    datamodule = LungDataModule(
        batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def test(args):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    # Hyperparameters
    config_name = "TransMorph"
    criterion_image = (nn.MSELoss(), 1)
    criterion_flow = (Grad3d(penalty="l2"), 1)

    # Model
    tm_model = TransMorph(CONFIGS_TM[config_name])
    model = TransMorphModel.load_from_checkpoint(
        args.path_to_ckpt,
        strict=False,
        tm_model=tm_model,
        criterion_image=criterion_image,
        criterion_flow=criterion_flow,
    )

    # Trainer
    accelerator = "auto" if args.devices is None else "gpu"
    devices = "auto" if args.devices is None else args.devices
    batch_size = 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
    )

    num_workers = 1 if args.num_workers is None else args.num_workers
    ckpt_path = None if args.path_to_ckpt is None else args.path_to_ckpt

    datamodule = LungDataModule(
        batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    trainer.test(model, datamodule=datamodule)


def pred(args):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="CLI for Training, Testing, and Prediction"
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--devices", type=int, help="The number of available CUDA devices"
    )
    train_parser.add_argument(
        "--num_workers", type=int, help="The number of workers for dataloader"
    )
    train_parser.add_argument(
        "--strategy", type=str, help="The strategy for training the model"
    )
    train_parser.add_argument("--path_to_ckpt", help="Path to checkpoint file")

    test_parser = subparsers.add_parser("test", help="Test the model")
    test_parser.add_argument(
        "--devices", type=int, help="The number of available CUDA devices"
    )
    test_parser.add_argument("path_to_ckpt", type=str, help="Path to checkpoint file")

    pred_parser = subparsers.add_parser("pred", help="Make predictions")
    pred_parser.add_argument(
        "--devices", type=int, help="The number of available CUDA devices"
    )
    pred_parser.add_argument("path_to_ckpt", help="Path to checkpoint file")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    elif args.command == "pred":
        pass  # pred(args)
    else:
        raise Exception("Error: invalid command")


if __name__ == "__main__":
    sys.exit(main())
