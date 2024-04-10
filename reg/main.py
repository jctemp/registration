from models.transmorph import TransMorph
from models.transmorph_bayes import TransMorphBayes
from models.transmorph_bspline import TransMorphBspline

from configs.transmorph import CONFIGS as CONFIG_DEFAULT
from configs.transmorph_bayes import CONFIGS as CONFIG_BAYES
from configs.transmorph_bspline import CONFIGS as CONFIG_BSPLINE

from metrics import *
from model import TransMorphModule
from dataset import LungDataModule

import argparse
import sys

import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


CONFIG_TM = {}
CONFIG_TM.update(CONFIG_DEFAULT)
CONFIG_TM.update(CONFIG_BAYES)
CONFIG_TM.update(CONFIG_BSPLINE)

CONFIGS_IMAGE_LOSS = {
    "mse": MSE(),
    "ncc": NCC(spatial_dims=2),
    "gmi": GMI(),
}

CONFIGS_FLOW_LOSS = {
    "gl": GL(penalty="l2"),
    "bel": BEL(normalize=True),
}

CONFIGS_OPTIMIZER = {
    "adam": torch.optim.Adam
}


def reg_train(args):
    # Hyper Parameters
    optimizer_name = str.lower(args.optimizer)
    lr = float(args.lr)
    series_reg = True if str.lower(args.reg_type) == "series" else False
    target_type = str.lower(args.target_type)
    max_epoch = int(args.max_epoch)
    series_len = int(args.series_len)

    model_name = args.model_name
    image_loss, image_loss_weight = str.split(str.lower(args.image_loss), ":")
    flow_loss, flow_loss_weight = str.split(str.lower(args.flow_loss), ":")

    # Prepare training
    criterion_image = (CONFIGS_IMAGE_LOSS[image_loss], float(image_loss_weight))
    criterion_flow = (CONFIGS_FLOW_LOSS[flow_loss], float(flow_loss_weight))
    optimizer = CONFIGS_OPTIMIZER[optimizer_name]

    # Model
    config = CONFIG_TM[model_name]
    config.img_size = (*config.img_size[:-1], series_len)
    config.series_reg = series_reg

    print(config)

    if "bayes" in model_name:
        net = TransMorphBayes(config)
    elif "bspline" in model_name:
        net = TransMorphBspline(config)
    else:
        net = TransMorph(config)

    model = TransMorphModule(
        net=net,
        optimizer=optimizer,
        lr=lr,
        criterion_image=criterion_image,
        criterion_flow=criterion_flow,
        target_type=target_type,
    )

    # Trainer
    accelerator = "auto" if args.devices is None else "gpu"
    devices = "auto" if args.devices is None else args.devices
    batch_size = 1
    trainer_logger = None

    if args.log:
        logger = WandbLogger(project="lung-registration-transmorph")
        logger.experiment.config["optimizer"] = optimizer_name
        logger.experiment.config["lr"] = str(lr)
        logger.experiment.config["reg_type"] = series_reg
        logger.experiment.config["target_type"] = target_type
        logger.experiment.config["max_epoch"] = max_epoch
        logger.experiment.config["series_len"] = series_len
        logger.experiment.config["model_name"] = model_name
        logger.experiment.config["image_loss"] = image_loss
        logger.experiment.config["flow_loss"] = flow_loss
        trainer_logger = [logger]

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="model_weights/",
        filename=f"{model_name}-{image_loss}-{flow_loss}-{optimizer_name}-{str(lr)}-{series_reg}-{target_type}-"
                 f"{max_epoch}-{series_len}-{max_epoch}-"
                 "{epoch}-{val_loss:.8f}",
        save_top_k=5,
    )

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        deterministic=False,
        benchmark=False,
        logger=trainer_logger,
        callbacks=[checkpoint_callback],
        precision="32",
    )

    num_workers = 1 if args.num_workers is None else args.num_workers
    ckpt_path = None if args.path_to_ckpt is None else args.path_to_ckpt

    datamodule = LungDataModule(batch_size=batch_size, num_workers=num_workers, pin_memory=True, series_len=series_len)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def reg_test(args):
    # Hyper Parameters
    model_name = args.model_name
    image_loss, image_loss_weight = str.split(args.image_loss, ":")
    flow_loss, flow_loss_weight = str.split(args.flow_loss, ":")
    criterion_image = (CONFIGS_IMAGE_LOSS[image_loss], float(image_loss_weight))
    criterion_flow = (CONFIGS_FLOW_LOSS[flow_loss], float(flow_loss_weight))

    # Model
    config = CONFIG_TM[model_name]
    if "bayes" in model_name:
        net = TransMorphBayes(config)
    elif "bspline" in model_name:
        net = TransMorphBspline(config)
    else:
        net = TransMorph(config)
    model = TransMorphModule.load_from_checkpoint(
        args.path_to_ckpt,
        strict=False,
        net=net,
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
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


def reg_pred(args):
    # TODO: Load input file, make prediction and output to specified directory
    raise NotImplementedError("Prediction is not implemented yet")


def main():
    parser = argparse.ArgumentParser(description="CLI for Training, Testing, and Prediction")
    # parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--log", action="store_true", help="Log to wandb")
    parser.add_argument("--devices", type=int, help="The number of available CUDA devices")
    parser.add_argument("--num_workers", type=int, help="The number of workers")
    parser.add_argument("--path_to_ckpt", help="Path to checkpoint file")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--optimizer", default="adam", help="Optimizer for the training")
    train_parser.add_argument("--lr", default="1e-4", help="Learning rate")
    train_parser.add_argument("--reg_type", default="volume", help="Volume or time series (volume, series)")
    train_parser.add_argument("--target_type", default="last", help="Volume or time series (last, mean, group)")
    train_parser.add_argument("--max_epoch", default=100, help="The maximum number of epochs")
    train_parser.add_argument("--series_len", default=192, help="The length of the series, e.g. 192")
    train_parser.add_argument("model_name", help="The name of the model")
    train_parser.add_argument("image_loss", help="The image loss function, e.g. mse:1")
    train_parser.add_argument("flow_loss", help="The flow loss function, e.g. gl:1")

    test_parser = subparsers.add_parser("test", help="Test the model")
    test_parser.add_argument("--target_type", default="last", help="Volume or time series (last, mean, group)")
    test_parser.add_argument("model_name", help="The name of the model")
    test_parser.add_argument("image_loss", help="The image loss function, e.g. mse:1")
    test_parser.add_argument("flow_loss", help="The flow loss function, e.g. gl:1")

    pred_parser = subparsers.add_parser("pred", help="Make predictions")
    pred_parser.add_argument("model_name", help="The name of the model")

    args = parser.parse_args()

    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    if args.command == "train":
        reg_train(args)
    elif args.command == "test":
        reg_test(args)
    elif args.command == "pred":
        reg_pred(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main())
