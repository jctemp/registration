from dataset import LungDataModule
from model import TransMorphModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils import load_losses, load_model_params
import argparse
import pytorch_lightning as pl
import sys
import torch


def reg_train(args):
    # Hyper Parameters

    optimizer_name = str.lower(args.optimizer)
    lr = float(args.lr)
    target_type = str.lower(args.target_type)
    series_len = int(args.series_len)
    max_epoch = int(args.max_epoch)
    data_mod = args.data_mod if args.data_mod != "None" else None

    image_losses, flow_losses = load_losses(args.image_loss, args.flow_loss)

    net, criteria_image, criteria_flow, criterion_disp, optimizer = load_model_params(
        model_name=args.model_name,
        image_losses=image_losses,
        flow_losses=flow_losses,
        optimizer_name=optimizer_name,
        series_len=series_len)

    model = TransMorphModule(
        net=net,
        optimizer=optimizer,
        lr=lr,
        criteria_image=criteria_image,
        criteria_flow=criteria_flow,
        criterion_disp=criterion_disp,
        target_type=target_type,
    )

    # Trainer
    devices = int(args.devices) if args.devices is not None else 0
    accelerator = "cpu" if devices == 0 else "gpu"
    batch_size = 1
    trainer_logger = None

    if args.log:
        logger = WandbLogger(project="lung-registration-transmorph")
        logger.experiment.config["optimizer"] = optimizer_name
        logger.experiment.config["lr"] = str(lr)
        logger.experiment.config["target_type"] = target_type
        logger.experiment.config["max_epoch"] = max_epoch
        logger.experiment.config["series_len"] = series_len
        logger.experiment.config["model_name"] = args.model_name
        logger.experiment.config["image_loss"] = [loss for loss, weight in image_losses]
        logger.experiment.config["image_loss_weight"] = [weight for loss, weight in image_losses]
        logger.experiment.config["flow_loss"] = [loss for loss, weight in flow_losses]
        logger.experiment.config["flow_loss_weight"] = [weight for loss, weight in flow_losses]
        logger.experiment.config["data_mod"] = data_mod
        trainer_logger = [logger]

    image_loss_str = "&".join([f"{loss}={weight}" for loss, weight in image_losses])
    flow_loss_str = "&".join([f"{loss}={weight}" for loss, weight in flow_losses])
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"model_weights_v2/{args.model_name}-{image_loss_str}-{flow_loss_str}-"
                f"{optimizer_name}-{str(lr)}-{target_type}-{max_epoch}-{series_len}-{data_mod}",
        filename="{val_loss:.8f}&{epoch}",
        save_top_k=3,
    )

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        accelerator=accelerator,
        log_every_n_steps=1,
        deterministic=False,
        benchmark=False,
        logger=trainer_logger,
        callbacks=[checkpoint_callback],
        precision="32",
    )

    num_workers = 1 if args.num_workers is None else args.num_workers
    ckpt_path = None if args.path_to_ckpt is None else args.path_to_ckpt

    datamodule = LungDataModule(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                series_len=None, mod=data_mod)  # norm, std, None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def reg_test(args):
    # Hyper Parameters
    target_type = str.lower(args.target_type)
    series_len = int(args.series_len)

    image_losses, flow_losses = load_losses(args.image_loss, args.flow_loss)

    net, criteria_image, criteria_flow, criterion_disp, optimizer = load_model_params(
        model_name=args.model_name,
        image_losses=image_losses,
        flow_losses=flow_losses,
        optimizer_name=None,
        series_len=series_len)

    model = TransMorphModule.load_from_checkpoint(
        args.path_to_ckpt,
        strict=False,
        net=net,
        criteria_image=criteria_image,
        criteria_flow=criteria_flow,
        criterion_disp=criterion_disp,
        target_type=target_type,
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

    datamodule = LungDataModule(batch_size=batch_size, num_workers=num_workers, pin_memory=True, series_len=200)
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
    parser.add_argument("--data_mod", type=str, help="Type of modification to data (norm, std, None)")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--optimizer", default="adam", help="Optimizer for the training")
    train_parser.add_argument("--lr", default="1e-4", help="Learning rate")
    train_parser.add_argument("--target_type", default="last", help="Volume or time series (last, mean, group)")
    train_parser.add_argument("--max_epoch", default=100, help="The maximum number of epochs")
    train_parser.add_argument("--series_len", default=192, help="The length of the series, e.g. 192")
    train_parser.add_argument("model_name", help="The name of the model")
    train_parser.add_argument("image_loss", help="The image loss function, e.g. mse:1")
    train_parser.add_argument("flow_loss", help="The flow loss function, e.g. gl2d:1")

    test_parser = subparsers.add_parser("test", help="Test the model")
    test_parser.add_argument("--target_type", default="last", help="Volume or time series (last, mean, group)")
    test_parser.add_argument("--series_len", default=192, help="The length of the series, e.g. 192")
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
