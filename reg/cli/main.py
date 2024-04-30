import os

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll
import torch
import json

from reg.cli.parser import create_parser
from reg.data import LungDataModule
from reg.model import TransMorphModuleBuilder


def main():
    """
    Main function for the CLI.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    parser = create_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()

    if args.command == "train":
        builder = (
            TransMorphModuleBuilder.from_ckpt(args.resume)
            if args.resume
            else TransMorphModuleBuilder()
        )
        if args.resume is None:
            (
                builder.set_network(args.network)
                .set_criteria_warped(args.criteria_warped)
                .set_criteria_flow(args.criteria_flow)
                .set_registration_strategy(args.registration_strategy)
                .set_registration_target(args.registration_target)
                .set_registration_depth(args.registration_depth)
                .set_registration_stride(args.registration_stride)
                .set_identity_loss(args.identity_loss)
                .set_optimizer(args.optimizer)
                .set_learning_rate(args.learning_rate)
            )
        model, config = builder.build()

        print(f"{'=' * 5} Configuration summary {'=' * 92}")
        print("")
        for key, value in config.items():
            print(f"{key:<25} = {value}")
        print("")
        print("=" * 120)

        loggers = []
        run = (
            f"network_{args.network}.criteria-warped_{args.criteria_warped}.criteria-flow_{args.criteria_flow}."
            f"reg-strategy_{args.registration_strategy}.reg-target_{args.registration_target}."
            f"reg-depth_{args.registration_depth}.ident-loss_{args.identity_loss}.optimizer_{args.optimizer}."
            f"learning-rate_{args.learning_rate:.0E}"
        )
        run_path = f"model_weights_v3/{run}"

        if args.log:
            wandb_logger = pll.WandbLogger(
                save_dir="logs", project="lung-registration", config=config
            )
            csv_logger = pll.CSVLogger(save_dir="logs", name=f"lung-registration_{run}")
            loggers = [wandb_logger, csv_logger]

        checkpoint_callback = plc.ModelCheckpoint(
            monitor="val_loss",
            dirpath=run_path,
            filename="{val_loss:.8f}&{epoch}",
            save_top_k=5,
        )
        progress_bar_callback = plc.RichProgressBar()
        early_stopping_callback = plc.EarlyStopping("val_loss", patience=5)

        callbacks = [
            checkpoint_callback,
            progress_bar_callback,
            early_stopping_callback,
        ]

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            log_every_n_steps=1,
            deterministic=False,
            benchmark=False,
            logger=loggers,
            callbacks=callbacks,
            precision="32",
        )

        n_available_cores = len(os.sched_getaffinity(0)) - 1
        n_available_cores = 1 if n_available_cores == 0 else n_available_cores

        datamodule = LungDataModule(
            root_dir="/media/agjvc_rad3/_TESTKOLLEKTIV/Daten/Daten",
            max_series_length=128,
            split=(0.7, 0.1, 0.2),
            seed=42,
            pin_memory=True,
            num_workers=n_available_cores,
        )

        torch.set_float32_matmul_precision("high")
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
        trainer.test(model, datamodule=datamodule, ckpt_path=args.resume)

    elif args.command == "test":
        ...
    elif args.command == "pred":
        ...
