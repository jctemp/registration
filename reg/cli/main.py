import os

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll
import torch

from reg.cli.parser import create_parser
from reg.data import LungDataModule
from reg.wrapper import TransMorphModuleBuilder


def main():
    """
    Main function for the CLI.
    """
    if not torch.cuda.is_available():
        print("Require CUDA to train wrapper")
        raise RuntimeError("CUDA is not available")

    print("Prepare training")

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
        print(f"")
        for key, value in config.items():
            print(f"{key:<25} = {value}")
        print(f"")
        print("=" * 120)

        pl_loggers = []
        run = (
            f"network_{args.network}.criteria-warped_{args.criteria_warped}.criteria-flow_{args.criteria_flow}."
            f"reg-strategy_{args.registration_strategy}.reg-target_{args.registration_target}."
            f"reg-depth_{args.registration_depth}.ident-loss_{args.identity_loss}.optimizer_{args.optimizer}."
            f"learning-rate_{args.learning_rate:.0E}"
        )
        run_path = f"model_weights_v3/{run}"
        print(f"Model weights are found in {run_path}")

        if args.log:
            pl_loggers = [
                pll.WandbLogger(
                    save_dir="logs", project="lung-registration", config=config
                )
            ]

        checkpoint_callback = plc.ModelCheckpoint(
            monitor="val_loss",
            dirpath=run_path,
            filename="{val_loss:.8f}&{epoch}",
            save_top_k=5,
        )

        callbacks = [
            checkpoint_callback,
        ]

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            log_every_n_steps=1,
            deterministic=False,
            benchmark=False,
            logger=pl_loggers,
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
        if args.network != "transmorph-identity":
            print(f"{'=' * 5} Training {'=' * 105}")
            trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
        print(f"{'=' * 5} Testing {'=' * 106}")
        trainer.test(model, datamodule=datamodule, ckpt_path=args.resume)
        print("=" * 120)

    elif args.command == "pred":
        model, config = TransMorphModuleBuilder.from_ckpt(args.ckpt).build()
        print(f"{'=' * 5} Configuration summary {'=' * 92}")
        print(f"")
        for key, value in config.items():
            print(f"{key:<25} = {value}")
        print(f"")
        print("=" * 120)

        # TODO: add output path to arguments
        raise RuntimeError("Prediction not implemented.")
