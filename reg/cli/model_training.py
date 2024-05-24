import os
import string
import random

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll
from pathlib import Path

from reg.data import LungDataModule
from reg.transmorph.wrapper import TransMorphModule
from reg.cli.utils import deserialize_toml, serialize_data


def main(args):
    weights_directory = Path(f"model_weights")
    subdirectory = args.weight_directory
    if subdirectory is None:
        subdirectory = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    run_directory = weights_directory / subdirectory

    path = Path(args.file)
    training_type = path.suffix[1:].upper()

    config = None
    if training_type == "TOML":
        config = deserialize_toml(str(path.absolute()))

        model = TransMorphModule(
            network=config["network"],
            criteria_warped=config["criteria_warped"],
            criteria_flow=config["criteria_flow"],
            registration_target=config["registration_target"],
            registration_strategy=config["registration_strategy"],
            registration_depth=config["registration_depth"],
            registration_stride=config["registration_stride"],
            registration_sampling=config["registration_sampling"],
            identity_loss=config["identity_loss"],
            optimizer=config["optimizer"],
            learning_rate=config["learning_rate"],
        )

        print(f"{'=' * 5} Configuration summary {'=' * 92}")
        print(f"")
        for key, value in config.items():
            print(f"{key:<25} = {value}")
        print(f"")
        print("=" * 120)

        ident = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        name = f"{config['network']}-{subdirectory}-{ident}"
        wandb_logger = pll.WandbLogger(
            save_dir="logs", project="lung-registration", config=config, name=name
        )
        model.id = wandb_logger.version
        model.update_hyperparameters()

    elif training_type == "CKPT":
        model = TransMorphModule.load_from_checkpoint(str(path), strict=True)

        print(f"{'=' * 5} Configuration summary {'=' * 92}")
        print(f"")
        print(f"Model training will be continued. No summary available.")
        print(f"")
        print("=" * 120)

        wandb_logger = pll.WandbLogger(
            save_dir="logs", project="lung-registration", version=model.id
        )
    else:
        print(f"The training type {training_type} is unknown. Only .toml or .ckpt is allowed as suffix")
        exit(1)

    wandb_logger.watch(model)
    run_path = run_directory / wandb_logger.version
    print(f"Model weights are found in {run_path}")

    checkpoint_callback = plc.ModelCheckpoint(
        monitor="val_loss",
        dirpath=run_path,
        filename="{val_loss:.8f}&{epoch}",
        save_top_k=3,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=1,
        deterministic=False,
        benchmark=False,
        logger=[wandb_logger],
        callbacks=[checkpoint_callback],
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

    if model.network != "transmorph-identity":
        print(f"{'=' * 5} Training {'=' * 105}")
        trainer.fit(model, datamodule=datamodule)

    print(f"{'=' * 5} Testing {'=' * 106}")

    trainer.test(model, datamodule=datamodule)
    print("=" * 120)

    if config:
        serialize_data(config, str(run_path / "config.toml"))
