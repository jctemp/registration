from typing import Tuple

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.transforms

from reg.data.utils import ZNormalization, RescaleIntensity
from reg.data.dataset import LungDataset


class LungDataModule(pl.LightningDataModule):
    """
    Data module for the lung dataset.

    Args:
        root_dir: Root directory of the dataset. Can contain wildcards.
        max_series_length: Maximum number of images to load.
        split: Tuple of floats representing the train, validation and test split.
        seed: Seed for the random split.
        pin_memory: If True, pin memory for faster data transfer.
        num_workers: Number of workers for the data loader.

    """

    def __init__(
        self,
        root_dir: str,
        max_series_length: int = None,
        split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
        pin_memory: bool | Tuple[bool, bool, bool] = True,
        num_workers: int = 1,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.max_series_length = max_series_length
        self.split = split
        self.seed = seed
        self.pin_memory = pin_memory if pin_memory is tuple else [pin_memory] * 3
        self.num_workers = num_workers
        self.batch_size = 1

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), ZNormalization, RescaleIntensity(0, 1)]
        )

        self.train_set = LungDataset(
            root_dir=self.root_dir,
            max_series_length=self.max_series_length,
            split=self.split,
            seed=self.seed,
            transform=transforms,
            train=True,
            val=False,
            test=False,
        )

        self.val_set = LungDataset(
            root_dir=self.root_dir,
            max_series_length=self.max_series_length,
            split=self.split,
            seed=self.seed,
            transform=transforms,
            train=False,
            val=True,
            test=False,
        )

        self.test_set = LungDataset(
            root_dir=self.root_dir,
            max_series_length=self.max_series_length,
            split=self.split,
            seed=self.seed,
            transform=transforms,
            train=False,
            val=False,
            test=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
