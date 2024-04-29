from typing import List, Tuple

from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms

from reg.data.utils import ZNormalization, RescaleIntensity
from reg.data.dataset import LungDataset


class LungDataModule(pl.LightningDataModule):
    """
    Data module for the lung dataset.
    
    Args:
        root_dirs: List of root directories of the dataset. Can contain wildcards.
        max_series_length: Maximum number of images to load.
        split: Tuple of floats representing the train, validation and test split.
        seed: Seed for the random split.
        pin_memory: If True, pin memory for faster data transfer.
        num_workers: Number of workers for the data loader.
        
    """

    def __init__(
        self,
        root_dirs: List[str],
        max_series_length: int = None,
        split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
        pin_memory: bool | Tuple[bool, bool, bool] = True,
        num_workers: int = 1,
    ):
        super().__init__()

        self.root_dirs = root_dirs
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

        if stage == "fit" or stage is None:
            self.train_set = ConcatDataset(
                [
                    LungDataset(
                        root_dir=root_dir,
                        max_series_length=self.max_series_length,
                        split=self.split,
                        seed=self.seed,
                        transform=transforms,
                        train=True,
                        val=False,
                        test=False,
                    )
                    for root_dir in self.root_dirs
                ]
            )

        if stage == "validate" or stage is None:
            self.val_set = ConcatDataset(
                [
                    LungDataset(
                        root_dir=root_dir,
                        max_series_length=self.max_series_length,
                        split=self.split,
                        seed=self.seed,
                        transform=transforms,
                        train=False,
                        val=True,
                        test=False,
                    )
                    for root_dir in self.root_dirs
                ]
            )

        if stage == "test" or stage is None:
            self.test_set = ConcatDataset(
                [
                    LungDataset(
                        root_dir=root_dir,
                        max_series_length=self.max_series_length,
                        split=self.split,
                        seed=self.seed,
                        transform=transforms,
                        train=False,
                        val=False,
                        test=True,
                    )
                    for root_dir in self.root_dirs
                ]
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
