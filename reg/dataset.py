import glob
import torch
import numpy as np
import pytorch_lightning as pl
import scipy.io as spio
from torch.utils.data import Dataset, DataLoader, random_split


def normalize(image):
    image_min = image.min()
    image_max = image.max()
    normalized = (image - image_min) / (image_max - image_min)
    return normalized


def standardize(image):
    image_mean = image.mean()
    image_std = image.std()
    normalized = (image - image_mean) / image_std
    return normalized


def reader(path, start=None, end=None, mat=False):
    mat = spio.loadmat(path)
    dcm = mat["dcm"]
    if start and end:
        dcm = dcm[start:end]
    elif start:
        dcm = dcm[start:]
    elif end:
        dcm = dcm[:end]
    data = np.transpose(dcm, (1, 2, 0))[None, :, :, :]  # C, W, H, t

    if mat:
        return data, mat
    return data


class LungDataset(Dataset):
    def __init__(
        self,
        train=True,
        val=False,
        split=(0.8, 0.1),
        seed=42,
        series_len=None,
        mod="norm",
    ):
        assert (
            series_len is None or series_len % 8 == 0
        ), "series_len must be divisible by 8"
        assert not (train and val), "Either train or val must be True, not both"
        assert (
            mod == "norm" or mod == "std" or mod is None
        ), "mod can be 'norm', 'std' or None"

        self.mod = mod

        self.series_len = series_len
        subjects = glob.glob("/media/agjvc_rad3/_TESTKOLLEKTIV/Daten/Daten/*/")

        total_len = len(subjects)
        train_len = int(total_len * split[0])
        val_len = int(total_len * split[1])
        test_len = total_len - (train_len + val_len)

        generator = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = random_split(
            subjects, [train_len, val_len, test_len], generator
        )

        sub_path = "Series*/dicoms.mat"
        if train:
            self.image_paths = [
                d for p in train_set for d in glob.glob(f"{p}/{sub_path}")
            ]
        elif val:
            self.image_paths = [
                d for p in val_set for d in glob.glob(f"{p}/{sub_path}")
            ]
        else:
            self.image_paths = [
                d for p in test_set for d in glob.glob(f"{p}/{sub_path}")
            ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data = reader(self.image_paths[idx], end=self.series_len)

        ndat = data
        if self.mod == "std":
            ndat = standardize(data)
        if self.mod == "norm":
            ndat = normalize(data)

        return torch.from_numpy(ndat.astype(np.float32))


class LungDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        seed=42,
        split=(0.8, 0.1),
        series_len=None,
        mod="norm",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.seed = seed
        self.split = split
        self.series_len = series_len
        self.mod = mod

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_date(self):
        # not needed as the data is not downloaded
        pass

    def setup(self, stage=None):
        self.train_set = LungDataset(
            train=True,
            val=False,
            split=self.split,
            seed=self.seed,
            series_len=self.series_len,
            mod=self.mod,
        )

        self.val_set = LungDataset(
            train=False,
            val=True,
            split=self.split,
            seed=self.seed,
            series_len=None,
            mod=self.mod,
        )

        self.test_set = LungDataset(
            train=False,
            val=False,
            split=self.split,
            seed=self.seed,
            series_len=None,
            mod=self.mod,
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
