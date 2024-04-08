import os, glob, torch
import numpy as np
import scipy.io as spio
import pytorch_lightning as pl
import torchio as tio

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

def reader(path, max=192):
    dcm = spio.loadmat(path)["dcm"][:max]
    data = np.transpose(dcm, (1, 2, 0))[None, :, :, :]# C, W, H, t
    return data

class LungDataset(Dataset):
    def __init__(self, train=True, val=False, split=(0.8, 0.1), seed=42):
        subjects = glob.glob("/media/agjvc_rad3/_TESTKOLLEKTIV/Daten/Daten/*/")
        
        total_len = len(subjects)
        train_len = int(total_len * split[0])
        val_len = int(total_len * split[1])
        test_len = total_len - (train_len + val_len)

        generator = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = random_split(subjects, [train_len, val_len, test_len], generator)

        sub_path = "Series*/dicoms.mat"
        if train:
            self.image_paths = [d for p in train_set for d in glob.glob(f"{p}/{sub_path}")]
        elif val:
            self.image_paths = [d for p in val_set for d in glob.glob(f"{p}/{sub_path}")]
        else:
            self.image_paths = [d for p in test_set for d in glob.glob(f"{p}/{sub_path}")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data = reader(self.image_paths[idx])
        ndat = normalize(data).astype(np.float32)
        return torch.from_numpy(ndat)

class LungDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=1, pin_memory=True):
        super().__init__()

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_date(self):
        # not needed as the data is not downloaded
        pass

    def setup(self, stage=None):
        self.train_set = LungDataset(train=True, val=False)
        self.val_set = LungDataset(train=False, val=True)
        self.test_set = LungDataset(train=False, val=False)

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