import os, glob, torch
import numpy as np
import scipy.io as spio
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

class LungDataset(Dataset):
    def __init__(self):
        samples = glob.glob("/media/agjvc_rad3/_TESTKOLLEKTIV/Daten/Daten/*")
        series = [
            glob.glob(os.path.join(sample, "Series*/dicoms.mat")) for sample in samples
        ]
        slices = [s[int(len(s) / 2)] for s in series if len(s) > 0]
        self.slices = slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        path = self.slices[idx]
        dcm = spio.loadmat(path)["dcm"].astype(np.int32)[
            :192
        ]  # must be int32 as uint16 is not supported and 32bit required for safe upcast
        dcm = (dcm / 255).astype(np.float32)
        tensor = torch.from_numpy(dcm)
        x = torch.permute(tensor, (1, 2, 0))[None, :, :, :]  # C, W, H, t
        return x

class LungDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=1, pin_memory=True):
        super().__init__()

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_date(self):
        # not needed as the data is not downloaded
        pass

    def setup(self, stage=None):
        self.dataset = LungDataset()
        generator = torch.Generator().manual_seed(42)

        total_len = len(self.dataset)
        train_len = int(total_len * 0.8)
        val_len = int(total_len * 0.1)
        test_len = total_len - (train_len + val_len)

        self.train_data, self.val_data, self.test_data = random_split(
            dataset=self.dataset,
            lengths=[train_len, val_len, test_len],
            generator=generator,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )