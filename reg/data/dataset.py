import os
import glob
from typing import List, Tuple, Any

from torch.utils.data import Dataset, random_split
import torch

from reg.data.utils import read_mat_data


def flatten(subjects: List[Any]):
    """
    Flatten a list of lists of series into a single list of series.

    Args:
        subjects: List of lists of series.
    """
    return [series for subject in subjects for series in subject]


class LungDataset(Dataset):
    """
    Dataset class for the lung dataset.

    Args:
        root_dir: Root directory of the dataset. Can contain wildcards.
        max_series_length: Maximum number of images to load.
        split: Tuple of floats representing the train, validation and test split.
        train: If True, load the training set.
        val: If True, load the validation set.
        test: If True, load the test set.
        seed: Seed for the random split.
        transform: Transformation to apply to the data.
    """

    def __init__(
        self,
        root_dir: str,
        max_series_length: int = None,
        split: Tuple[float, float] | Tuple[float, float, float] = None,
        train: bool = False,
        val: bool = False,
        test: bool = False,
        seed: int = 42,
        transform: callable = None,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.max_series_length = max_series_length
        self.split = split
        self.seed = seed
        self.transform = transform

        assert os.path.exists(root_dir), f"{root_dir} does not exists"
        assert os.path.isdir(root_dir), f"{root_dir} is not a directory"

        subject_paths = glob.glob(glob.escape(os.path.join(root_dir, "*")))
        self.subject_series = [
            glob.glob(os.path.join(p, "Series*/dicoms.mat")) for p in subject_paths
        ]

        if split:
            generator = torch.Generator().manual_seed(seed)
            total_len = len(self.subject_series)
            train_len = int(total_len * split[0])
            train_set, val_set, test_set = None, None, None

            if len(split) == 2:
                test_len = total_len - train_len
                train_set, test_set = random_split(
                    self.subject_series, [train_len, test_len], generator  # type: ignore
                )
            elif len(split) == 3:
                val_len = int(total_len * split[1])
                test_len = total_len - train_len - val_len
                train_set, val_set, test_set = random_split(
                    self.subject_series, [train_len, val_len, test_len], generator  # type: ignore
                )
            else:
                raise ValueError("split failed: dim insufficient")

            if train:
                self.subject_series = train_set
            elif val:
                self.subject_series = val_set
            elif test:
                self.subject_series = test_set
            else:
                raise ValueError("split failed: insufficient")

        else:
            self.subject_series = flatten(self.subject_series)

    def __len__(self):
        """
        Returns the number of series in the dataset.
        """
        return len(self.subject_series)

    def __getitem__(self, idx):
        """
        Returns the series at the given index.
        """
        data = read_mat_data(self.subject_series[idx], end=self.max_series_length)
        if self.transform:
            data = self.transform(data)
        return data
