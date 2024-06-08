from scipy.io import loadmat
from numpy import transpose
import torch


class RescaleIntensity(object):
    """
    Rescale intensity values to a certain range.

    Args:
        out_min: the new minimal value for an image
        out_max: the new maximal value for an image
    """

    def __init__(self, out_min: float = 0, out_max: float = 1):
        self.out_min = out_min
        self.out_max = out_max

    def __call__(self, tensor: torch.Tensor):
        tensor = tensor.clone().float()

        in_min = tensor.min()
        in_max = tensor.max()
        in_range = in_max - in_min
        if in_range == 0:
            return tensor
        tensor -= in_min
        tensor /= in_range
        tensor *= self.out_max
        tensor += self.out_min
        return tensor


class ZNormalization(object):
    """
    Subtract mean and divide by standard deviation.
    """

    def __call__(self, tensor: torch.Tensor):
        tensor = tensor.clone().float()
        mean, std = tensor.mean(), tensor.std()
        if std == 0:
            return tensor
        tensor -= mean
        tensor /= std
        return tensor


def read_mat_data_w_meta(path, start=None, end=None):
    """
    Read a matlab file containing an image series (T, W, H) which is then converted to torch conventions.

    Args:
        path: filesystem path to matlab file
        start: inclusive index to load data (starting image)
        end: exclusive index to load data (ending image - 1)
    """
    mat = loadmat(path)
    dcm = mat["dcm"]

    if end is not None and dcm.shape[0] < end:
        end = None

    if start and end:
        dcm = dcm[start:end]
    elif start:
        dcm = dcm[start:]
    elif end:
        dcm = dcm[:end]
    data = transpose(dcm, (1, 2, 0))[None, :, :, :]  # C, W, H, t

    return data, mat


def read_mat_data(path, start=None, end=None):
    """
    Read a matlab file containing an image series (T, W, H) which is then converted to torch conventions.

    Args:
        path: filesystem path to matlab file
        start: inclusive index to load data (starting image)
        end: exclusive index to load data (ending image - 1)
    """
    data, _ = read_mat_data_w_meta(path, start, end)
    return data
