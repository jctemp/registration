import torch

from .grad_loss import Grad2dLoss, Grad3dLoss
from .jacobian_det import jacobian_det
from .gncc_loss import GlobalNormalizedCrossCorrelationLoss

from torch.nn import MSELoss
from monai.losses import LocalNormalizedCrossCorrelationLoss
from monai.losses import GlobalMutualInformationLoss
from monai.losses import BendingEnergyLoss

CONFIGS_WAPRED_LOSS = {
    "mse": MSELoss(),
    "ncc": LocalNormalizedCrossCorrelationLoss(kernel_size=5, spatial_dims=2),
    "gmi": GlobalMutualInformationLoss(),
    "gncc": GlobalNormalizedCrossCorrelationLoss(),
    "zero": lambda x, y: torch.tensor(0, dtype=torch.float32)
}

CONFIGS_FLOW_LOSS = {
    "gl3d": Grad3dLoss(penalty="l2"),
    "gl2d": Grad2dLoss(penalty="l2"),
    "bel": BendingEnergyLoss(normalize=True),
    "zero": lambda x: torch.tensor(0, dtype=torch.float32)
}

__all__ = [
    "CONFIGS_WAPRED_LOSS",
    "CONFIGS_FLOW_LOSS",
    "jacobian_det",
]
