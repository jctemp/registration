from .grad_loss import Grad2d, Grad3d
from .jacobian_det import jacobian_det

from torch.nn import MSELoss
from monai.losses import LocalNormalizedCrossCorrelationLoss
from monai.losses import GlobalMutualInformationLoss
from monai.losses import BendingEnergyLoss
from monai.losses import SSIMLoss

CONFIGS_IMAGE_LOSS = {
    "mse": MSELoss(),
    "ncc": LocalNormalizedCrossCorrelationLoss(kernel_size=7, spatial_dims=2),
    "gmi": GlobalMutualInformationLoss(),
    "ssim": SSIMLoss(spatial_dims=2),
}

CONFIGS_FLOW_LOSS = {
    "gl3d": Grad3d(penalty="l2"),
    "gl2d": Grad2d(penalty="l2"),
    "bel": BendingEnergyLoss(normalize=True),
}

__all__ = [
    "CONFIGS_IMAGE_LOSS",
    "CONFIGS_FLOW_LOSS",
    "jacobian_det",
]
