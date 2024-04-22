from .grad_loss import Grad2d as gl2d, Grad3d as gl3d
from .jacobian_det import jacobian_det

from torch.nn import MSELoss as mse
from monai.losses import LocalNormalizedCrossCorrelationLoss as lncc
from monai.losses import GlobalMutualInformationLoss as gmi
from monai.losses import BendingEnergyLoss as bel
from monai.losses import SSIMLoss as ssim

CONFIGS_IMAGE_LOSS = {
    "mse": mse(),
    "ncc": lncc(kernel_size=7, spatial_dims=2),
    "gmi": gmi(),
    "ssim": ssim(spatial_dims=2),
}

CONFIGS_FLOW_LOSS = {
    "gl3d": gl3d(penalty="l2"),
    "gl2d": gl2d(penalty="l2"),
    "bel": bel(normalize=True),
}

__all__ = [
    "CONFIGS_IMAGE_LOSS",
    "CONFIGS_FLOW_LOSS",
    "jacobian_det",
]
