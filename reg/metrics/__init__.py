from .grad_loss import Grad as GL
from .jacobian_det import jacobian_det

# from .ncc_loss import *

from torch.nn import MSELoss as MSE
from monai.losses import LocalNormalizedCrossCorrelationLoss as NCC
from monai.losses import GlobalMutualInformationLoss as GMI
from monai.losses import BendingEnergyLoss as BEL

__all__ = [
    "jacobian_det",
    # Image (dis)similarity, input = (pred, target)
    "MSE",
    "NCC",
    "GMI",
    # Regulisation 
    "GL",
    "BEL",
]
