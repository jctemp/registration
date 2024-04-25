from models.transmorph import TransMorph
from configs import CONFIG_TM
from metrics import CONFIGS_IMAGE_LOSS, CONFIGS_FLOW_LOSS
from models.transmorph_bayes import TransMorphBayes
from models.transmorph_bspline import TransMorphBspline
import torch

CONFIGS_OPTIMIZER = {"adam": torch.optim.Adam}


def load_losses(image_losses, flow_losses, delimiter=":"):
    image_losses = [
        str.split(s, delimiter) for s in str.split(str.lower(image_losses), "&")
    ]
    flow_losses = [
        str.split(s, delimiter) for s in str.split(str.lower(flow_losses), "&")
    ]
    return image_losses, flow_losses


def load_model_params(
    model_name, image_losses, flow_losses, optimizer_name, series_len
):
    criteria_image = [
        (CONFIGS_IMAGE_LOSS[loss], float(weight)) for loss, weight in image_losses
    ]
    criteria_flow = [
        (CONFIGS_FLOW_LOSS[loss], float(weight)) for loss, weight in flow_losses
    ]
    criterion_disp = None
    optimizer = (
        CONFIGS_OPTIMIZER[optimizer_name] if optimizer_name is not None else None
    )

    config = CONFIG_TM[model_name]
    config.img_size = (*config.img_size[:-1], series_len)

    if "bayes" in model_name:
        net = TransMorphBayes(config)
    elif "bspline" in model_name:
        net = TransMorphBspline(config)
    else:
        net = TransMorph(config)

    return net, criteria_image, criteria_flow, criterion_disp, optimizer
