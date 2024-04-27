from pathlib2 import Path
import torch

from models.transmorph import TransMorph
from models.transmorph_bayes import TransMorphBayes
from models.transmorph_bspline import TransMorphBspline

from configs import CONFIG_TM
from metrics import CONFIGS_IMAGE_LOSS, CONFIGS_FLOW_LOSS
from model import TransMorphModule

CONFIGS_OPTIMIZER = {"adam": torch.optim.Adam}


def load_losses(image_losses, flow_losses, delimiter=":"):
    image_losses = [
        s.split(delimiter) for s in str.lower(image_losses).split("&")
    ]
    flow_losses = [
        s.split(delimiter) for s in str.lower(flow_losses).split("&")
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


def load_best_model(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    ckpt_file_names = sorted([c.name for c in ckpt_dir.glob("*.ckpt")], reverse=True)

    best_ckpt = ckpt_dir / ckpt_file_names[0]
    val_loss, epoch = ckpt_file_names[0].split("&")
    val_loss = val_loss.split("=")[1]
    epoch = epoch.split("=")[1].split(".")[0]

    ident = str(ckpt_dir.name).split("-")
    if len(ident) == 9:
        model_name, image_loss, flow_loss, optimizer_name, lr, target_type, max_epoch, series_len, data_mod = ident
    else:
        model_name, model_ver, image_loss, flow_loss, optimizer_name, lr, target_type, max_epoch, series_len, data_mod = ident
        model_name = f"{model_name}-{model_ver}"

    target_type = str.lower(target_type)
    series_len = int(series_len)
    max_epoch = int(max_epoch)

    image_losses, flow_losses = load_losses(image_loss, flow_loss, delimiter="=")

    net, criteria_image, criteria_flow, criterion_disp, optimizer = load_model_params(
        model_name=model_name,
        image_losses=image_losses,
        flow_losses=flow_losses,
        optimizer_name=None,
        series_len=series_len)

    model = TransMorphModule.load_from_checkpoint(
        str(best_ckpt),
        strict=False,
        net=net,
        criteria_image=criteria_image,
        criteria_flow=criteria_flow,
        criterion_disp=criterion_disp,
        target_type=target_type,
    )

    print("=" * 80)
    print(f"Best performing model")
    print("")
    print(f"    val_loss    : {val_loss}")
    print(f"    epoch       : {epoch}")
    print("")
    print(f"    model_name  : {model_name}")
    print(f"    image_loss  : {criteria_image}")
    print(f"    flow_loss   : {criteria_flow}")
    print(f"    disp_loss   : {criterion_disp}")
    print(f"    target_type : {target_type}")
    print(f"    max_epoch   : {max_epoch}")
    print(f"    series_len  : {series_len}")
    print(f"    data_mod    : {data_mod}")
    print("")
    print("=" * 80)

    return model, data_mod
