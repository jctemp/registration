from pathlib2 import Path

import argparse
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from configs.transmorph import CONFIGS as CONFIG_DEFAULT
from configs.transmorph_bayes import CONFIGS as CONFIG_BAYES
from configs.transmorph_bspline import CONFIGS as CONFIG_BSPLINE
from dataset import LungDataModule
from metrics import *
from model import TransMorphModule
from models.transmorph import TransMorph
from models.transmorph_bayes import TransMorphBayes
from models.transmorph_bspline import TransMorphBspline

CONFIG_TM = {}
CONFIG_TM.update(CONFIG_DEFAULT)
CONFIG_TM.update(CONFIG_BAYES)
CONFIG_TM.update(CONFIG_BSPLINE)

CONFIGS_IMAGE_LOSS = {
    "mse": mse(),
    "ncc": lncc(spatial_dims=2),
    "gmi": gmi(),
}

CONFIGS_FLOW_LOSS = {
    "gl2d": gl2d(penalty="l2"),
    "bel": bel(normalize=True),
}


def main():
    parser = argparse.ArgumentParser("CLI to create evaluation")
    parser.add_argument("ckpt_dir", type=str, help="Path to check points dir")
    parser.add_argument("save_dir", type=str, help="Path to figure save dir")
    parser.add_argument("--test", default=False)
    parser.add_argument("--num_worker", default=1)
    parser.add_argument("--show-title", default=False)
    args = parser.parse_args()

    show_title = args.show_title
    num_worker = int(args.num_worker)
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save_dir)

    ckpt_file_names = sorted([c.name for c in ckpt_dir.glob("*.ckpt")], reverse=True)
    best_ckpt = ckpt_dir / ckpt_file_names[0]
    val_loss, epoch = ckpt_file_names[0].split("&")
    val_loss = val_loss.split("=")[1]
    epoch = epoch.split("=")[1]

    model_name, image_loss, flow_loss, optimizer_name, lr, target_type, max_epoch, series_len, data_mod = str(
        ckpt_dir.name).split("-")
    image_loss, image_loss_weight = image_loss.split("=")
    flow_loss, flow_loss_weight = flow_loss.split("=")
    max_epoch = int(max_epoch)
    series_len = int(series_len)

    print("=" * 80)
    print(f"Best performing model")
    print("")
    print(f"    val_loss    : {val_loss}")
    print(f"    epoch       : {epoch}")
    print("")
    print(f"    model_name  : {model_name}")
    print(f"    image_loss  : {image_loss}:{image_loss_weight}")
    print(f"    flow_loss   : {flow_loss}:{flow_loss_weight}")
    print(f"    target_type : {target_type}")
    print(f"    max_epoch   : {max_epoch}")
    print(f"    series_len  : {series_len}")
    print(f"    data_mod    : {data_mod}")
    print("")
    print("=" * 80)

    criterion_image = (CONFIGS_IMAGE_LOSS[image_loss], float(image_loss_weight))
    criterion_flow = (CONFIGS_FLOW_LOSS[flow_loss], float(flow_loss_weight))
    criterion_disp = None

    data_module = LungDataModule(batch_size=1, num_workers=num_worker, pin_memory=False, mod=data_mod,
                                 series_len=series_len)
    data_module.setup()
    data_loader = data_module.test_dataloader()

    config = CONFIG_TM[model_name]
    config.img_size = (*config.img_size[:-1], series_len)

    model = TransMorphModule.load_from_checkpoint(str(best_ckpt), criterion_image=criterion_image,
                                                  criterion_flow=criterion_flow, criterion_disp=criterion_disp,
                                                  strict=False, net=TransMorph(config))
    trainer = pl.Trainer()

    torch.set_float32_matmul_precision("high")

    if args.test:
        trainer.test(model, data_loader, verbose=True)
        return

    predictions = trainer.predict(model, data_loader)

    nums = [10, 20, 30, 40, 50]
    subset_moving = []
    subset_warped = []

    for i, batch in enumerate(data_loader):
        if i in nums:
            subset_moving.append(batch)
            subset_warped.append(predictions[i])

    for moving, (warped, dvf) in zip(subset_moving, subset_warped):
        dim = (3, 1, 2, 0)

        # re-order channels of outputs (t, w, h, c)
        moving = moving[0].detach().cpu().permute(dim)
        warped = warped[0].detach().cpu().permute(dim)
        dvf = dvf[0].detach().cpu().permute(dim)

        # TODO: target types???
        fixed = moving[-1]

        # create diff map using comparison to neighbour
        abs_diff = torch.stack([torch.abs(w - fixed) for w in warped], dim=0)

        # create dvf colour map
        dvf = torch.tanh(dvf[:, :, :])
        dvf_x = (dvf[:, :, :, 0] + 1) / 2
        dvf_y = (dvf[:, :, :, 1] + 1) / 2
        dvf_z = dvf_x * 0
        dvf = torch.stack([dvf_x, dvf_y, dvf_z], dim=-1)

        # plots
        num_cols = 5
        num_rows = 1  # show two worst image

        metric = torch.nn.MSELoss()
        loss = torch.stack(
            [metric(w.permute(2, 0, 1).unsqueeze(0), fixed.permute(2, 0, 1).unsqueeze(0)) for w in warped], dim=0)

        top_k_values, top_k_indices = torch.topk(loss, num_rows)  # select images with the highest loss
        k_moving = torch.index_select(moving, 0, top_k_indices)
        k_warped = torch.index_select(warped, 0, top_k_indices)
        k_dvf = torch.index_select(dvf, 0, top_k_indices)
        k_abs_diff = torch.index_select(abs_diff, 0, top_k_indices)

        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(2 * num_cols, 2 * num_rows), squeeze=False,
                                dpi=300)
        axs = axs.flatten()

        for i, (m, w, d, a) in enumerate(zip(k_moving, k_warped, k_dvf, k_abs_diff)):
            if show_title and i == 0:
                axs[i * num_cols + 0].set(title=r"$\mathit{m}$")
                axs[i * num_cols + 1].set(title=r"$\mathit{f}$")
                axs[i * num_cols + 2].set(title=r"$\mathit{m \circ \phi}$")
                axs[i * num_cols + 3].set(title=r"$\mathit{\left| \; m - f \; \right|}$")
                axs[i * num_cols + 4].set(title=r"$\mathit{\phi}$")

            axs[i * num_cols + 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], aspect="equal")
            axs[i * num_cols + 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], aspect="equal")
            axs[i * num_cols + 2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], aspect="equal")
            axs[i * num_cols + 3].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], aspect="equal")
            axs[i * num_cols + 4].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], aspect="equal")

            axs[i * num_cols + 0].imshow(m, cmap="gray")
            axs[i * num_cols + 1].imshow(fixed, cmap="gray")
            axs[i * num_cols + 2].imshow(w, cmap="gray")
            axs[i * num_cols + 3].imshow(a, cmap="gray")
            axs[i * num_cols + 4].imshow(d, cmap="gray")

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        fig.savefig(save_dir / f"{ckpt_dir.name}-result.png")
        plt.close()


if __name__ == "__main__":
    main()
