import csv
import os
import argparse
from functools import reduce

from pathlib2 import Path
import torchvision.transforms.functional as ttf
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
import torchvision

from configs.transmorph import CONFIGS as CONFIG_DEFAULT
from configs.transmorph_bayes import CONFIGS as CONFIG_BAYES
from configs.transmorph_bspline import CONFIGS as CONFIG_BSPLINE
from dataset import LungDataModule
from model import TransMorphModule
from models.transmorph import TransMorph


def main():
    parser = argparse.ArgumentParser("CLI to create evaluation")
    parser.add_argument("csv_file")
    parser.add_argument("directory")
    parser.add_argument("selected_model")
    args = parser.parse_args()

    dir_name = Path(args.directory)
    dir_res = dir_name / "results"
    csv_file = Path(args.csv_file)
    selected_model = int(args.selected_model)

    config_tm = {}
    config_tm.update(CONFIG_DEFAULT)
    config_tm.update(CONFIG_BAYES)
    config_tm.update(CONFIG_BSPLINE)

    if not os.path.exists(dir_res):
        os.mkdir(dir_res)

    models_param = {}
    with open(csv_file, "r") as f:
        runs = csv.reader(f)
        for i, run in enumerate(runs):

            if i == 0:
                continue

            model_name, image_loss, flow_loss, optimizer_name, lr, series_reg, target_type, max_epoch, series_len = run
            image_loss = str.split(image_loss, ":")[0]
            flow_loss = str.split(flow_loss, ":")[0]
            series_reg = series_reg.lower().capitalize()
            lr = str(float(lr))
            run = (model_name, image_loss, flow_loss, optimizer_name, lr, series_reg, target_type, max_epoch,
                   series_len)
            group_name = reduce(lambda acc, cur: acc + "-" + cur, run)
            group_dir = dir_name / group_name
            file_names = sorted([d for d in os.listdir(group_dir)])
            file_names = [f for f in filter(lambda x: x != '.ipynb_checkpoints', file_names)]

            if len(file_names) == 0:
                continue

            best = group_dir / file_names[0]
            models_param[run] = group_name, best

    run, gn, ckpt = None, None, None
    for i, (run, (gn, ckpt)) in enumerate(models_param.items()):
        if i == selected_model:
            break

    model_name, image_loss, flow_loss, optimizer_name, lr, series_reg, target_type, max_epoch, series_len = run
    series_reg = bool(series_reg)
    series_len = int(series_len)

    print(f" run: {run}")
    print(f"  gn: {gn}")
    print(f"ckpt: {ckpt}")

    data_module = LungDataModule(batch_size=1, num_workers=4, pin_memory=True, mod="norm", series_len=series_len)
    data_module.setup()
    data = data_module.test_dataloader()

    config = config_tm[model_name]
    config.img_size = (*config.img_size[:-1], series_len)
    config.series_reg = series_reg

    model = TransMorphModule.load_from_checkpoint(str(ckpt), strict=False, net=TransMorph(config))
    trainer = pl.Trainer()

    torch.set_float32_matmul_precision("high")
    predictions = trainer.predict(model, data)

    del model
    del trainer
    del data_module
    del data

    num = 0
    out, flow = predictions[num]

    image_data = out[0, 0].detach().cpu()
    target = image_data[:, :, -1]
    image_data_diff = torch.stack([image_data[:, :, i] - target for i in range(series_len)], -1)

    num_cols = 5
    num_rows = 2

    difference = torch.abs(image_data_diff)
    top_k_values, top_k_indices = torch.topk(torch.sum(difference.flatten(start_dim=0, end_dim=-2), dim=0),
                                             num_cols)

    max_image_data = torch.index_select(image_data, -1, top_k_indices)
    max_image_data_diff = torch.index_select(image_data_diff, -1, top_k_indices)

    fig, axs = plt.subplots(num_rows, num_cols + 1, figsize=(2 * num_cols, 2 * num_rows), squeeze=False, dpi=300)

    imgs = torchvision.utils.make_grid(max_image_data).permute(2, 0, 1)
    axs[0, 0].imshow(np.asarray(target), cmap="gray")
    axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs[0, 0].set_aspect("equal")
    axs[0, 0].set_title("Ground Truth")
    for i, img in enumerate(imgs):
        i += 1
        img = torch.abs(img.detach())
        img = ttf.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap="gray")
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_aspect("equal")

    imgs = torchvision.utils.make_grid(max_image_data_diff).permute(2, 0, 1)
    axs[1, 0].imshow(np.asarray(target - target), cmap="gray")
    axs[1, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs[1, 0].set_aspect("equal")
    axs[1, 0].set_title("Difference")
    for i, img in enumerate(imgs):
        i += 1
        img = torch.abs(img.detach())
        img = ttf.to_pil_image(img)
        axs[1, i].imshow(np.asarray(img), cmap="gray")
        axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[1, i].set_aspect("equal")

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.savefig(dir_res / f"{gn}-morph-diff.png")
    fig.show()
    plt.close()

    del predictions
    del image_data


if __name__ == "__main__":
    main()
