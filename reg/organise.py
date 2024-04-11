import argparse
import os
import glob
import csv
from pathlib2 import Path
from functools import reduce


def main():
    parser = argparse.ArgumentParser("CLI to organise model weights")
    parser.add_argument("csv_file")
    parser.add_argument("directory")
    args = parser.parse_args()

    dir_name = Path(args.directory)
    csv_file = Path(args.csv_file)
    files = glob.glob(glob.escape(dir_name) + "/*.ckpt")

    with open(csv_file, "r") as f:
        runs = csv.reader(f)
        grouped = {}

        for i, run in enumerate(runs):
            if i == 0:
                continue
            print(f"{i:>2}: {run}")

            model_name, image_loss, flow_loss, optimizer_name, lr, series_reg, target_type, max_epoch, series_len = run
            image_loss = str.split(image_loss, ":")[0]
            flow_loss = str.split(flow_loss, ":")[0]
            series_reg = series_reg.lower().capitalize()
            lr = str(float(lr))
            run = (model_name, image_loss, flow_loss, optimizer_name, lr, series_reg, target_type, max_epoch,
                   series_len)

            run_group = []
            for file in files:
                contains_substring = [s in file for s in run]
                contains_all = reduce(lambda x, y: x and y, contains_substring)
                if contains_all:
                    run_group.append(file)

            grouped[run] = run_group

        for ident, paths in grouped.items():
            group_name = reduce(lambda acc, cur: acc + "-" + cur, ident)
            group_dir = dir_name / group_name
            if not os.path.exists(group_dir):
                os.mkdir(group_dir)
            for path in paths:
                file_name = Path(path).stem
                file_name = file_name.removeprefix(group_name + "-")
                file_name = file_name.split("-")
                file_name = Path(file_name[2] + "-" + file_name[1])
                file_name = Path(file_name)
                file_name.with_suffix(".ckpt")

                os.rename(path, group_dir / file_name)


if __name__ == "__main__":
    main()
