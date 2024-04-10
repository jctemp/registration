import glob
import argparse


def main():
    parser = argparse.ArgumentParser("CLI to organise model weights")
    parser.add_argument("directory")

    args = parser.parse_args()
    dir_name = args.directory

    files = [glob.glob(glob.escape(dir_name) + "/*.ckpt")]
    ...


if __name__ == "__main__":
    main()
