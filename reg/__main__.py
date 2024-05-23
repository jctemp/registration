import argparse

from reg.cli.parameter_generator import main as generate_main
from reg.cli.batch_training import main as batch_main
from reg.cli.model_training import main as train_main
# from reg.cli.predict import main as predict_main


def main():
    parser = argparse.ArgumentParser(
        prog="reg", description="Series-based TransMorph registration CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
    )

    create_parser_generate(subparsers)
    create_parser_train(subparsers)
    create_parser_predict(subparsers)
    create_parser_batch(subparsers)

    args = parser.parse_args()

    if args.command == "generate":
        generate_main(args)

    elif args.command == "train":
        import torch
        if not torch.cuda.is_available():
            print("Require CUDA to train wrapper")
            raise RuntimeError("CUDA is not available")
        torch.set_float32_matmul_precision("high")
        train_main(args)

    elif args.command == "predict":
        import torch
        if not torch.cuda.is_available():
            print("Require CUDA to train wrapper")
            raise RuntimeError("CUDA is not available")
        torch.set_float32_matmul_precision("high")
        exit(1)

    elif args.command == "batch":
        batch_main(args)

    else:
        parser.print_help()


def create_parser_generate(subparsers):
    from reg.transmorph.configs import CONFIG_TM
    from reg.transmorph.wrapper import CONFIGS_OPTIMIZER

    parser = subparsers.add_parser(
        name="generate",
        description="Generates a configuration file with specified parameters",
    )

    parser.add_argument(
        "--network",
        type=str,
        default="transmorph",
        help="Set the TransMorph configuration name",
        choices=list(CONFIG_TM.keys()),
    )

    parser.add_argument(
        "--criteria_warped",
        type=str,
        default="mse-1.0",
        help="Measure to evaluate the quality of warped images, e.g. 'gmi-1-ncc-1'",
    )

    parser.add_argument(
        "--criteria_flow",
        type=str,
        default="gl2d-1.0",
        help="Measure to regularise the deformation vector fields, e.g. 'gl2d-1-bel-0.125'",
    )

    parser.add_argument(
        "--registration_strategy",
        type=str,
        default="soreg",
        help="Strategy for registration, meaning process the series segment-wise or group-wise",
        choices=["soreg", "goreg"],
    )

    parser.add_argument(
        "--registration_target",
        type=str,
        default="last",
        help="The selection process for the target image to which all images will be aligned to",
        choices=["last", "mean"],
    )

    parser.add_argument(
        "--registration_depth",
        type=int,
        default=32,
        help="The temporal dimension size the model can process at once",
        choices=range(32, 256, 32),
    )

    parser.add_argument(
        "--registration_stride",
        type=int,
        default=1,
        help="The size of subsampling in the temporal dimension",
        choices=range(1, 8)
    )

    parser.add_argument(
        "--registration_sampling",
        type=int,
        default=1,
        help="Sample the series to present single segments to the model",
        choices=range(0, 9)
    )

    parser.add_argument(
        "--identity_loss",
        action="store_true",
        help="Set flag to have an identity loss considered",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Set the optimization algorithm for gradient descent",
        choices=list(CONFIGS_OPTIMIZER.keys()),
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="The step size for gradient descent during training",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="config",
        help="Set the output file name"
    )


def create_parser_train(subparsers):
    parser = subparsers.add_parser(
        name="train",
        description="Train the model using the specified configuration file",
    )

    parser.add_argument(
        "file",
        type=str,
        default="config.toml",
        help="A file that will be loaded for training (configuration or checkpoint)",
    )

    parser.add_argument(
        "--weight_directory",
        type=str,
        default=None,
        help="The name of the subdirectory where weights are saved. If none is specified, it will be a random "
             "identifier."
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="The number of epochs to perform training"
    )


def create_parser_predict(subparsers):
    parser = subparsers.add_parser(
        name="predict",
        description="Run the predictor for files",
    )


def create_parser_batch(subparsers):
    parser = subparsers.add_parser(
        name="batch",
        description="Offload training to the HPC as a batch job.",
    )
    parser.add_argument("file", type=str)
    parser.add_argument("param", type=str, default=None, nargs="?")
    parser.add_argument("values", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight_directory", type=str, default=None, )
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--mem", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=1)


if __name__ == "__main__":
    main()
