import argparse

from reg.transmorph.configs import CONFIG_TM


def create_parser() -> argparse.ArgumentParser:
    """
    Create the parser for the CLI.
    """

    parser = argparse.ArgumentParser(
        prog="reg", description="Series-based TransMorph registration CLI"
    )
    parser.add_argument("-l", "--log", action="store_true", help="enable logging")

    subparsers = parser.add_subparsers(
        dest="command",
    )

    parser_train = subparsers.add_parser(
        name="train",
        description="Creates a new wrapper. Sets hyperparameters and starts training",
    )
    parser_train.add_argument(
        "--resume", type=str, help="checkpoint file load to continue training"
    )
    parser_train.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train the wrapper"
    )
    parser_train.add_argument(
        "network",
        type=str,
        default="transmorph",
        help="the name of the network configuration preset.",
        choices=list(CONFIG_TM.keys()),
        nargs="?",
    )
    parser_train.add_argument(
        "--criteria_warped",
        type=str,
        default="gmi-1",
        help="metric to score deformation, e.g. 'gmi-1-ncc-1'",
    )
    parser_train.add_argument(
        "--criteria_flow",
        type=str,
        default="gl2d-1",
        help="regularization for DVF, e.g. 'gl2d-1-bel-1'",
    )
    parser_train.add_argument(
        "--optimizer", type=str, default="adam", help="optimizer for gradient descent"
    )
    parser_train.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="step size for gradient descent",
    )
    parser_train.add_argument(
        "--registration_strategy",
        type=str,
        default="soreg",
        help="register segment or group-wise",
        choices=["soreg", "goreg"],
    )
    parser_train.add_argument(
        "--registration_target",
        type=str,
        default="last",
        help="the selection process to find a fixed image",
        choices=["last", "mean"],
    )
    parser_train.add_argument(
        "--registration_depth",
        type=int,
        default=32,
        help="the number of simultaneously registered images",
        choices=range(32, 256, 32),
    )
    parser_train.add_argument(
        "--registration_stride",
        type=int,
        default=1,
        help="defines how to subsample temporal dimension",
    )
    parser_train.add_argument(
        "--registration_sampling",
        type=int,
        default=1,
        help="chunk series during training or only present n samples; zero means chunking",
        choices=range(0, 9)
    )
    parser_train.add_argument(
        "--identity_loss",
        action="store_true",
        help="consider target image deformation",
    )

    parser_pred = subparsers.add_parser(
        name="pred",
        description="Loads a wrapper from a ckpt file and provides a prediction for a single file.",
    )
    parser_pred.add_argument(
        "ckpt", type=str, help="path to a checkpoint file loading a TransMorphModule"
    )
    parser_pred.add_argument(
        "sample", type=str, help="path to a matlab file holding the series data"
    )

    return parser
