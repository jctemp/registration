from pathlib2 import Path

from reg.cli.utils import serialize_data


def main(args):
    filename = Path(args.output)
    if filename.suffix != "toml":
        filename = filename.with_suffix(".toml")

    if filename.exists():
        print(
            f"{filename.absolute()} already exists. Please remove to generate config."
        )
        exit(1)

    criteria_warped_list = args.criteria_warped.split("-")
    criteria_warped = []
    for i in range(0, len(criteria_warped_list), 2):
        name = criteria_warped_list[i]
        weight = float(criteria_warped_list[i + 1])
        criteria_warped.append((name, weight))

    criteria_flow_list = args.criteria_flow.split("-")
    criteria_flow = []
    for i in range(0, len(criteria_flow_list), 2):
        name = criteria_flow_list[i]
        weight = float(criteria_flow_list[i + 1])
        criteria_flow.append((name, weight))

    config = {
        "network": args.network,
        "criteria_warped": criteria_warped,
        "criteria_flow": criteria_flow,
        "registration_target": args.registration_target,
        "registration_strategy": args.registration_strategy,
        "registration_depth": args.registration_depth,
        "registration_stride": args.registration_stride,
        "registration_sampling": args.registration_sampling,
        "identity_loss": args.identity_loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
    }

    print("Writing model configuration: \n", config)
    print(f"Path: {str(filename.absolute())}")
    serialize_data(config, str(filename.absolute()))
