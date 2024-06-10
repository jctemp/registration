import os
import tempfile
import subprocess
from pathlib import Path
from reg.cli.utils import deserialize_toml


def run_process(command):
    try:
        # Start the process and wait for it to complete
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Get the output and return code
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode

        return stdout, stderr, return_code

    except subprocess.CalledProcessError as e:
        # Handle the error case
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        return e.stdout, e.stderr, e.returncode


def build_job(job_config, header, body, args):
    if not os.path.exists("./tmp"):
        os.mkdir("./tmp")

    dir_path = Path(tempfile.mkdtemp(prefix="reg_", dir="./tmp"))
    config_path = dir_path / "config.toml"

    config_command = [
        "python",
        "-m",
        "reg",
        "generate",
        "--network",
        job_config["network"],
        "--criteria_warped",
        job_config["criteria_warped"],
        "--criteria_flow",
        job_config["criteria_flow"],
        "--registration_strategy",
        job_config["registration_strategy"],
        "--registration_target",
        job_config["registration_target"],
        "--registration_depth",
        str(job_config["registration_depth"]),
        "--registration_stride",
        str(job_config["registration_stride"]),
        "--registration_sampling",
        str(job_config["registration_sampling"]),
        "--optimizer",
        job_config["optimizer"],
        "--learning_rate",
        str(job_config["learning_rate"]),
        "--output",
        str(config_path),
    ]

    if job_config["identity_loss"]:
        config_command.append("--identity_loss")

    stdout, stderr, return_code = run_process(config_command)
    if return_code != 0:
        print(stderr)
        exit(1)

    script = (
        header
        + f"#SBATCH --output={dir_path / 'out.txt'}"
        + f"#SBATCH --error={dir_path / 'err.txt'}"
        + body
        + f"python -m reg train {config_path} --epochs {args.epochs} "
    )

    if args.weight_directory:
        script += f"--weight_directory {args.weight_directory}"

    script_path = dir_path / "batch.sh"
    with open(script_path, "w") as f:
        f.write(script)

    stdout, stderr, return_code = run_process(["sbatch", str(script_path)])
    if return_code != 0:
        print(stderr)
        exit(1)
    print(dir_path)


def main(args):
    header = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name=registration_job_gpu\n"
        f"#SBATCH --mail-user=jamie.temple@stud.hs-hannover.de\n"
        f"#SBATCH --mail-type=ALL\n"
        f"#SBATCH --time=7-00:00:00\n"
        f"#SBATCH --partition leinegpu_long\n"
        f"#SBATCH --nodelist leinewra100\n"
        f"#SBATCH --cpus-per-task={args.cpu}\n"
        f"#SBATCH --mem={args.mem}GB\n"
        f"#SBATCH --gres=gpu:{args.gpu}\n\n"
    )

    body = (
        "module load Python/3.10.4\n"
        "cd /hpc/scratch/project/jvc-lab/stud/registration\n"
        "source ./.venv/bin/activate\n\n"
    )

    file = Path(args.file)
    suffix = file.suffix[1:].upper()

    if suffix == "TOML":
        config = deserialize_toml(str(file.absolute()))

        config["criteria_warped"] = "-".join(
            [str(v) for c in config["criteria_warped"] for v in c]
        )
        config["criteria_flow"] = "-".join(
            [str(v) for c in config["criteria_flow"] for v in c]
        )

        if args.param:
            for v in args.values:
                new_config = config.copy()
                new_config[args.param] = v
                build_job(new_config, header, body, args)
        else:
            build_job(config, header, body, args)
    else:
        print("File suffix unrecognised. Require .toml suffix for file.")
        exit(1)
