import os
import re
import torch
import pandas as pd
from pytorch_lightning import Trainer
from torchvision.utils import save_image
import cv2

from reg.data import LungDataModule
from reg.transmorph.wrapper import TransMorphModule

UPPER_LIMIT = 250


# Define input functions
def get_param_name():
    return input("Enter parameter name: ")


def get_group_directory():
    return input("Enter path to group directory: ")


def get_sample_index():
    return int(input("Enter index to select sample for visualization: "))


# Define helper functions
def sorted_files_in_directory(directory: str):
    files = os.listdir(directory)
    files.sort(key=lambda f: float(re.findall(r'val_loss=([\d.]+)', f)[0]))
    return files


def load_best_model(model_path: str):
    files = sorted_files_in_directory(model_path)
    best_model_path = os.path.join(model_path, files[0])
    model = TransMorphModule.load_from_checkpoint(str(best_model_path), strict=True)
    print(model.hparams)
    return model


def setup_trainer():
    trainer = Trainer()
    return trainer


def setup_data_module():
    n_available_cores = len(os.sched_getaffinity(0)) - 1
    n_available_cores = 1 if n_available_cores == 0 else n_available_cores
    data_module = LungDataModule(
        root_dir="/media/agjvc_rad3/_TESTKOLLEKTIV/Daten/Daten",
        split=(0.7, 0.1, 0.2),
        seed=42,
        pin_memory=True,
        num_workers=n_available_cores,
    )
    data_module.setup()
    return data_module


def predict(trainer: Trainer, model: TransMorphModule, dataloader):
    return trainer.predict(model, dataloader)


def extract_fixed_image(model: TransMorphModule, moving_series: torch.Tensor):
    # Define your extraction method here
    return model.extract_fixed_image(moving_series)


def compute_series_diff(warped_series: torch.Tensor, fixed: torch.Tensor):
    abs_diff_series = torch.abs(warped_series - fixed)
    return abs_diff_series


def transform_flow_series(flow_series: torch.Tensor):
    # Normalizing flow_series to [0, 1] range
    return (flow_series - flow_series.min()) / (flow_series.max() - flow_series.min())


def save_images_and_video(directory, images):
    if not os.path.exists(directory):
        os.makedirs(directory)

    img_paths = []
    for idx, img in enumerate(images):
        img_path = os.path.join(directory, f"{idx}.png")
        save_image(img, img_path)
        img_paths.append(img_path)

    # Create video from images
    frame = cv2.imread(img_paths[0])
    height, width, layers = frame.shape
    video_path = os.path.join(directory, 'video.avi')
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    for img_path in img_paths:
        video.write(cv2.imread(img_path))

    cv2.destroyAllWindows()
    video.release()


def export_results_to_csv(directory, results):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(directory, 'results.csv'), index=False)


def fetch_sample_from_dataloader(dataloader, sample_idx):
    # Fetch a sample from the dataloader based on sample_idx
    for i, batch in enumerate(dataloader):
        if i == sample_idx or sample_idx is None:
            return batch['moving'], batch['fixed']
    # If sample_idx is None, return a random sample
    return next(iter(dataloader))['moving'], next(iter(dataloader))['fixed']


# Main execution
def main(args):
    param_name = args.param
    group_dir = args.group_dir
    sample_idx = args.idx

    model_path = f"model_weights/{group_dir}"
    eval_path = f"model_eval/{group_dir}"

    all_runs = sorted_files_in_directory(model_path)
    for run_id in all_runs:
        run_model_path = os.path.join(model_path, run_id)

        # 1. Load best model
        model = load_best_model(run_model_path)

        # 2. Setup trainer and data module
        trainer = setup_trainer()
        data_module = setup_data_module()
        dataloader = data_module.test_dataloader()

        # 3. Make prediction
        results = predict(trainer, model, dataloader)

        # 4. Extract predictions and inputs
        warped_series, flow_series = results[sample_idx]
        moving_series, fixed_image = fetch_sample_from_dataloader(dataloader, sample_idx)

        # 5. Compute series difference
        series_diff = compute_series_diff(warped_series, fixed_image)

        # 6. Transform flow series
        transformed_flow = transform_flow_series(flow_series)

        # 7. Save images and video
        for images in [warped_series, transformed_flow, series_diff]:
            save_images_and_video(f"{eval_path}/{run_id}", images)

    # # 8. Export run statistics
    # run_statistics = fetch_run_statistics(group_dir,
    #                                       param_name)
    # Define this function to fetch statistics from WandB or other source
    # export_results_to_csv(eval_path, run_statistics)
