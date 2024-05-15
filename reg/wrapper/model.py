from enum import Enum
from typing import Any, Tuple, List, Mapping
import gc

import monai.metrics
import numpy as np
import pytorch_lightning as pl
import torch

from reg.transmorph.configs import CONFIG_TM
from reg.transmorph.transmorph_bayes import TransMorphBayes
from reg.transmorph.transmorph import TransMorph
from reg.transmorph.modules.spatial_transformer import SpatialTransformerSeries
from reg.measure import jacobian_det, CONFIGS_WAPRED_LOSS, CONFIGS_FLOW_LOSS

CONFIGS_OPTIMIZER = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adam-w": torch.optim.AdamW,
}


class RegistrationTarget(Enum):
    LAST = 0
    MEAN = 1


class RegistrationStrategy(Enum):
    SOREG = 0
    GOREG = 1


class TransMorphModule(pl.LightningModule):
    def __init__(
            self,
            network: str,
            criteria_warped: List[Tuple[str, float]],
            criteria_flow: List[Tuple[str, float]],
            registration_target: str,
            registration_strategy: str,
            registration_depth: int,
            registration_stride: int,
            identity_loss: bool,
            optimizer: str,
            learning_rate: float,
    ):
        super().__init__()

        self.network = network
        self.criteria_warped = criteria_warped
        self.criteria_flow = criteria_flow
        self.registration_target = registration_target
        self.registration_strategy = registration_strategy
        self.registration_depth = registration_depth
        self.registration_stride = registration_stride
        self.identity_loss = identity_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.save_hyperparameters(ignore=["net", "criteria_warped_nnf", "criteria_flow_nnf", "optimizer_nnf",
                                          "registration_target_e", "registration_strategy_e", "stn"])

        config = CONFIG_TM[self.network]
        config.img_size = (
            *config.img_size[:-1],
            self.registration_depth,
        )

        self.stn = SpatialTransformerSeries(config.img_size)

        descriptors = self.network.split("-")
        self.net = (
            TransMorphBayes(config)
            if len(descriptors) > 1 and descriptors[1] == "bayes"
            else TransMorph(config)
        )

        self.criteria_warped_nnf = [
            (CONFIGS_WAPRED_LOSS[name], weight)
            for name, weight in criteria_warped
        ]

        self.criteria_flow_nnf = [
            (CONFIGS_FLOW_LOSS[name], weight)
            for name, weight in criteria_flow
        ]

        self.registration_target_e = RegistrationTarget[registration_target.upper()]
        self.registration_strategy_e = RegistrationStrategy[registration_strategy.upper()]

        self.optimizer_nnf = CONFIGS_OPTIMIZER[optimizer]

    def _compute_warped_loss(self, warped, fixed) -> float:
        loss = 0
        for loss_fn, weight in self.criteria_warped_nnf:
            loss += torch.mean(
                torch.stack(
                    [loss_fn(w, fixed[..., 0]) for w in warped.permute(-1, 0, 1, 2, 3)]
                )
            )
            loss *= weight
        return loss

    def _compute_flow_loss(self, flow) -> float:
        loss = 0
        for loss_fn, weight in self.criteria_flow_nnf:
            loss += torch.mean(
                torch.stack([loss_fn(f) for f in flow.permute(-1, 0, 1, 2, 3)])
            )
            loss *= weight
        return loss

    def _segment_registration(self, series):
        # Prepare to process series
        fixed = self.extract_fixed_image(series).unsqueeze(-1)
        stride = self.registration_depth - 1
        max_depth = series.shape[-1]

        # Series is smaller than TransMorph input => padding
        if max_depth < stride:
            padding = stride - max_depth
            zeros = torch.zeros((*(series.shape[:-1]), padding), device=series.device)
            in_series = torch.cat([fixed, series, zeros], dim=-1)
            warped, flow = self.net(in_series)

            del series, zeros, in_series

            gc.collect()
            torch.cuda.empty_cache()

            return warped, flow

        # Pre-allocate memory for data concatenation
        warped_shape = (*(series.shape[:-1]), max_depth)
        warped_series = torch.zeros(warped_shape, device=series.device)

        flow_shape = (series.shape[0], 2, *(series.shape[2:-1]), max_depth)
        flow_series = torch.zeros(flow_shape, device=series.device)

        # Process series in segments
        for idx_start in range(0, max_depth, stride):
            idx_end = idx_start + stride
            shift = 0

            # Need to shift to have a valid series input, avoiding zero padding
            if idx_end > max_depth:
                shift = idx_end - max_depth
                idx_start -= shift
                idx_end -= shift

            in_series = torch.cat([fixed, series[..., idx_start:idx_end]], dim=-1)
            warped, flow = self.net(in_series)

            # Note: we need to shift by one to exclude fixed
            warped_series[..., idx_start + shift: idx_end] = warped[..., shift + 1:]
            flow_series[..., idx_start + shift: idx_end] = flow[..., shift + 1:]
            del warped, flow, in_series

        # Clean all de-referenced data
        gc.collect()
        torch.cuda.empty_cache()

        return warped_series, flow_series, fixed

    def _group_registration(self, series):
        image_means = torch.mean(series, keepdim=-1)
        image_means = torch.sort(image_means)
        image_range = torch.max(image_means) - torch.min(image_means)

        def compute_image_bin(curr_mean, curr_higher_lim, bin_step) -> int:
            bin_index = 0
            while curr_mean >= curr_higher_lim:
                bin_index += 1
                curr_higher_lim += bin_step
            return bin_index

        num_bins = 9  # has to be uneven
        step = image_range / num_bins

        image_bins = torch.tensor([compute_image_bin(image_mean, step, step) for image_mean in image_means],
                                  device=series.device)

        # Pre-allocate memory for data concatenation
        flow_series = torch.zeros((series.shape[0], 2, *(series.shape[2:])), device=series.device)

        # 1. intragroup
        avg_group_images = []
        series_fixed = None
        for image_bin in range(0, num_bins):
            group = series[image_bins == image_bin]
            group_mean = torch.mean(image_means[image_bins == image_bin])
            group_mid = group[0]

            for group_image in group:
                mid_diff = torch.abs(group_mean - group_mid)
                cur_diff = torch.abs(group_mean - group_image)
                if cur_diff < mid_diff:
                    group_mid = group_image

            if image_bin == len(avg_group_images) // 2:
                series_fixed = group_mid

            group_series = torch.cat([group_mid, group], dim=-1)
            warped, flow, fixed = self._segment_registration(group_series)

            flow_series[image_bins == image_bin] = flow
            avg_group_images.append(torch.mean(warped, dim=-1))

            del warped, fixed, flow

        # 2. intergroup
        group_series = torch.cat([avg_group_images[len(avg_group_images) // 2]] + avg_group_images, dim=-1)
        warped, flow, fixed = self._segment_registration(group_series)
        del warped, fixed

        for image_bin in range(0, num_bins):
            if image_bin == len(avg_group_images) // 2:
                continue
            flow_series[image_bins == image_bin] += flow[image_bin]
        del flow, image_means, image_bins

        # 3. apply newly computed warp
        warped_series = self.stn(series, flow_series)

        # Clean all de-referenced data
        gc.collect()
        torch.cuda.empty_cache()

        return warped_series, flow_series, series_fixed

    def extract_fixed_image(self, series: torch.Tensor):
        if self.registration_target_e == RegistrationTarget.LAST:
            return series[..., -1].view(series.shape[:-1])
        elif self.registration_target_e == RegistrationTarget.MEAN:
            means = torch.mean(series, (-2, -3)).view(-1)[20:]
            mean_of_means = torch.mean(means)
            diff = torch.abs(means - mean_of_means)
            _, i = torch.topk(diff, 1, largest=False)
            return series[..., i].view(series.shape[:-1])
        else:
            raise ValueError(
                f"strategy has to be last or mean, was {self.registration_target_e.name.lower()}"
            )

    def forward(self, series: torch.Tensor):
        if self.registration_strategy_e == RegistrationStrategy.SOREG:
            return self._segment_registration(series)
        elif self.registration_strategy_e == RegistrationStrategy.GOREG:
            return self._group_registration(series)
        raise ValueError(f"Invalid strategy {self.registration_strategy_e}")

    def training_step(self, batch, **kwargs: Any) -> float:
        warped, flow, fixed = self(batch)

        loss = 0
        loss += self._compute_warped_loss(warped, fixed)
        loss += self._compute_flow_loss(flow)
        del warped, flow

        if self.identity_loss:
            max_depth_transmorph: int = self.net.transformer.img_size[-1] - 1
            in_series = torch.repeat_interleave(fixed, max_depth_transmorph, dim=-1)
            warped, flow, _ = self(in_series)

            loss += self._compute_warped_loss(warped, fixed)
            del warped, flow, in_series

        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, **kwargs: Any) -> float:
        warped, flow, fixed = self(batch)

        loss = 0
        loss += self._compute_warped_loss(warped, fixed)
        loss += self._compute_flow_loss(flow)
        del warped, flow

        self.log_dict(
            {"val_loss": loss}, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )

        return loss

    def test_step(self, batch, **kwargs: Any) -> Mapping[str, Any]:
        warped, flow, fixed = self(batch)

        fixed_np = fixed.detach().cpu().numpy()[0, :, :, :]
        flow_np = flow.detach().cpu().numpy()[0, :, :, :, :]
        jac_det_list = [jacobian_det(flow_np[:, :, :, i]) for i in range(flow_np.shape[-1])]
        perc_neg_jac_det_list = [
            np.sum(jac_det <= 0) / np.prod(fixed_np.shape) for jac_det in jac_det_list
        ]
        perc_neg_jac_det_mean = np.mean(perc_neg_jac_det_list)
        perc_neg_jac_det_var = np.var(perc_neg_jac_det_list)
        perc_neg_jac_det_min = np.min(perc_neg_jac_det_list)
        perc_neg_jac_det_max = np.max(perc_neg_jac_det_list)

        mse = torch.nn.MSELoss()
        mse_list = torch.stack(
            [mse(w, fixed[..., 0]) for w in warped.permute(-1, 0, 1, 2, 3)]
        )
        mse_mean = torch.mean(mse_list)
        mse_var = torch.var(mse_list)
        mse_min = torch.min(mse_list)
        mse_max = torch.max(mse_list)

        ssim = monai.metrics.SSIMMetric(2)
        ssim_list = torch.stack(
            [ssim(w, fixed[..., 0]) for w in warped.permute(-1, 0, 1, 2, 3)]
        )
        ssim_mean = torch.mean(ssim_list)
        ssim_var = torch.var(ssim_list)
        ssim_min = torch.min(ssim_list)
        ssim_max = torch.max(ssim_list)

        loss = 0
        loss += self._compute_warped_loss(warped, fixed)
        loss += self._compute_flow_loss(flow)
        del warped, flow, fixed_np, flow_np

        result = {
            "test_loss": loss,
            "perc_neg_jac_det_mean": perc_neg_jac_det_mean,
            "perc_neg_jac_det_var": perc_neg_jac_det_var,
            "perc_neg_jac_det_min": perc_neg_jac_det_min,
            "perc_neg_jac_det_max": perc_neg_jac_det_max,
            "mse_mean": mse_mean,
            "mse_var": mse_var,
            "mse_min": mse_min,
            "mse_max": mse_max,
            "ssim_mean": ssim_mean,
            "ssim_var": ssim_var,
            "ssim_min": ssim_min,
            "ssim_max": ssim_max,
        }

        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        return result

    def predict_step(self, batch, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        warped, flow, _ = self(batch)
        return warped, flow

    def configure_optimizers(self):
        return self.optimizer_nnf(self.net.parameters(), lr=self.learning_rate)
