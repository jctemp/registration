from enum import Enum
from typing import Any, Tuple, List, Mapping

import monai.metrics
import numpy as np
import pytorch_lightning as pl
import torch

from reg.transmorph.configs import CONFIG_TM
from reg.transmorph.transmorph import TransMorph
from reg.transmorph.transmorph_bayes import TransMorphBayes
from reg.transmorph.transmorph_identity import TransMorphIdentity
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
    MAX = 2
    MIN = 3


class RegistrationStrategy(Enum):
    SOREG = 0
    GOREG = 1


class STN:
    def __init__(self, net, stride):
        self.net = net
        self.stride = stride


class TransMorphModule(pl.LightningModule):
    def __init__(
        self,
        network: str = "transmorph",
        criteria_warped: List[Tuple[str, float]] = tuple([("mse", 1.0)]),
        criteria_flow: List[Tuple[str, float]] = tuple([("gl2d", 1.0)]),
        registration_target: str = "last",
        registration_strategy: str = "soreg",
        registration_depth: int = 32,
        registration_stride: int = 1,
        registration_sampling: int = 0,
        identity_loss: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
    ):
        super().__init__()

        # Hyperparameters (tracked)
        self.network = network
        self.criteria_warped = criteria_warped
        self.criteria_flow = criteria_flow
        self.registration_target = registration_target
        self.registration_strategy = registration_strategy
        self.registration_depth = registration_depth
        self.registration_stride = registration_stride
        self.registration_sampling = registration_sampling
        self.identity_loss = identity_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.id = None

        # Derived parameters (untracked)
        self.net = None
        self.stn = None
        self.criteria_warped_nnf = None
        self.criteria_flow_nnf = None
        self.registration_strategy_e = None
        self.registration_target_e = None
        self.optimizer_nnf = None

        # Section 1: Network
        config = CONFIG_TM[network]
        config.img_size = (
            *config.img_size[:-1],
            self.registration_depth,
        )

        descriptors = network.split("-")
        if len(descriptors) > 1 and descriptors[1] == "bayes":
            self.net = TransMorphBayes(config)
        elif len(descriptors) > 1 and descriptors[1] == "identity":
            self.net = TransMorphIdentity(config)
        else:
            self.net = TransMorph(config)

        self.stn = STN(SpatialTransformerSeries((*config.img_size[:-1], 32)), 32)

        # Section 2: Criteria params
        self.criteria_warped_nnf = [
            (CONFIGS_WAPRED_LOSS[name], weight) for name, weight in criteria_warped
        ]

        self.criteria_flow_nnf = [
            (CONFIGS_FLOW_LOSS[name], weight) for name, weight in criteria_flow
        ]

        # Section 3: Registration params
        self.registration_strategy_e = RegistrationStrategy[
            self.registration_strategy.upper()
        ]

        if self.registration_strategy_e == RegistrationStrategy.GOREG:
            self.registration_target = "mean"

        self.registration_target_e = RegistrationTarget[
            self.registration_target.upper()
        ]

        # Section 4: Miscellaneous params
        self.optimizer_nnf = CONFIGS_OPTIMIZER[optimizer]

        self.update_hyperparameters()

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

    def _sampled_segment_registration(self, series, fixed=None):
        """
        Sampled segment registration of a series of images.
        """

        # Extract fixed image and prepare for processing
        if fixed is None:
            fixed = self.extract_fixed_image(series).unsqueeze(-1)
        max_reg_depth = self.registration_depth - 1
        max_depth = (
            series.shape[-1] if series.shape[-1] < max_reg_depth else max_reg_depth
        )

        # Pre-allocate memory for output series
        warped_shape = (*(series.shape[:-1]), max_depth * self.registration_sampling)
        warped_series = torch.zeros(warped_shape, device=series.device)

        flow_shape = (
            series.shape[0],
            2,
            *(series.shape[2:-1]),
            max_depth * self.registration_sampling,
        )
        flow_series = torch.zeros(flow_shape, device=series.device)

        for i in range(self.registration_sampling):
            # Handle cases where series is smaller than the required input size
            if max_depth < max_reg_depth:
                padding = max_reg_depth - max_depth
                zeros = torch.zeros(
                    (*(series.shape[:-1]), padding), device=series.device
                )
                in_series = torch.cat([fixed, series, zeros], dim=-1)
                del zeros
            else:
                shift = np.random.randint(0, series.shape[-1] - max_depth)
                in_series = torch.cat(
                    [fixed, series[..., shift : shift + max_depth]], dim=-1
                )

            warped, flow = self.net(in_series)

            # Assign the result to the corresponding segment
            warped_series[..., i * max_depth : (i + 1) * max_depth] = warped[
                ..., 1 : max_depth + 1
            ]
            flow_series[..., i * max_depth : (i + 1) * max_depth] = flow[
                ..., 1 : max_depth + 1
            ]

            del warped, flow, in_series

        return warped_series, flow_series, fixed

    def _segment_registration(self, series, fixed=None):
        """
        Segment-wise registration of a series of images.
        """

        # Extract fixed image and prepare for processing
        if fixed is None:
            fixed = self.extract_fixed_image(series).unsqueeze(-1)
        stride = self.registration_depth - 1
        max_depth = series.shape[-1]

        # Pre-allocate memory for output series
        warped_shape = (*(series.shape[:-1]), max_depth)
        warped_series = torch.zeros(warped_shape, device=series.device)

        flow_shape = (series.shape[0], 2, *(series.shape[2:-1]), max_depth)
        flow_series = torch.zeros(flow_shape, device=series.device)

        # Handle cases where series is smaller than the required input size
        if max_depth < stride:
            padding = stride - max_depth
            zeros = torch.zeros((*(series.shape[:-1]), padding), device=series.device)
            in_series = torch.cat([fixed, series, zeros], dim=-1)
            warped, flow = self.net(in_series)

            warped_series[..., :] = warped[..., 1 : max_depth + 1]
            flow_series[..., :] = flow[..., 1 : max_depth + 1]

            del series, zeros, in_series, warped, flow

            return warped_series, flow_series, fixed

        # Process the series in segments
        for idx_start in range(0, max_depth, stride):
            idx_end = idx_start + stride
            shift = 0

            # Adjust indices if segment exceeds the series depth
            if idx_end > max_depth:
                shift = idx_end - max_depth
                idx_start -= shift
                idx_end -= shift

            in_series = torch.cat([fixed, series[..., idx_start:idx_end]], dim=-1)
            warped, flow = self.net(in_series)

            # Assign the result to the corresponding segment
            warped_series[..., idx_start + shift : idx_end] = warped[..., shift + 1 :]
            flow_series[..., idx_start + shift : idx_end] = flow[..., shift + 1 :]

            del warped, flow, in_series

        return warped_series, flow_series, fixed

    def _group_registration(self, series):
        """
        Group-based registration of a series of images.
        """
        raise RuntimeError("Function is not properly implemented.")
        #
        # # Calculate mean intensities and sort them
        # image_means = torch.mean(series, (-2, -3)).view(-1)
        # if series.shape[-1] > 100:
        #     image_min = torch.min(image_means[50:])
        #     image_range = torch.max(image_means[50:]) - image_min
        # else:
        #     image_min = torch.min(image_means)
        #     image_range = torch.max(image_means) - image_min
        #
        # def compute_image_bin(curr_mean, curr_higher_lim, bin_step, max_bin) -> int:
        #     bin_index = 0
        #     while curr_mean >= curr_higher_lim:
        #         bin_index += 1
        #         curr_higher_lim += bin_step
        #         if bin_index == max_bin:
        #             return max_bin
        #     return bin_index
        #
        # num_bins = 9  # Number of bins (must be odd)
        # step = image_range / num_bins
        #
        # # Assign images to bins
        # image_bins = torch.tensor(
        #     [compute_image_bin(image_mean, image_min + step, step, num_bins) for image_mean in image_means],
        #     device=series.device,
        # )
        # # Pre-allocate memory for output series
        # warped_series = torch.zeros(series.shape, device=series.device)
        # flow_series = torch.zeros(
        #     (series.shape[0], 2, *(series.shape[2:])), device=series.device
        # )
        #
        # # Intragroup processing
        # avg_group_images = []
        # series_fixed = None
        # for image_bin in range(0, num_bins):
        #     group = series[..., image_bins == image_bin]
        #
        #     warped, flow, fixed = self._segment_registration(group)
        #     if image_bin == len(avg_group_images) // 2:
        #         series_fixed = fixed
        #
        #     flow_series[..., image_bins == image_bin] = flow
        #     avg_group_images.append(torch.mean(warped, dim=-1))
        #
        #     del warped, flow, fixed
        #
        # # Intergroup processing
        # group_series = torch.cat([t.unsqueeze(-1) for t in avg_group_images], dim=-1)
        # warped, flow, fixed = self._segment_registration(group_series)
        #
        # for image_bin in range(0, num_bins):
        #     group = flow_series[..., image_bins == image_bin]
        #     for i in range(group.shape[-1]):
        #         group[..., i] += (flow[..., image_bin])
        # del warped, flow, fixed, group_series, avg_group_images, image_means, image_bins
        #
        # # Apply newly computed warp
        # for k in range(0, series.shape[-1], 32):
        #     warped_series_seg = self.stn(series[..., k:k + 32], flow_series[..., k:k + 32])
        #     warped_series[..., k:k + 32] = warped_series_seg
        #     del warped_series_seg
        #
        # gc.collect()
        # torch.cuda.empty_cache()
        #
        # return warped_series, flow_series, series_fixed

    def update_hyperparameters(self):
        self.save_hyperparameters(
            ignore=[
                "net",
                "stn",
                "criteria_warped_nnf",
                "criteria_flow_nnf",
                "registration_target_e",
                "registration_strategy_e",
                "optimizer_nnf",
            ]
        )

    def extract_fixed_image(self, series: torch.Tensor):
        if self.registration_target_e == RegistrationTarget.LAST:
            return series[..., -1].view(series.shape[:-1])
        elif self.registration_target_e == RegistrationTarget.MEAN:
            means = torch.mean(series, (-2, -3)).view(-1)
            if means.shape[-1] > 100:
                means = means[30:]
            mean_of_means = torch.mean(means)
            diff = torch.abs(means - mean_of_means)
            _, i = torch.topk(diff, 1, largest=False)
            return series[..., i].view(series.shape[:-1])
        elif self.registration_target_e == RegistrationTarget.MAX:
            if series.shape[-1] > 100:
                series = series[30:]
            means = torch.mean(series, (-2, -3)).view(-1)
            _, i = torch.topk(means, 1, largest=True)
            return series[..., i].view(series.shape[:-1])
        elif self.registration_target_e == RegistrationTarget.MIN:
            if series.shape[-1] > 100:
                series = series[30:]
            means = torch.mean(series, (-2, -3)).view(-1)
            _, i = torch.topk(means, 1, largest=False)
            return series[..., i].view(series.shape[:-1])
        else:
            raise ValueError(
                f"strategy has to be last or mean, was {self.registration_target_e.name.lower()}"
            )

    def forward(self, series: torch.Tensor, training=False):
        if training:
            series = series[..., :: self.registration_stride]

        if (
            training
            and self.registration_sampling > 0
            and self.registration_strategy_e == RegistrationStrategy.SOREG
        ):
            return self._sampled_segment_registration(series)
        if self.registration_strategy_e == RegistrationStrategy.SOREG:
            return self._segment_registration(series)
        elif self.registration_strategy_e == RegistrationStrategy.GOREG:
            return self._group_registration(series)

        raise ValueError(f"Invalid strategy {self.registration_strategy_e}")

    def training_step(self, batch, **kwargs: Any) -> float:
        warped, flow, fixed = self.forward(batch, training=True)

        loss = 0
        loss += self._compute_warped_loss(warped, fixed)
        loss += self._compute_flow_loss(flow)
        del warped, flow

        if self.identity_loss:
            max_depth_transmorph: int = self.net.transformer.img_size[-1] - 1
            in_series = torch.repeat_interleave(fixed, max_depth_transmorph, dim=-1)
            warped, flow, _ = self.forward(in_series, training=True)

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
        jac_det_list = [
            jacobian_det(flow_np[:, :, :, i]) for i in range(flow_np.shape[-1])
        ]
        perc_neg_jac_det_list = [
            np.sum(jac_det <= 0) / np.prod(fixed_np.shape) for jac_det in jac_det_list
        ]
        perc_neg_jac_det_mean = np.mean(perc_neg_jac_det_list)
        perc_neg_jac_det_var = np.var(perc_neg_jac_det_list)

        mse = torch.nn.MSELoss()
        mse_list = torch.stack(
            [mse(w, fixed[..., 0]) for w in warped.permute(-1, 0, 1, 2, 3)]
        )
        mse_mean = torch.mean(mse_list)
        mse_var = torch.var(mse_list)

        ssim = monai.metrics.SSIMMetric(2)
        ssim_list = torch.stack(
            [ssim(w, fixed[..., 0]) for w in warped.permute(-1, 0, 1, 2, 3)]
        )
        ssim_mean = torch.mean(ssim_list)
        ssim_var = torch.var(ssim_list)

        loss = 0
        loss += self._compute_warped_loss(warped, fixed)
        loss += self._compute_flow_loss(flow)
        del warped, flow, fixed_np, flow_np

        result = {
            "test_loss": loss,
            "perc_neg_jac_det_mean": perc_neg_jac_det_mean,
            "perc_neg_jac_det_var": perc_neg_jac_det_var,
            "mse_mean": mse_mean,
            "mse_var": mse_var,
            "ssim_mean": ssim_mean,
            "ssim_var": ssim_var,
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
