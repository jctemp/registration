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
            context_length: int = 256,
            optimizer: str = "adam",
            learning_rate: float = 1e-4,
    ):
        super().__init__()

        # Hyperparameters (tracked)
        self.network = network
        self.criteria_warped = criteria_warped
        self.criteria_flow = criteria_flow
        self.context_length = context_length
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.id = None

        # Derived parameters (untracked)
        self.net = None
        self.stn = None
        self.criteria_warped_nnf = None
        self.criteria_flow_nnf = None
        self.optimizer_nnf = None

        # Section 1: Network
        config = CONFIG_TM[network]
        config.img_size = (
            *config.img_size[:-1],
            self.context_length,
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

        # Section 4: Miscellaneous params
        self.optimizer_nnf = CONFIGS_OPTIMIZER[optimizer]

        self.update_hyperparameters()

    def _compute_warped_loss(self, warped) -> float:
        loss = 0.0

        warped = warped.permute(-1, 0, 1, 2, 3)
        successor_warped = torch.roll(warped, -1, dims=0)
        predecessor_warped = torch.roll(warped, 1, dims=1)

        s_warped = zip(warped[:-1], successor_warped[:-1])
        p_warped = zip(warped[1:], predecessor_warped[1:])

        for loss_fn, weight in self.criteria_warped_nnf:
            s_values = [loss_fn(w, s) for (w, s) in s_warped]
            p_values = [loss_fn(w, p) for (w, p) in p_warped]

            s_mean = sum(s_values) / len(s_values) * weight
            p_mean = sum(p_values) / len(p_values) * weight

            loss += (s_mean + p_mean) * 0.5

        return loss

    def _compute_flow_loss(self, flow) -> float:
        loss = 0.0
        for loss_fn, weight in self.criteria_flow_nnf:
            loss += torch.mean(
                torch.stack([loss_fn(f) for f in flow.permute(-1, 0, 1, 2, 3)])
            )
            loss *= weight
        return loss

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

    def forward(self, series: torch.Tensor):
        """
        Segment-wise registration of a series of images.
        """
        max_depth = series.shape[-1]

        # Pre-allocate memory for output series
        warped_shape = (*(series.shape[:-1]), max_depth)
        warped_series = torch.zeros(warped_shape, device=series.device)

        flow_shape = (series.shape[0], 2, *(series.shape[2:-1]), max_depth)
        flow_series = torch.zeros(flow_shape, device=series.device)

        # Handle cases where series is smaller than the required input size
        if max_depth < self.context_length:
            padding = self.context_length - max_depth
            zeros = torch.zeros((*(series.shape[:-1]), padding), device=series.device)
            in_series = torch.cat([series, zeros], dim=-1)
            warped, flow = self.net(in_series)

            warped_series[..., :] = warped[..., : max_depth]
            flow_series[..., :] = flow[..., : max_depth]

            del series, zeros, in_series, warped, flow

            return warped_series, flow_series

        # Process the series in segments
        for idx_start in range(0, max_depth, self.context_length):
            idx_end = idx_start + self.context_length
            shift = 0

            # Adjust indices if segment exceeds the series depth
            if idx_end > max_depth:
                shift = idx_end - max_depth
                idx_start -= shift
                idx_end -= shift

            warped, flow = self.net(series[..., idx_start:idx_end])

            # Assign the result to the corresponding segment
            warped_series[..., idx_start + shift: idx_end] = warped[..., shift:]
            flow_series[..., idx_start + shift: idx_end] = flow[..., shift:]

            del warped, flow

        return warped_series, flow_series

    def training_step(self, batch, **kwargs: Any) -> float:
        warped, flow = self.forward(batch)

        loss = 0
        loss += self._compute_warped_loss(warped)
        loss += self._compute_flow_loss(flow)
        del warped, flow

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
        loss += self._compute_warped_loss(warped)
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
        loss += self._compute_warped_loss(warped)
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
