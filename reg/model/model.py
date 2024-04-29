from enum import Enum
from typing import Any, Tuple, List

from reg.transmorph.transmorph_bayes import TransMorphBayes
from reg.transmorph.transmorph import TransMorph
from reg.metrics import jacobian_det

import gc
import torch
import numpy as np
import pytorch_lightning as pl


class RegistrationTarget(Enum):
    LAST = 0
    MEAN = 1


class RegistrationStrategy(Enum):
    SOREG = 0
    GOREG = 1


class TransMorphModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net: TransMorph | TransMorphBayes | None = None
        self.criteria_warped: List[Tuple[torch.nn.Module, float]] = []
        self.criteria_flow: List[Tuple[torch.nn.Module, float]] = []
        self.optimizer: torch.optim = torch.optim.Adam
        self.learning_rate: float = 1e-4
        self.registration_target: RegistrationTarget = RegistrationTarget.LAST
        self.registration_strategy: RegistrationStrategy = RegistrationStrategy.SOREG
        self.registration_depth: int = 32
        self.identity_loss: bool = False
        self.config = {}

    def _compute_warped_loss(self, warped, fixed) -> float:
        loss = 0
        for loss_fn, weight in self.criteria_warped:
            loss += torch.mean(
                torch.stack([loss_fn(w, fixed) for w in warped.permute(-1, 0, 1, 2)])
            )
            loss *= weight
        return loss

    def _compute_flow_loss(self, flow) -> float:
        loss = 0
        for loss_fn, weight in self.criteria_flow:
            loss += torch.mean(
                torch.stack([loss_fn(f) for f in flow.permute(-1, 0, 1, 2)])
            )
            loss *= weight
        return loss

    def _extract_fixed_image(self, series: torch.Tensor):
        if self.registration_target == RegistrationTarget.LAST:
            return series[..., -1].view(series.shape[:-1])
        elif self.registration_target == RegistrationTarget.MEAN:
            means = torch.mean(series, (-2, -3)).view(-1)[20:]
            mean_of_means = torch.mean(means)
            diff = torch.abs(means - mean_of_means)
            _, i = torch.topk(diff, 1, largest=False)
            return series[..., i].view(series.shape[:-1])
        else:
            raise ValueError(
                f"strategy has to be last or mean, was {self.registration_target.name.lower()}"
            )

    def _segment_registration(self, series):
        # Prepare to process series
        fixed = self._extract_fixed_image(series).unsqueeze(-1)
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

            warped_series[..., idx_start + shift : idx_end] = warped[..., shift:]
            flow_series[..., idx_start + shift : idx_end] = flow[..., shift:]
            del warped, flow, in_series

        # Clean all de-referenced data
        gc.collect()
        torch.cuda.empty_cache()

        return warped_series, flow_series, fixed

    def _group_registration(self, series):
        raise NotImplementedError()

    def forward(self, series: torch.Tensor):
        # Check image width and height
        assert (
            torch.prod(series.shape[:-1])
            == torch.prod(self.net.transformer.img_size.shape[:-1]),
            "The tensor x does not fulfill TransMorph input requirements. "
            f"Expected {self.net.transformer.img_size.shape[:-1]}, Got {series.shape[:-1]}. "
            f"Note: last dimension was removed as it is not relevant for the input.",
        )

        if RegistrationStrategy.SOREG:
            return self._segment_registration(series)
        elif RegistrationStrategy.GOREG:
            return self._group_registration(series)
        raise ValueError(f"Invalid strategy {self.registration_strategy}")

    def training_step(self, batch, **kwargs: Any) -> None:
        warped, flow, fixed = self(batch)

        loss = 0
        loss += self._compute_warped_loss(warped, fixed)
        loss += self._compute_flow_loss(flow)
        del warped, flow

        if self.identity_loss:
            max_depth_transmorph: int = self.net.transformer.img_size[-1] - 1
            in_series = torch.repeat_interleave(fixed, max_depth_transmorph, dim=-1)
            warped, flow, _ = self(in_series, fixed)

            loss += self._compute_warped_loss(warped, fixed)
            del warped, flow, in_series

        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def validation_step(self, batch, **kwargs: Any) -> None:
        warped, flow, fixed = self(batch)

        loss = 0
        loss += self._compute_warped_loss(warped, fixed)
        loss += self._compute_flow_loss(flow)
        del warped, flow

        self.log_dict(
            {"val_loss": loss}, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )

    def test_step(self, batch, **kwargs: Any) -> None:
        warped, flow, fixed = self(batch)

        tar = fixed.detach().cpu().numpy()[0, :, :, :]
        flows = flow.detach().cpu().numpy()[0, :, :, :, :]
        jac_det_list = [jacobian_det(flows[:, :, :, i]) for i in range(flows.shape[-1])]
        neg_det_list = [
            np.sum(jac_det <= 0) / np.prod(tar.shape) for jac_det in jac_det_list
        ]
        neg_det = np.mean(neg_det_list)

        loss = 0
        loss += self._compute_warped_loss(warped, fixed)
        loss += self._compute_flow_loss(flow)
        del warped, flow, tar, flows

        self.log_dict(
            {"test_loss": loss, "mean_neg_det": neg_det},
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def predict_step(self, batch, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        warped, flow, _ = self(batch)
        return warped, flow

    def configure_optimizers(self):
        return self.optimizer(self.net.parameters(), lr=self.learning_rate)