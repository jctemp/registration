import math

from models.transmorph_bspline import TransMorphBspline
from models.transmorph_bayes import TransMorphBayes
from models.transmorph import TransMorph
from metrics import jacobian_det

import pytorch_lightning as pl
import numpy as np
import torch


class TransMorphModule(pl.LightningModule):
    def __init__(self, net: TransMorph | TransMorphBspline | TransMorphBayes, criteria_image=None, criteria_flow=None,
                 criterion_disp=None, optimizer=torch.optim.Adam,
                 lr=1e-4, target_type="last"):
        super().__init__()
        self.net = net
        self.criteria_image = criteria_image
        self.criteria_flow = criteria_flow
        self.criterion_disp = criterion_disp
        self.optimizer = optimizer
        self.lr = lr
        self.target_type = target_type

    def _predict_segment(self, batch, fixed):
        assert batch.device == fixed.device
        assert (len(batch.shape) == len(fixed.shape) and batch.shape[:-1] == fixed.shape[:-1]
                or batch.shape[:-1] == fixed.shape)
        if len(batch.shape) == len(fixed.shape) + 1:
            fixed = fixed.unsqueeze(-1)

        # Are we working with a diffeomorphic variant
        is_diff = isinstance(self.net, TransMorphBspline)

        # Determine the series length and transformer max input size regarding temporal dimension (depth)
        series_len: int = batch.shape[-1]
        slice_max: int = self.net.transformer.img_size[-1] - 1
        offset_max: int = series_len - slice_max

        start_idx = np.random.randint(0, offset_max)
        batch_slice = batch[..., start_idx:start_idx + slice_max]
        batch_slice = torch.cat([fixed, batch_slice], dim=-1)

        if is_diff:
            warped, flows, disp = self.net(batch_slice)
        else:
            (warped, flows), disp = self.net(batch_slice), None

        if disp:
            return warped[..., 1:slice_max + 1], flows[..., 1:slice_max + 1], disp[..., 1:slice_max + 1]
        return warped[..., 1:slice_max + 1], flows[..., 1:slice_max + 1], disp

    def _predict_series(self, batch, fixed):
        assert batch.device == fixed.device
        device = fixed.device

        assert (len(batch.shape) == len(fixed.shape) and batch.shape[:-1] == fixed.shape[:-1]
                or batch.shape[:-1] == fixed.shape)
        if len(batch.shape) == len(fixed.shape) + 1:
            fixed = fixed.unsqueeze(-1)

        # Are we working with a diffeomorphic variant
        is_diff = isinstance(self.net, TransMorphBspline)

        # Determine the series length and transformer max input size regarding temporal dimension (depth)
        series_len = batch.shape[-1]
        max_len = self.net.transformer.img_size[-1] - 1

        # Pre-allocate output vectors to avoid copies
        warped_out = torch.zeros((*(batch.shape[:-1]), series_len), device=device)
        flows_out = torch.zeros((batch.shape[0], 2, *(batch.shape[2:-1]), series_len), device=device)
        disp_out = torch.zeros_like(warped_out) if is_diff else torch.tensor([0], device=device) if is_diff else None

        # Process series
        for start_idx in range(0, series_len, max_len):
            batch_slice_max = max_len
            end_idx = start_idx + batch_slice_max

            if start_idx + max_len < series_len:
                batch_slice = batch[..., start_idx:start_idx + max_len]
                batch_slice = torch.cat([fixed, batch_slice], dim=-1)
            else:
                batch_slice = batch[..., start_idx:series_len]
                pad_depth = start_idx + max_len - series_len
                batch_slice_max -= pad_depth
                zeros = torch.zeros((*(batch.shape[:-1]), pad_depth), device=device)
                batch_slice = torch.cat([fixed, batch_slice, zeros], dim=-1)

            if is_diff:
                warped, flows, disp = self.net(batch_slice)
                disp_out[..., start_idx:end_idx] = disp[..., 1:batch_slice_max + 1]
            else:
                warped, flows = self.net(batch_slice)

            warped_out[..., start_idx:end_idx] = warped[..., 1:batch_slice_max + 1]
            flows_out[..., start_idx:end_idx] = flows[..., 1:batch_slice_max + 1]

        return warped_out, flows_out, disp_out

    def _fixed_image(self, batch):
        if self.target_type == "last":
            return batch[..., -1]
        elif self.target_type == "mean":
            raise NotImplementedError("mean loss not implemented")
        else:
            raise NotImplementedError("group loss not implemented")

    def _get_preds_loss(self, batch, training=True):
        fixed = self._fixed_image(batch)
        warped, flows, disp = self._predict_segment(batch, fixed) if training else self._predict_series(batch, fixed)

        # identity loss?
        # targets = torch.repeat_interleave(fixed[:, :, :, :, None], batch.shape[-1], dim=-1)

        loss = 0
        series_len = warped.shape[-1]
        for loss_fn, w in self.criteria_image:
            loss += torch.mean(
                torch.stack([loss_fn(warped[..., i], fixed) for i in range(series_len)])) * w

        for loss_fn, w in self.criteria_flow:
            loss += torch.mean(torch.stack([loss_fn(flows[..., i]) for i in range(series_len)])) * w

        if disp:
            loss_fn, w = self.criterion_disp
            loss += torch.mean(torch.stack([loss_fn(flows[..., i]) for i in range(series_len)])) * w

        return loss, fixed, warped, flows, disp

    def training_step(self, batch, _):
        loss, _, _, _, _ = self._get_preds_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        loss, _, _, _, _ = self._get_preds_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def test_step(self, batch, _):
        loss, fixed, _, flows, _ = self._get_preds_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        tar = fixed.detach().cpu().numpy()[0, :, :, :]
        flows = flows.detach().cpu().numpy()[0, :, :, :, :]
        jac_det_list = [jacobian_det(flows[:, :, :, i]) for i in range(flows.shape[-1])]
        neg_det_list = [np.sum(jac_det <= 0) / np.prod(tar.shape) for jac_det in jac_det_list]
        neg_det = np.mean(neg_det_list)
        self.log("mean_neg_det", neg_det)

        return loss, {"mean_neg_det": neg_det}

    def predict_step(self, batch, _):
        fixed = self._fixed_image(batch)
        warped, flows, disp = self._predict(batch, fixed)
        return warped, flows, disp

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.lr)
        return optimizer
