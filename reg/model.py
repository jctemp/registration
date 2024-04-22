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

    def predict_segment(self, batch, fixed):
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

    def predict_series(self, batch, fixed):
        assert batch.device == fixed.device
        device = fixed.device

        assert (len(batch.shape) == len(fixed.shape) and batch.shape[:-1] == fixed.shape[:-1]
                or batch.shape[:-1] == fixed.shape)
        if len(batch.shape) == len(fixed.shape) + 1:
            fixed = fixed.unsqueeze(-1)

        # Are we working with a diffeomorphic variant
        is_diff = isinstance(self.net, TransMorphBspline)

        # Determine the series length and transformer max input size regarding temporal dimension (depth)
        max_length = batch.shape[-1]
        max_segment_length = self.net.transformer.img_size[-1] - 1

        # Pre-allocate output vectors to avoid copies
        warped_out = torch.zeros((*(batch.shape[:-1]), max_length), device=device)
        flows_out = torch.zeros((batch.shape[0], 2, *(batch.shape[2:-1]), max_length), device=device)
        disp_out = torch.zeros_like(warped_out) if is_diff else torch.tensor([0], device=device) if is_diff else None

        # Process series
        for start_idx in range(0, max_length, max_segment_length):
            end_idx = start_idx + max_segment_length

            batch_slice_start = 1  # start at one because fixed is always at position 0
            batch_slice_end = max_segment_length + 1  # need to offset by one due to fixed

            if end_idx < max_length:
                batch_slice = batch[..., start_idx:end_idx]
                batch_slice = torch.cat([fixed, batch_slice], dim=-1)

            # TransMorph depth is not larger than input series, so we can place it inside
            #
            # n = 16, msl (max_segment_length) = 4
            # [n0, n1, n2, ..., n-4, n-3, n-2, n-1] n n+1
            # --- start_idx ---------------^ -- msl ---^
            #
            # Compute new starting point to get valid batch within series
            # [n0, n1, n2, ..., n-4, n-3, n-2, n-1] n
            #                    ^-------------------
            #
            # Shift batch_slice_start because images between start_idx and new_start are already registered
            # Note: new_start is always smaller than start_idx
            # [n0, n1, n2, ..., n-4, n-3, n-2, n-1] n
            # --- start_idx ---------------^
            # --- new_start -----^=== k ===^
            #
            # The value k is the difference (e.g. 2) cropping the start of a batch slice away.
            #
            elif max_length - max_segment_length > 0:
                new_start = max_length - max_segment_length
                batch_slice = batch[..., new_start:max_length]
                batch_slice = torch.cat([fixed, batch_slice], dim=-1)
                batch_slice_start += start_idx - new_start

            else:
                batch_slice = batch[..., start_idx:max_length]
                pad_depth = end_idx - max_length
                batch_slice_end -= pad_depth
                zeros = torch.zeros((*(batch.shape[:-1]), pad_depth), device=device)
                batch_slice = torch.cat([fixed, batch_slice, zeros], dim=-1)

            if is_diff:
                warped, flows, disp = self.net(batch_slice)
                disp_out[..., start_idx:end_idx] = disp[..., batch_slice_start:batch_slice_end]
            else:
                warped, flows = self.net(batch_slice)

            warped_out[..., start_idx:end_idx] = warped[..., batch_slice_start:batch_slice_end]
            flows_out[..., start_idx:end_idx] = flows[..., batch_slice_start:batch_slice_end]

        return warped_out, flows_out, disp_out

    def fixed_image(self, batch):
        if self.target_type == "last":
            return batch[..., -1]
        elif self.target_type == "mean":
            means = torch.mean(batch, (-2, -3))[0, 0][20:]
            mean_of_means = torch.mean(means)
            diff = torch.abs(means - mean_of_means)
            _, i = torch.topk(diff, 1, largest=False)
            return batch[..., i].squeeze(-1)
        else:
            raise NotImplementedError("group loss not implemented")

    def _get_preds_loss(self, batch, training=True):
        fixed = self.fixed_image(batch)
        warped, flows, disp = self.predict_segment(batch, fixed) if training else self.predict_series(batch, fixed)

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
        loss, _, _, _, _ = self._get_preds_loss(batch, training=False)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def test_step(self, batch, _):
        loss, fixed, _, flows, _ = self._get_preds_loss(batch, training=False)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        tar = fixed.detach().cpu().numpy()[0, :, :, :]
        flows = flows.detach().cpu().numpy()[0, :, :, :, :]
        jac_det_list = [jacobian_det(flows[:, :, :, i]) for i in range(flows.shape[-1])]
        neg_det_list = [np.sum(jac_det <= 0) / np.prod(tar.shape) for jac_det in jac_det_list]
        neg_det = np.mean(neg_det_list)
        self.log("mean_neg_det", neg_det)

        return loss, {"mean_neg_det": neg_det}

    def predict_step(self, batch, _):
        fixed = self.fixed_image(batch)
        warped, flows, disp = self.predict_series(batch, fixed)
        return warped, flows, disp

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.lr)
        return optimizer
