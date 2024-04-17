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

    def _predict(self, batch, fixed):
        assert (len(batch.shape) == len(fixed.shape) and batch.shape[:-1] == fixed.shape[:-1]
                or batch.shape[:-1] == fixed.shape)
        if len(batch.shape) == len(fixed.shape) + 1:
            fixed = fixed.unsqueeze(-1)

        series_len = batch.shape[-1]
        max_len = self.net.transformer.img_size[-1] - 1

        outputs_batch = []
        flows_batch = []
        disp_batch = []

        for n in range(0, series_len, max_len):
            batch_slice_max = max_len
            if n + max_len < series_len:
                batch_slice = batch[..., n:n + max_len]
            else:
                batch_slice = batch[..., n:series_len]
                pad_depth = n + max_len - series_len - 1
                batch_slice_max -= pad_depth
                zeros = torch.zeros((*(batch.shape[:-1]), pad_depth))
                batch_slice = torch.cat([batch_slice, zeros], dim=-1)
            batch_slice = torch.cat([fixed, batch_slice], dim=-1)

            is_diff = isinstance(self.net, TransMorphBspline)
            if is_diff:
                warped, flows, disp = self.net(batch_slice)
                disp_batch.append(disp[1:batch_slice_max])
            else:
                warped, flows = self.net(batch_slice)

            outputs_batch.append(warped[1:batch_slice_max])
            flows_batch.append(flows[1:batch_slice_max])

        warped = torch.stack(outputs_batch)
        flows = torch.stack(flows_batch)
        disp = torch.stack(disp_batch) if disp_batch != [] else torch.tensor([0])

        return warped, flows, disp

    def _fixed_image(self, batch):
        if self.target_type == "last":
            return batch[..., -1]
        elif self.target_type == "mean":
            raise NotImplementedError("mean loss not implemented")
        else:
            raise NotImplementedError("group loss not implemented")

    def _get_preds_loss(self, batch):
        # targets = torch.repeat_interleave(batch[:, :, :, :, -1][:, :, :, :, None], batch.shape[-1], dim=-1)

        fixed = self._fixed_image(batch)
        warped, flows, disp = self._predict(batch, fixed)

        loss = 0
        series_len = batch.shape[-1]
        for loss_fn, w in self.criteria_image:
            loss += torch.mean(
                torch.stack([loss_fn(warped[..., i], fixed) for i in range(series_len)])) * w

        for loss_fn, w in self.criteria_flow:
            loss += torch.mean(torch.stack([loss_fn(flows[..., i]) for i in range(series_len)])) * w

        is_diff = isinstance(self.net, TransMorphBspline)
        if is_diff:
            loss_fn, w = self.criterion_disp
            loss += torch.mean(torch.stack([loss_fn(flows[..., i]) for i in range(series_len)])) * w

        return loss, fixed, warped, flows, disp

    def training_step(self, batch, _):
        loss, _, _, _, _ = self._get_preds_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        loss, _, _, _, _ = self._get_preds_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True, )

    def test_step(self, batch, _):
        loss, target, _, flows, _ = self._get_preds_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        tar = target.detach().cpu().numpy()[0, :, :, :]
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
