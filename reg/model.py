from models.transmorph_bspline import TransMorphBspline
from metrics import jacobian_det

import pytorch_lightning as pl
import numpy as np
import torch


class TransMorphModule(pl.LightningModule):
    def __init__(self, net, criterion_image=None, criterion_flow=None, criterion_disp=None, optimizer=torch.optim.Adam,
                 lr=1e-4, target_type="last"):
        super().__init__()
        self.net = net
        self.criterion_image = criterion_image
        self.criterion_flow = criterion_flow
        self.criterion_disp = criterion_disp
        self.optimizer = optimizer
        self.lr = lr
        self.target_type = target_type

    def _get_pred_last_loss(self, batch):
        # always use last image in a seq. (B,C,W,H)
        # targets = torch.repeat_interleave(batch[:, :, :, :, -1][:, :, :, :, None], batch.shape[-1], dim=-1)
        target = batch[:, :, :, :, -1]

        is_diff = isinstance(self.net, TransMorphBspline)
        if is_diff:
            outputs, flows, disp = self.net(batch)
        else:
            outputs, flows = self.net(batch)

        loss = 0

        loss_fn, w = self.criterion_image
        loss += torch.mean(torch.stack([loss_fn(outputs[:, :, :, :, i], target) for i in range(outputs.shape[-1])])) * w

        loss_fn, w = self.criterion_flow
        loss += torch.mean(torch.stack([loss_fn(flows[:, :, :, :, i]) for i in range(flows.shape[-1])])) * w

        if is_diff:
            loss_fn, w = self.criterion_disp
            loss += torch.mean(torch.stack([loss_fn(flows[:, :, :, :, i]) for i in range(flows.shape[-1])])) * w

        return loss, target, outputs, flows

    def _get_preds_loss(self, batch):
        is_diff = isinstance(self.net, TransMorphBspline)

        assert self.criterion_image is not None
        assert self.criterion_flow is not None
        assert self.target_type is not None
        assert not is_diff or self.criterion_disp is not None

        if self.target_type == "last":
            return self._get_pred_last_loss(batch)
        elif self.target_type == "mean":
            raise NotImplementedError("mean loss not implemented")
        else:
            raise NotImplementedError("group loss not implemented")

    def training_step(self, batch, _):
        loss, _, _, _ = self._get_preds_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        loss, _, _, _ = self._get_preds_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True, )

    def test_step(self, batch, _):
        loss, target, _, flows = self._get_preds_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        tar = target.detach().cpu().numpy()[0, :, :, :, :]
        jac_det = jacobian_det(flows.detach().cpu().numpy()[0, :, :, :, :])
        neg_det = np.sum(jac_det <= 0) / np.prod(tar.shape)
        self.log("neg_det", neg_det)

    def predict_step(self, batch, _):
        outputs, flows = self.net(batch)
        return outputs, flows

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.lr)
        return optimizer
