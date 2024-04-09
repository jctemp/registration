import pytorch_lightning as pl
import numpy as np
from metrics import jacobian_det


class TransMorphModule(pl.LightningModule):
    def __init__(
            self,
            net,
            criterion_image=None,
            criterion_flow=None,
            optimizer=None,
            lr=None,
            target_type=None,
    ):
        super().__init__()
        self.net = net
        self.criterion_image = criterion_image
        self.criterion_flow = criterion_flow
        self.optimizer = optimizer
        self.lr = lr
        self.target_type = target_type

    def _get_pred_last_loss(self, batch):
        # always use last image in a seq. (B,C,W,H)
        target = batch[:, :, :, :, -1]
        outputs, flows = self.net(batch)
        loss = 0

        loss_fn, w = self.criterion_image
        losses = [loss_fn(outputs[..., i], target) for i in range(outputs.shape[-1])]
        loss += sum(losses) * w
        # loss += (sum(losses) / outputs.shape[-1]) * w

        loss_fn, w = self.criterion_flow
        loss += loss_fn(flows) * w

        return loss, target, outputs, flows

    def _get_preds_loss(self, batch):

        if self.criterion_image is None:
            raise Exception("criterion_image is None")
        elif self.criterion_flow is None:
            raise Exception("criterion_flow is None")
        elif self.target_type is None:
            raise Exception("target_type is None")

        if self.target_type == "last":
            return self._get_pred_last_loss(batch)
        elif self.target_type == "mean":
            raise NotImplementedError("mean loss not implemented")
        else:
            raise NotImplementedError("group loss not implemented")

    def training_step(self, batch, _):
        loss, _, _, _ = self._get_preds_loss(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, _):
        loss, _, _, _ = self._get_preds_loss(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True,
        )

    def test_step(self, batch, _):
        loss, target, _, flows = self._get_preds_loss(batch)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True
        )

        tar = target.detach().cpu().numpy()[0, :, :, :]
        jac_det = jacobian_det(
            flows.detach().cpu().numpy()[0, :, :, :, :]
        )
        neg_det = np.sum(jac_det <= 0) / np.prod(tar.shape)
        self.log("neg_det", neg_det)

    def predict_step(self, batch, _):
        outputs, flows = self.net(batch)
        return outputs, flows

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.lr)
        return optimizer
