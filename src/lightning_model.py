from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from PIL import Image
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
import torchvision
import wandb

class TransMorphModel(pl.LightningModule):
    def __init__(self, tm_model, criterion_image, criterion_flow, optimizer=None, lr=None):
        super().__init__()
        self.tm_model = tm_model
        self.optimizer = optimizer
        self.lr = lr
        self.criterion_image = criterion_image
        self.criterion_flow = criterion_flow

    def _get_preds_loss(self, batch):
        # always use last image in a seq. (B,C,W,H)
        target = batch[:, :, :, :, -1]
        outputs, flows = self.tm_model(batch)
        loss = 0
        
        loss_fn, w = self.criterion_image
        losses = [loss_fn(outputs[..., i], target) for i in range(outputs.shape[-1])]
        loss += sum(losses) * w

        loss_fn, w = self.criterion_flow
        loss += loss_fn(flows) * w

        return loss, target, outputs, flows

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._get_preds_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, _ = self._get_preds_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, target, outputs, flows = self._get_preds_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        tar = target.detach().cpu().numpy()[0,:,:,:]
        jac_det = jacobian_determinant(flows.detach().cpu().numpy()[0, :, :, :, :])
        neg_det = np.sum(jac_det <= 0) / np.prod(tar.shape)
        self.log("neg_det", neg_det)
        
    def predict_step(self, batch, batch_idx):        
        _, _, outputs, flows = self._get_preds_loss(batch)
        return outputs, flows 

    def configure_optimizers(self):
        optimizer = self.optimizer(self.tm_model.parameters(), lr=self.lr)
        return optimizer

# https://github.com/adalca/pystrum/blob/master/pystrum/pynd/ndutils.py
def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.

    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)

# https://github.com/adalca/pystrum/blob/master/pystrum/pynd/ndutils.py
def volsize_to_ndgrid(volsize):
    """
    return the dense nd-grid for the volume with size volsize
    essentially return the ndgrid fpr
    """
    ranges = [np.arange(e) for e in volsize]
    return ndgrid(*ranges)

# https://github.com/voxelmorph/voxelmorph
def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = volsize_to_ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
