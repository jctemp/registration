import torch
from monai.utils import LossReduction
EPS = 1e-8


class GlobalNormalizedCrossCorrelationLoss(torch.nn.Module):
    """
    Global squared zero-normalized cross-correlation.

    Compute the squared cross-correlation between the reference and moving images
    y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference:

        - Zero-normalized cross-correlation (ZNCC):
            https://en.wikipedia.org/wiki/Cross-correlation

    Ported from DeepReg (tf) to torch
    """

    def __init__(
            self,
            name: str = "GlobalNormalizedCrossCorrelation",
            reduction: LossReduction | str = LossReduction.MEAN,
    ):
        """
        Init.
        :param name: name of the loss
        """
        super().__init__(name=name)
        self.reduction = reduction

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch, ...)
        """
        dim = y_pred.shape[1:]

        mu_pred = torch.mean(y_pred, dim=dim, keepdim=True)
        mu_true = torch.mean(y_true, dim=dim, keepdim=True)

        var_pred = torch.var(y_pred, dim=dim)
        var_true = torch.var(y_true, dim=dim)

        numerator = torch.mean(torch.abs((y_pred - mu_pred) * (y_true - mu_true)), dim=dim)

        ncc = (numerator * numerator + EPS) / (var_pred * var_true + EPS)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(ncc).neg()  # sum over the batch and channel ndims
        if self.reduction == LossReduction.NONE.value:
            return ncc.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(ncc).neg()  # average over the batch and channel ndims
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
