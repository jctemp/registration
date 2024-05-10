import torch


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

    def __init__(self):
        """
        Init.
        """
        super().__init__()
        self.EPS = 1e-8

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Zero-normalized cross-correlation
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch, ...)
        """

        mu_pred = torch.mean(y_pred)
        mu_true = torch.mean(y_true)

        var_pred = torch.var(y_pred)
        var_true = torch.var(y_true)

        numerator = torch.mean((y_pred - mu_pred) * (y_true - mu_true))
        ncc = (numerator * numerator) / (var_pred * var_true + self.EPS)

        return ncc.neg()
