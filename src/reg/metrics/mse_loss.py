import torch


class MSE(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        loss = torch.mean((target - pred) ** 2)
        if self.weight is not None:
            loss *= self.weight
        return loss
