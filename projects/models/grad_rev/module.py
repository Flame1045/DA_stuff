from .functional import revgrad
import torch
from torch import nn

class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()
        # self.alpha = torch.tensor(alpha, requires_grad=True)
        # self.alpha = alpha

    def forward(self, x, alpha):
        return revgrad(x, alpha)