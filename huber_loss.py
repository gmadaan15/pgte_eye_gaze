import torch
import torch.nn as nn
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        abs_error = torch.abs(input - target)
        quadratic = torch.min(abs_error, torch.tensor(self.delta).to(abs_error.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()