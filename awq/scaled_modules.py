# custom pytorch modules to apply AWQ weight scaling before forward pass of another module
import torch.nn as nn

class ScaledModule(nn.Module):

    def __init__(self, scales, module):
        super().__init__()
        self.scales = nn.Parameter(scales.data)
        self.module = module

    def forward(self, x, *args, **kwargs):
        x = x / self.scales.view(1, 1, -1).to(x.device)
        return self.module(x, *args, **kwargs)
