import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoder:
    def __init__(self, frequency_number=10):
        self.b = (2.0 ** torch.arange(start=0, end=frequency_number, step=1).float()) * 0.5 * np.pi

    def __call__(self, in_tensor):
        if self.b.device != in_tensor.device:
            self.b = self.b.to(in_tensor.device)
        proj = torch.einsum('ij, k -> ijk', in_tensor, self.b)
        sin = torch.sin(proj)
        cos = torch.cos(proj)
        cat = torch.cat([sin, cos], dim=1)
        N, C, F = cat.shape
        output = cat.contiguous().view(N, C * F)
        return output


class IMLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=2, hidden_dim=128, num_layers=4):
        super(IMLP, self).__init__()

        self.hidden = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                dim = input_dim
            else:
                dim = hidden_dim

            if i == num_layers - 1:
                self.hidden.append(nn.Linear(dim, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(dim, hidden_dim, bias=True))

        self.num_layers = num_layers

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.leaky_relu(x, negative_slope=0.2)
            x = layer(x)
        return x

