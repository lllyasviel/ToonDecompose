import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CUBE(nn.Module):
    def __init__(self, T=100, H=1024, W=1024, C=3):
        super(CUBE, self).__init__()
        self.T = T
        self.H = H
        self.W = W
        self.C = C
        self.weight = torch.nn.Parameter(data=torch.Tensor(T, C, H, W), requires_grad=False)
        self.weight.data.uniform_(0, 1)

    def forward(self, t, h, w):
        mt = torch.clip(torch.round(t * self.T).long(), 0, self.T - 1)
        mh = torch.clip(torch.round(h * self.H).long(), 0, self.H - 1)
        mw = torch.clip(torch.round(w * self.W).long(), 0, self.W - 1)
        result = self.weight[mt, :, mh, mw]
        return result

    def step(self, t, h, w, c, a):
        mt = torch.clip(torch.round(t * self.T).long(), 0, self.T - 1)
        mh = torch.clip(torch.round(h * self.H).long(), 0, self.H - 1)
        mw = torch.clip(torch.round(w * self.W).long(), 0, self.W - 1)
        result = self.weight[mt, :, mh, mw]
        ma = torch.clip(a, 0, 1)
        result = result * (1 - ma) + c * ma
        self.weight[mt, :, mh, mw] = result.detach()
        return 0

    def step_hard(self, t, h, w, c):
        mt = torch.clip(torch.round(t * self.T).long(), 0, self.T - 1)
        mh = torch.clip(torch.round(h * self.H).long(), 0, self.H - 1)
        mw = torch.clip(torch.round(w * self.W).long(), 0, self.W - 1)
        self.weight[mt, :, mh, mw] = c.detach()
        return 0
