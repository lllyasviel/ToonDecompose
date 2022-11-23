import torch
import cv2
import numpy as np

from .raft import RAFT


DEVICE = 'cuda'
model = torch.nn.DataParallel(RAFT())
model.load_state_dict(torch.load('./raft/models/raft-sintel.pth'))

model = model.module
model.to(DEVICE)
model.eval()


def load_image(x):
    img = x[:, :, ::-1].copy()
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def do_match(x, y, scale, iter):
    with torch.no_grad():
        ix = load_image(x)
        iy = load_image(y)
        flow = model(ix, iy, iters=iter, test_mode=True, flow_init=None, scale=scale)[1][0].permute(1, 2, 0).cpu().numpy()
    return flow

