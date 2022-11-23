import numpy as np


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    s = x1-x
    t = y1-y

    wa = s * t
    wb = s * (1 - t)
    wc = (1 - s) * t
    wd = (1 - s) * (1 - t)

    wa = wa[:, None]
    wb = wb[:, None]
    wc = wc[:, None]
    wd = wd[:, None]

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def get_points(H, W):
    gird = np.mgrid[:W, :H].transpose((2, 1, 0))
    return gird.astype(np.float32)


def interpolate_batch(position_batch, array):
    x = position_batch[:, 0]
    y = position_batch[:, 1]
    r = bilinear_interpolate(array, x, y)
    return r


def fast_warp_nn(flow, img):
    H, W = img.shape
    pos = (get_points(H, W) + flow).reshape(H * W, 2)
    x = np.round(pos[:, 0]).clip(0, W - 1).astype(np.int32)
    y = np.round(pos[:, 1]).clip(0, H - 1).astype(np.int32)
    return img[y, x].reshape(H, W)


def fast_warp_nnc(flow, img):
    H, W, C = img.shape
    pos = (get_points(H, W) + flow).reshape(H * W, 2)
    x = np.round(pos[:, 0]).clip(0, W - 1).astype(np.int32)
    y = np.round(pos[:, 1]).clip(0, H - 1).astype(np.int32)
    return img[y, x].reshape(H, W, C)


def flow_to_normal(flow, uint8=False):
    x = flow[:, :, 0] / 32.0
    y = flow[:, :, 1] / 32.0
    d = (x ** 2 + y ** 2) ** 0.5
    d = np.maximum(d, 1)
    x /= d
    y /= d
    z = (1 - x ** 2 - y ** 2).clip(0, 1) ** 0.5
    normal = np.stack([z, x, y], axis=2)
    if uint8:
        normal = normal * 127.5 + 127.5
        normal = normal.clip(0, 255).astype(np.uint8)
    return normal


def flow_to_normal_abs(flow, uint8=False):
    r = flow[:, :, 0] + 127
    g = flow[:, :, 1] + 127
    b = np.zeros_like(r) + 127
    normal = np.stack([b, g, r], axis=2)
    if uint8:
        normal = normal.clip(0, 255).astype(np.uint8)
    return normal


def uv_to_image(uv, fk=64.0):
    img = uv.astype(np.float32)
    f = np.sin(img * fk)
    r = np.concatenate([f, np.zeros_like(f)], axis=2)[:, :, 0:3]
    return (r * 128.0 + 128.0).clip(0, 255).astype(np.uint8)
