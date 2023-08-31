USE_ALPHA_OUTPUTS = False

import shutil
import os
import cv2
import config
import torch
import numpy as np

from imlp.model import IMLP, PositionalEncoder
from fastwarp.api import uv_to_image
from pycube import CUBE


device = torch.device('cuda')

S = 10000
UNIT = 10000

FRAMES = np.load(config.frame_path + 'all_frames.npy').astype(np.float32)
LABELS = np.load(config.working_path + 'label.npy').astype(np.float32)
homo_matrix = [np.load(config.model_path + 'homo_matrix.' + str(object_id) + '.npy') for object_id in range(config.number_of_objects)]
homo_matrix_np = homo_matrix

T, H, W, NL = LABELS.shape

FRAMES = torch.from_numpy(FRAMES).to(device)
LABELS = torch.from_numpy(LABELS).to(device)
homo_matrix = [torch.from_numpy(x).to(device) for x in homo_matrix]
homo_matrix_inv = [torch.inverse(x) for x in homo_matrix]
PALETTE = torch.from_numpy(config.PALETTE).to(device)

pscoder = PositionalEncoder(frequency_number=12)

mlps = [IMLP(input_dim=3 + 3 * NL, output_dim=4, hidden_dim=256, num_layers=8).to(device) for _ in range(NL)]
contents = [IMLP(input_dim=3 * 12 * 2, output_dim=3, hidden_dim=256, num_layers=10).to(device)]
contents += [CUBE(T=T, H=512, W=512, C=3).to(device) for _ in range(NL - 1)]


for object_id in range(config.number_of_objects):
    mlps[object_id].load_state_dict(torch.load(config.model_path + 'mlps.' + str(object_id) + '.pth'))
    contents[object_id].load_state_dict(torch.load(config.model_path + 'contents.' + str(object_id) + '.pth'))


def np_torch_image(img):
    return img.cpu().detach().numpy().clip(0, 255).astype(np.uint8)


def index_xyt2uv(x, y, t, Hm):
    muvf = torch.stack([x.float(), y.float(), torch.ones_like(x, device=x.device).float()], dim=1)
    muvf = torch.bmm(Hm[t], muvf[:, :, None])
    u = muvf[:, 0, 0] / muvf[:, 2, 0]
    v = muvf[:, 1, 0] / muvf[:, 2, 0]
    return u, v


def uvt2a(u, v, t, obj_id):
    with torch.no_grad():
        mt = torch.clip(torch.round(t * T).long(), 0, T - 1)
        mu = u
        mv = v
        mf = torch.ones_like(mu, device=mu.device)
        hs = homo_matrix_inv[obj_id][mt]
        muvf = torch.stack([mu, mv, mf], dim=1)[:, :, None]
        muvf = torch.bmm(hs, muvf)
        ox = torch.round(muvf[:, 0, 0] / muvf[:, 2, 0]).long()
        oy = torch.round(muvf[:, 1, 0] / muvf[:, 2, 0]).long()
        mkey = LABELS[:, :, :, obj_id][torch.clip(mt, 0, T - 1), torch.clip(oy, 0, H - 1), torch.clip(ox, 0, W - 1)]
        result = torch.ones_like(t, device=mt.device).float()
        result[mkey < 0.5] = 0
        result[mt < 0] = 0
        result[oy < 0] = 0
        result[ox < 0] = 0
        result[mt > T - 1] = 0
        result[oy > H - 1] = 0
        result[ox > W - 1] = 0
        return result


def infer_content(u, v, t, obj_id):
    if obj_id == 0:
        uvt = torch.stack([u, v, t], dim=1)
        result = contents[obj_id](pscoder(uvt))
    else:
        result = contents[obj_id](t, v, u)
    return result


out = cv2.VideoWriter(config.working_path + 'vis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (W * NL, H * 2))

vis_path = config.working_path + 'vis/'
shutil.rmtree(vis_path, ignore_errors=True)
os.makedirs(vis_path, exist_ok=True)


for test_frame in range(T):
    frame = np.zeros(shape=(H * 2, W * NL, 3), dtype=np.uint8)

    gt_bgr_output = torch.zeros((H, W, 3))
    mask_output = torch.zeros((H, W, NL))
    sprite_outputs = torch.zeros((H, W, NL, 3))
    with torch.no_grad():
        lins = torch.linspace(0, W - 1, W).long()
        for h in range(H):
            random_t = test_frame * torch.ones_like(lins).to(device)
            random_y = h * torch.ones_like(lins).to(device)
            random_x = lins.to(device)

            gt_bgr = FRAMES[random_t, random_y, random_x] / 255.0

            gt_bgr_output[h] = gt_bgr * 255.0

            t = random_t / float(T)
            y = random_y / float(H)
            x = random_x / float(W)

            b = gt_bgr[:, 0]
            g = gt_bgr[:, 1]
            r = gt_bgr[:, 2]

            sprite_contents = []
            for object_id in range(NL):
                u, v = index_xyt2uv(random_x, random_y, random_t, homo_matrix[object_id])
                sprite_content = infer_content(u, v, t, object_id)
                sprite_contents.append(sprite_content)
                sprite_outputs[h, :, object_id, :] = sprite_content * 255.0

            for object_id in range(NL):
                feed = torch.stack([t, y, x, b, g, r], dim=1)
                for obe in range(NL):
                    if obe != object_id:
                        feed = torch.cat([feed, sprite_contents[obe]], dim=1)

                prd_bgra = mlps[object_id](feed)
                prd_bgr = prd_bgra[:, 0:3]
                prd_a = prd_bgra[:, 3]

                mask_output[h, :, object_id] = prd_a
    gt_bgr_output = gt_bgr_output.cpu().detach().numpy()
    mask_output = mask_output.cpu().detach().numpy()
    sprite_outputs = sprite_outputs.cpu().detach().numpy()
    vis = []
    for c in range(NL):
        uv = np.mgrid[:H, :W].astype(np.float32)
        uvf = np.stack([uv[1], uv[0], np.ones_like(uv[0])], axis=0).reshape((3, H * W))
        uvf = homo_matrix_np[c][test_frame].dot(uvf)
        uv = uvf[0:2] / uvf[2:3]
        uv = uv.reshape((2, H, W)).transpose((1, 2, 0))
        vi = uv_to_image(uv, fk=96.0 if c == 0 else 48.0).astype(np.float32)
        vis.append(vi)
    vis = np.stack(vis, axis=2)
    vis = np.sum(vis * mask_output[:, :, :, None], axis=2)
    cv2.imwrite(vis_path + str(test_frame) + '.img.png', gt_bgr_output.clip(0, 255).astype(np.float32))
    cv2.imwrite(vis_path + str(test_frame) + '.homo.png', vis.clip(0, 255).astype(np.float32))
    frame[0 * H:1 * H, 0 * W:1 * W, :] = gt_bgr_output
    frame[0 * H:1 * H, 1 * W:2 * W, :] = vis
    for c in range(NL):
        if c == 0:
            result = sprite_outputs[:, :, c, :]
        else:
            mask = mask_output[:, :, c:c+1]
            result = gt_bgr_output * mask + 48 * (1 - mask)
        if not USE_ALPHA_OUTPUTS:
            cv2.imwrite(vis_path + str(test_frame) + '.obj.' + str(c) + '.png', result.clip(0, 255).astype(np.float32))
        else:
            if c == 0:
                alpha_output = result
            else:
                mask = mask_output[:, :, c:c + 1] * 255
                color = gt_bgr_output
                alpha_output = np.concatenate([color, mask], axis=2)
            cv2.imwrite(vis_path + str(test_frame) + '.obj.' + str(c) + '.png', alpha_output.clip(0, 255).astype(np.float32))
        frame[1 * H:2 * H, c * W:c * W + W, :] = result
    print(test_frame)
    out.write(frame.clip(0, 255).astype(np.uint8))

out.release()
