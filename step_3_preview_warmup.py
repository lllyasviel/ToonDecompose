import os
import shutil
import cv2
import config
import numpy as np

from fastwarp.api import uv_to_image


FRAMES = np.load(config.frame_path + 'all_frames.npy').astype(np.float32)
LABELS = np.load(config.working_path + 'label.npy').astype(np.float32)
homo_matrix = [np.load(config.model_path + 'homo_matrix.' + str(object_id) + '.npy') for object_id in range(config.number_of_objects)]

T, H, W, C = LABELS.shape


out = cv2.VideoWriter(config.working_path + 'preview_position.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (W * 2, H))

preview_vis_path = config.working_path + 'preview_vis/'
shutil.rmtree(preview_vis_path, ignore_errors=True)
os.makedirs(preview_vis_path, exist_ok=True)

for t in range(T):
    vis = []
    for c in range(C):
        uv = np.mgrid[:H, :W].astype(np.float32)
        uvf = np.stack([uv[1], uv[0], np.ones_like(uv[0])], axis=0).reshape((3, H * W))
        uvf = homo_matrix[c][t].dot(uvf)
        uv = uvf[0:2] / uvf[2:3]
        uv = uv.reshape((2, H, W)).transpose((1, 2, 0))
        vi = uv_to_image(uv, fk=96.0 if c == 0 else 48.0).astype(np.float32)
        vis.append(vi)
    vis = np.stack(vis, axis=2)
    lab = LABELS[t, :, :, :, None]
    result = np.concatenate([FRAMES[t], np.sum(vis * lab, axis=2)], axis=1).clip(0, 255).astype(np.uint8)
    cv2.imwrite(preview_vis_path + str(t) + '.png', result)
    out.write(result)
    print(preview_vis_path + str(t) + '.png')

out.release()
