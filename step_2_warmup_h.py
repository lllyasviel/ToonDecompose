import os
import numpy as np
import cv2
import shutil
import config

from raft.api import do_match
from fastwarp.api import flow_to_normal as flow_to_normal
from fastwarp.api import fast_warp_nnc


FRAMES = np.load(config.frame_path + 'all_frames.npy').astype(np.float32)
LABELS = np.load(config.working_path + 'label.npy').astype(np.float32)

T, H, W, C = FRAMES.shape


for object_id in range(config.number_of_objects):
    homo_matrix = np.zeros(shape=(T, 3, 3), dtype=np.float32)
    homo_matrix[0] = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)


    def homo_match(self_id, target_id):
        try:
            label_mask = LABELS[self_id, :, :, object_id]
            mask = np.zeros(shape=(H, W), dtype=np.uint8)
            mask[label_mask < 0.5] = 255
            flow = do_match(FRAMES[self_id], FRAMES[target_id], scale=1.0, iter=20)
            flow_inv = do_match(FRAMES[target_id], FRAMES[self_id], scale=1.0, iter=20)
            flow_inv = fast_warp_nnc(flow, flow_inv)
            mask[np.sum(np.square(flow + flow_inv), axis=2) > 2] = 127
            x0y0 = np.mgrid[:H, :W].astype(np.float32)
            x0y0 = np.stack([x0y0[1], x0y0[0]], axis=2)
            x0y0x1y1 = np.concatenate([x0y0, x0y0 + flow], axis=2)
            x0y0x1y1_masked = x0y0x1y1[mask == 0]
            x0y0 = x0y0x1y1_masked[:, 0:2].astype(np.float32)
            x1y1 = x0y0x1y1_masked[:, 2:4].astype(np.float32)
            transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(x0y0, x1y1)
            Hm = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
            Hm[0:2] = transformation_rigid_matrix
            status = rigid_mask
            mt = mask[x0y0[:, 1].astype(np.int32), x0y0[:, 0].astype(np.int32)]
            mt[status[:, 0] < 0.5] = 200
            mask[x0y0[:, 1].astype(np.int32), x0y0[:, 0].astype(np.int32)] = mt
            k_area = (float(np.where(mask == 0)[0].shape[0]) + 0.1) / (
                        float(np.where(label_mask < 0.5)[0].shape[0]) + 0.1)
            if np.linalg.matrix_rank(Hm) < 3:
                print('Homo rank < 3!')
                raise Exception('Homo rank < 3!')
        except:
            print('Homo match failed!')
            Hm = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
            k_area = 0.0
            flow = np.zeros(shape=(H, W, 2), dtype=np.float32)
            mask = np.zeros(shape=(H, W), dtype=np.uint8) + 255
        return Hm, k_area, flow, mask


    position_path = config.working_path + 'position.' + str(object_id) + '/'
    shutil.rmtree(position_path, ignore_errors=True)
    os.makedirs(position_path, exist_ok=True)

    target_index = 0
    index = 1
    while index < T:
        Hm, k_area, flow, mask = homo_match(index, target_index)
        logger = 'From ' + str(index) + ' to ' + str(target_index) + ' - K = ' + str(k_area)
        if index == target_index + 1:
            logger += ' Last!'
            # if k_area <= 0.33333:
            #     Hm = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
            #     logger += ' Identity!'
            k_area = 1.0
        if k_area > 0.9:
            logger += ' Success!'
            flag = 'F' + str(index) + 'T' + str(target_index)
            cv2.imwrite(position_path + flag + '.flow.png', flow_to_normal(flow, uint8=True))
            cv2.imwrite(position_path + flag + '.mask.png', mask)
            np.save(position_path + flag + '.h', Hm)
            homo_matrix[index] = homo_matrix[target_index].dot(Hm)
            index += 1
        else:
            logger += ' Fail!'
            target_index = index - 1
        print(logger)

    np.save(config.model_path + 'homo_matrix_raw.' + str(object_id), homo_matrix)
    print(config.model_path + 'homo_matrix_raw.' + str(object_id) + '.npy')

    UVMIN = 1e64
    UVMAX = -1e64
    homos = homo_matrix.copy()
    for t in range(T):
        uvf = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], dtype=np.float32).transpose((1, 0))
        uvf = homos[t].dot(uvf)
        uv = uvf[0:2] / uvf[2:3]
        UVMIN = min(UVMIN, np.min(uv))
        UVMAX = max(UVMAX, np.max(uv))

    normH = np.array([1.0, 0, -UVMIN, 0, 1.0, -UVMIN, 0, 0, UVMAX - UVMIN]).reshape(3, 3).astype(np.float32)
    for t in range(T):
        homos[t] = normH.dot(homos[t])

    np.save(config.model_path + 'homo_matrix.' + str(object_id), homos)
    print(config.model_path + 'homo_matrix.' + str(object_id) + '.npy')
