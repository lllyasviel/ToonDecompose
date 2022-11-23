import os
import shutil
import cv2
import config
import torch
import numpy as np

from imlp.model import IMLP, PositionalEncoder
from fastwarp.api import flow_to_normal as flow_to_normal


device = torch.device('cuda')

S = 10000
UNIT = 10000

FLOWS = np.load(config.flow_path + 'all_flows.npy')
FRAMES = np.load(config.frame_path + 'all_frames.npy')

PALETTE = torch.from_numpy(config.PALETTE).to(device)

FN, TN, H, W, _ = FLOWS.shape

FLOWS = torch.from_numpy(FLOWS).to(device)
FRAMES = torch.from_numpy(FRAMES).to(device)
LABELS = torch.zeros(size=(FN, TN, H, W), dtype=torch.int) - 1

pscoder = PositionalEncoder(frequency_number=6)
mlp_hs = [IMLP(input_dim=24, output_dim=6, hidden_dim=256, num_layers=6).to(device)] + \
         [IMLP(input_dim=26, output_dim=2, hidden_dim=256, num_layers=12).to(device) for _ in range(config.number_of_objects - 1)]
mlp_m = IMLP(input_dim=6, output_dim=config.number_of_objects, hidden_dim=256, num_layers=8).to(device)


def transform(ip_x, ip_y, hs):
    h_a = hs[:, 0]
    h_b = hs[:, 1]
    h_c = hs[:, 2]
    h_d = hs[:, 3]
    h_e = hs[:, 4]
    h_f = hs[:, 5]
    ox = h_a * ip_x + h_b * ip_y + h_c
    oy = h_d * ip_x + h_e * ip_y + h_f
    return ox, oy


def predict_flow(f, t, x, y, i):
    if i == 0:
        hs = mlp_hs[i](pscoder(torch.stack([f, t], dim=1)))
        ox, oy = transform(x, y, hs)
        flow_x = ox - x
        flow_y = oy - y
        flow = torch.stack([flow_x, flow_y], dim=1)
    else:
        feed_ft = torch.stack([f, t], dim=1)
        feed_xy = torch.stack([x, y], dim=1)
        feed_ft = pscoder(feed_ft)
        feeds = torch.cat([feed_ft, feed_xy], dim=1)
        flow = mlp_hs[i](feeds)
    return flow


def debug_preview(object_id, filename):
    test_frame_f = np.random.randint(low=0, high=FN)
    test_frame_t = np.random.randint(low=0, high=TN)
    gt_flow_output = torch.zeros((H, W, 2))
    gt_bgr_output = torch.zeros((H, W, 3))
    flow_output = torch.zeros((H, W, 2))
    label_output = torch.zeros((H, W))

    with torch.no_grad():
        lins = torch.linspace(0, W - 1, W).long()
        for h in range(H):
            random_f = test_frame_f * torch.ones_like(lins).to(device)
            random_t = test_frame_t * torch.ones_like(lins).to(device)
            random_y = h * torch.ones_like(lins).to(device)
            random_x = lins.to(device)

            gt_flow = FLOWS[random_f, random_t, random_y, random_x]
            gt_bgr = FRAMES[random_f, random_y, random_x]
            gt_label = LABELS[random_f, random_t, random_y, random_x]

            gt_flow_output[h] = gt_flow
            gt_bgr_output[h] = gt_bgr

            f = random_f / float(FN)
            t = torch.clip(random_f - 3 + random_t * 2, 0, FN - 1) / float(FN)
            y = random_y / float(H)
            x = random_x / float(W)

            flow_output[h] = predict_flow(f, t, x, y, object_id)

            label_output[h] = gt_label

    gt_flow_output = gt_flow_output.detach().numpy()
    gt_bgr_output = gt_bgr_output.detach().numpy()
    flow_output = flow_output.detach().numpy()
    label_output = label_output.detach().numpy()

    gt_flow_output = flow_to_normal(gt_flow_output, uint8=True)
    gt_bgr_output = gt_bgr_output.clip(0, 255).astype(np.uint8)
    flow_output = flow_to_normal(flow_output, uint8=True)

    flow_output[label_output > -1] = 0

    cv2.imwrite(filename, np.concatenate([gt_bgr_output, gt_flow_output, flow_output], axis=1))
    return


def debug_blending(filename):
    test_frame_f = np.random.randint(low=0, high=FN)
    test_frame_t = np.random.randint(low=0, high=TN)
    gt_flow_output = torch.zeros((H, W, 2))
    gt_bgr_output = torch.zeros((H, W, 3))
    flow_output = torch.zeros((H, W, 2))
    mask_output = torch.zeros((H, W, 3))

    with torch.no_grad():
        lins = torch.linspace(0, W - 1, W).long()
        for h in range(H):
            random_f = test_frame_f * torch.ones_like(lins).to(device)
            random_t = test_frame_t * torch.ones_like(lins).to(device)
            random_y = h * torch.ones_like(lins).to(device)
            random_x = lins.to(device)

            gt_flow = FLOWS[random_f, random_t, random_y, random_x]
            gt_bgr = FRAMES[random_f, random_y, random_x]

            gt_flow_output[h] = gt_flow
            gt_bgr_output[h] = gt_bgr

            f = random_f / float(FN)
            t = torch.clip(random_f - 3 + random_t * 2, 0, FN - 1) / float(FN)
            y = random_y / float(H)
            x = random_x / float(W)

            b = gt_bgr[:, 0] / 255.0
            g = gt_bgr[:, 1] / 255.0
            r = gt_bgr[:, 2] / 255.0

            fxybgr = torch.stack([f, x, y, b, g, r], dim=1)

            weights = mlp_m(fxybgr)
            flows = torch.stack([predict_flow(f, t, x, y, ik) for ik in range(config.number_of_objects)], 1)

            weighted_flow = torch.sum(weights[:, :, None] * flows, dim=1)
            flow_output[h] = weighted_flow

            weighted_vis = torch.sum(weights[:, :, None] * PALETTE[None, :, :], dim=1)
            mask_output[h] = weighted_vis

    gt_flow_output = gt_flow_output.detach().numpy()
    gt_bgr_output = gt_bgr_output.detach().numpy()
    flow_output = flow_output.detach().numpy()
    mask_output = mask_output.detach().numpy()

    gt_flow_output = flow_to_normal(gt_flow_output, uint8=True)
    gt_bgr_output = gt_bgr_output.clip(0, 255).astype(np.uint8)
    flow_output = flow_to_normal(flow_output, uint8=True)
    mask_output = mask_output.clip(0, 255).astype(np.uint8)

    cv2.imwrite(filename, np.concatenate([gt_bgr_output, gt_flow_output, flow_output, mask_output], axis=1))
    return


for object_id in range(config.number_of_objects):
    # continue
    optimizer = torch.optim.Adam([{'params': list(mlp_hs[object_id].parameters())}], lr=1e-4)

    insider_fs, insider_ts, insider_ys, insider_xs = torch.where(LABELS < 0)
    insider_count = insider_ts.shape[0]
    insider_fs = insider_fs.long()
    insider_ts = insider_ts.long()
    insider_ys = insider_ys.long()
    insider_xs = insider_xs.long()

    print('Object ' + str(object_id) + ' training start.')

    for it in range(4 * UNIT):
        random_i = torch.randint(low=0, high=insider_count, size=(S,)).long()

        random_f = insider_fs[random_i].to(device)
        random_t = insider_ts[random_i].to(device)
        random_y = insider_ys[random_i].to(device)
        random_x = insider_xs[random_i].to(device)

        gt_flow = FLOWS[random_f, random_t, random_y, random_x]

        f = random_f / float(FN)
        t = torch.clip(random_f - 3 + random_t * 2, 0, FN - 1) / float(FN)
        y = random_y / float(H)
        x = random_x / float(W)

        flow = predict_flow(f, t, x, y, object_id)

        dist = torch.sqrt(torch.sum(torch.square(flow - gt_flow), dim=1))

        if object_id == 0:
            threshold = 65535 * 1024
            if it > 1 * UNIT:
                threshold = 32
            if it > 2 * UNIT:
                threshold = 16
            if it > 3 * UNIT:
                threshold = 8
            loss = torch.sum(dist[dist < threshold])
        else:
            loss = torch.sum(dist)

        iter_flag = str(object_id) + '.' + str(it)

        print('Training iter = ' + iter_flag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            debug_preview(object_id, config.temp_path + iter_flag + '.png')

    print('Object ' + str(object_id) + ' saving.')

    torch.save(mlp_hs[object_id].state_dict(), config.model_path + 'mlp_hs.' + str(object_id) + '.pth')

    object_path = config.working_path + 'object.' + str(object_id) + '/'
    shutil.rmtree(object_path, ignore_errors=True)
    os.makedirs(object_path, exist_ok=True)

    for f_index in range(FN):
        for t_index in range(TN):
            gt_flow = FLOWS[f_index, t_index]
            prd_flow = torch.zeros((H, W, 2)).to(device)
            prd_inliner = torch.zeros((H, W)).to(device)

            with torch.no_grad():
                lins = torch.linspace(0, W - 1, W).long()
                for h in range(H):
                    random_f = f_index * torch.ones_like(lins).to(device)
                    random_t = t_index * torch.ones_like(lins).to(device)
                    random_y = h * torch.ones_like(lins).to(device)
                    random_x = lins.to(device)
                    f = random_f / float(FN)
                    t = torch.clip(random_f - 3 + random_t * 2, 0, FN - 1) / float(FN)
                    y = random_y / float(H)
                    x = random_x / float(W)
                    flow = predict_flow(f, t, x, y, object_id)
                    prd_flow[h] = flow

            dist = torch.sqrt(torch.sum(torch.square(gt_flow - prd_flow), dim=2))
            prd_inliner[dist < 2] = 1.0

            prd_flow = prd_flow.cpu().detach().numpy()
            prd_inliner = prd_inliner.cpu().detach().numpy()

            flag = str(f_index) + 'T' + str(np.clip(f_index - 3 + 2 * t_index, 0, config.number_of_frames - 1))
            np.save(object_path + flag, prd_flow)
            cv2.imwrite(object_path + flag + '.png', flow_to_normal(prd_flow, uint8=True))
            cv2.imwrite(object_path + flag + '.mask.png', (prd_inliner * 255.0).clip(0, 255).astype(np.uint8))

            LABELS[f_index, t_index][prd_inliner > 0.5] = object_id

            print('Flow Computed: ' + flag)

    print('Object solved: ' + str(object_id))


print('Begin solving blending')

# for object_id in range(config.number_of_objects):
#     mlp_hs[object_id].load_state_dict(torch.load(config.model_path + 'mlp_hs.' + str(object_id) + '.pth'))

params = []
params += [{'params': list(mlp_hs[i].parameters())} for i in range(config.number_of_objects)]
params += [{'params': list(mlp_m.parameters())}]
optimizer = torch.optim.Adam(params, lr=1e-4)


for it in range(4 * UNIT):
    random_f = torch.randint(low=0, high=FN, size=(S,)).long().to(device)
    random_t = torch.randint(low=0, high=TN, size=(S,)).long().to(device)
    random_y = torch.randint(low=0, high=H, size=(S,)).long().to(device)
    random_x = torch.randint(low=0, high=W, size=(S,)).long().to(device)

    gt_flow = FLOWS[random_f, random_t, random_y, random_x]
    gt_bgr = FRAMES[random_f, random_y, random_x]

    f = random_f / float(FN)
    t = torch.clip(random_f - 3 + random_t * 2, 0, FN - 1) / float(FN)
    y = random_y / float(H)
    x = random_x / float(W)

    b = gt_bgr[:, 0] / 255.0
    g = gt_bgr[:, 1] / 255.0
    r = gt_bgr[:, 2] / 255.0

    fxybgr = torch.stack([f, x, y, b, g, r], dim=1)

    weights = mlp_m(fxybgr)
    flows = torch.stack([predict_flow(f, t, x, y, ik) for ik in range(config.number_of_objects)], 1)

    sp_flow_dist = flows - gt_flow[:, None, :]
    sp_flow_dist = torch.sqrt(torch.sum(torch.square(sp_flow_dist), dim=2))
    weight_gt_min, weight_gt_indices = torch.min(sp_flow_dist, dim=1)
    weight_gt = torch.nn.functional.one_hot(weight_gt_indices, config.number_of_objects).detach()

    dispatcher = torch.zeros_like(weight_gt)[:, 0:1]
    dispatcher[torch.std(sp_flow_dist, dim=1)[:, None] > 4] = 1

    loss_weight = torch.abs(weight_gt - weights) * dispatcher
    loss_flow = torch.sqrt(torch.sum(torch.square(torch.sum(weight_gt[:, :, None] * flows, dim=1) - gt_flow), dim=1))

    loss = torch.sum(loss_weight) + torch.sum(loss_flow)

    print('Training blending iter = ' + str(it))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 1000 == 0:
        debug_blending(config.temp_path + 'blending.' + str(it) + '.jpg')


for object_id in range(config.number_of_objects):
    torch.save(mlp_hs[object_id].state_dict(), config.model_path + 'mlp_hs.' + str(object_id) + '.pth')

torch.save(mlp_m.state_dict(), config.model_path + 'mlp_m.pth')


preview_label_path = config.working_path + 'preview_labels/'
shutil.rmtree(preview_label_path, ignore_errors=True)
os.makedirs(preview_label_path, exist_ok=True)


LABELS = []
for frame_index in range(FN):
    label_output = torch.zeros((H, W, config.number_of_objects))
    with torch.no_grad():
        lins = torch.linspace(0, W - 1, W).long()
        for h in range(H):
            random_f = frame_index * torch.ones_like(lins).to(device)
            random_y = h * torch.ones_like(lins).to(device)
            random_x = lins.to(device)

            gt_bgr = FRAMES[random_f, random_y, random_x]

            f = random_f / float(FN)
            y = random_y / float(H)
            x = random_x / float(W)

            b = gt_bgr[:, 0] / 255.0
            g = gt_bgr[:, 1] / 255.0
            r = gt_bgr[:, 2] / 255.0

            fxybgr = torch.stack([f, x, y, b, g, r], dim=1)

            labels = torch.softmax(mlp_m(fxybgr), dim=1)

            label_max, label_indices = torch.max(labels, dim=1)
            labels = torch.nn.functional.one_hot(label_indices, config.number_of_objects)

            label_output[h] = labels
    label_output = label_output.detach().numpy()

    label_preview = np.sum(label_output[:, :, :, None] * config.PALETTE[None, None, :, :], axis=2)
    label_preview = label_preview.clip(0, 255).astype(np.uint8)

    cv2.imwrite(preview_label_path + str(frame_index) + '.png', label_preview)
    print(preview_label_path + str(frame_index) + '.png')

    LABELS.append(label_output)


LABELS = np.stack(LABELS, axis=0)
np.save(config.working_path + 'label', LABELS)
print('OK.')
