import cv2
import config
import torch
import numpy as np

from imlp.model import IMLP, PositionalEncoder
from pycube import CUBE


device = torch.device('cuda')

S = 10000
UNIT = 10000

FRAMES = np.load(config.frame_path + 'all_frames.npy').astype(np.float32)
LABELS = np.load(config.working_path + 'label.npy').astype(np.float32)
homo_matrix = [np.load(config.model_path + 'homo_matrix.' + str(object_id) + '.npy') for object_id in range(config.number_of_objects)]

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

params = []
params += [{'params': list(mlps[i].parameters())} for i in range(NL)]
params += [{'params': list(contents[i].parameters())} for i in range(NL)]
optimizer = torch.optim.Adam(params, lr=1e-4)


def write_torch_image(imgs, filename):
    np_imgs = [i.cpu().detach().numpy().clip(0, 255).astype(np.uint8) for i in imgs]
    cv2.imwrite(filename, np.concatenate(np_imgs, axis=1))
    return


def debug_log(filename):
    test_frame = np.random.randint(low=0, high=T)
    gt_bgr_output = torch.zeros((H, W, 3))

    color_outputs = [torch.zeros((H, W, 3)) for _ in range(NL)]
    mask_outputs = [torch.zeros((H, W, 3)) for _ in range(NL)]
    sprite_outputs = [torch.zeros((H, W, 3)) for _ in range(NL)]

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
                sprite_outputs[object_id][h] = sprite_content * 255.0

            for object_id in range(NL):
                feed = torch.stack([t, y, x, b, g, r], dim=1)
                for obe in range(NL):
                    if obe != object_id:
                        feed = torch.cat([feed, sprite_contents[obe]], dim=1)

                prd_bgra = mlps[object_id](feed)
                prd_bgr = prd_bgra[:, 0:3]
                prd_a = prd_bgra[:, 3:4]

                color_outputs[object_id][h] = prd_bgr * 255.0
                mask_outputs[object_id][h] = prd_a * 255.0

    write_torch_image([gt_bgr_output] + sprite_outputs + mask_outputs + color_outputs, filename)
    return


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


def save_all():
    for object_id in range(config.number_of_objects):
        torch.save(mlps[object_id].state_dict(), config.model_path + 'mlps.' + str(object_id) + '.pth')
        torch.save(contents[object_id].state_dict(), config.model_path + 'contents.' + str(object_id) + '.pth')


for object_id in range(config.number_of_objects):
    mlps[object_id].load_state_dict(torch.load(config.model_path + 'mlps.' + str(object_id) + '.pth'))
    contents[object_id].load_state_dict(torch.load(config.model_path + 'contents.' + str(object_id) + '.pth'))


for it in range(6 * UNIT):
    random_t = torch.randint(low=0, high=T, size=(S,)).long().to(device)
    random_y = torch.randint(low=0, high=H, size=(S,)).long().to(device)
    random_x = torch.randint(low=0, high=W, size=(S,)).long().to(device)

    gt_bgr = FRAMES[random_t, random_y, random_x] / 255.0
    gt_label = LABELS[random_t, random_y, random_x]

    t = random_t / float(T)
    y = random_y / float(H)
    x = random_x / float(W)

    b = gt_bgr[:, 0]
    g = gt_bgr[:, 1]
    r = gt_bgr[:, 2]

    sprite_contents = []
    uvs = []
    for object_id in range(NL):
        u, v = index_xyt2uv(random_x, random_y, random_t, homo_matrix[object_id])
        uvs.append([u, v])
        sprite_contents.append(infer_content(u, v, t, object_id))

    prd_blending = 0
    loss_memory = 0
    loss_occ = 0

    loss_optical_flow = 0

    prd_bgrs = []
    prd_as = []

    for object_id in range(NL):
        feed = torch.stack([t, y, x, b, g, r], dim=1)
        for obe in range(NL):
            if obe != object_id:
                feed = torch.cat([feed, sprite_contents[obe].detach()], dim=1)

        prd_bgra = mlps[object_id](feed)
        prd_bgr = prd_bgra[:, 0:3]
        prd_a = prd_bgra[:, 3:4]
        prd_blending += prd_bgr * torch.clip(prd_a, 0, 1)

        prd_bgrs.append(prd_bgr)
        prd_as.append(prd_a)

        gt_a = gt_label[:, object_id:object_id + 1]
        loss_optical_flow += torch.sum(torch.abs(gt_a - prd_a))

        u, v = uvs[object_id]

        if object_id == 0:
            if it < 1 * UNIT:
                loss_memory += torch.sum((torch.abs(infer_content(u, v, torch.rand(size=(S,), device=device), object_id) - gt_bgr) * gt_a))
            else:
                loss_memory += torch.sum(torch.abs(sprite_contents[object_id] - gt_bgr) * torch.clip(prd_a, 0, 1).detach())
        else:
            if it < 3 * UNIT:
                loss_memory += contents[object_id].step(t,
                                                        v + torch.randn(size=(S,), device=device) * 0.03,
                                                        u + torch.randn(size=(S,), device=device) * 0.03,
                                                        torch.clip(gt_bgr * torch.clip(prd_a, 0, 1), 0, 1),
                                                        0.5 * torch.ones_like(prd_a))
            else:
                loss_memory += contents[object_id].step_hard(t, v, u, torch.clip(gt_bgr * torch.clip(prd_a, 0, 1), 0, 1))

    loss_blending = torch.sum(torch.abs(prd_blending - gt_bgr))

    prd_sprite_contents = torch.stack(sprite_contents, dim=1)
    prd_as = torch.cat(prd_as, dim=1)

    sp_dist = prd_sprite_contents - gt_bgr[:, None, :]
    sp_dist = torch.sqrt(torch.sum(torch.square(sp_dist), dim=2))
    sp_dist_0 = sp_dist[:, 0]
    sp_dist_0[sp_dist_0 > 16.0 / 256.0] = 65536
    sp_dist[:, 0] = sp_dist_0
    weight_gt_min, weight_gt_indices = torch.min(sp_dist, dim=1)
    weight_gt = torch.nn.functional.one_hot(weight_gt_indices, config.number_of_objects).detach()

    loss_occ = torch.sum(torch.abs(weight_gt - prd_as))

    if it < UNIT:
        loss = loss_occ + loss_optical_flow
    else:
        loss = loss_occ + loss_memory + loss_optical_flow

    print('Training blending iter = ' + str(it))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 1000 == 0:
        debug_log(config.temp_path + str(it) + '.png')

    if it % UNIT == UNIT - 1:
        save_all()


save_all()

