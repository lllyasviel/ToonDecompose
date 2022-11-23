import numpy as np
import cv2
import config


all_frames = []
for i, filename in enumerate(config.input_filenames):
    frame = cv2.imread(filename)
    frame = cv2.resize(frame, (config.target_width, config.target_height), interpolation=cv2.INTER_AREA)
    all_frames.append(frame)
    cv2.imwrite(config.frame_path + str(i) + '.png', frame)
all_frames = np.stack(all_frames, axis=0)
np.save(config.frame_path + 'all_frames', all_frames)
print('All Frames Saved: ' + config.frame_path + 'all_frames.npy')


from raft.api import do_match
from fastwarp.api import flow_to_normal as flow_to_normal


def compute_flow(f, t):
    frame = cv2.imread(config.frame_path + str(f) + '.png')
    target_frame = cv2.imread(config.frame_path + str(t) + '.png')
    flow = do_match(frame, target_frame, scale=1.0, iter=20)
    flag = str(f) + 'T' + str(t)
    np.save(config.flow_path + flag, flow)
    cv2.imwrite(config.flow_path + flag + '.png', flow_to_normal(flow, uint8=True))
    print('Flow Computed: ' + flag)
    return flow


all_flows = []
for i in range(config.file_count):
    flows = []
    for t in config.key_flow_range(i):
        flow = compute_flow(i, t)
        flows.append(flow)
    flows = np.stack(flows, axis=0)
    all_flows.append(flows)
all_flows = np.stack(all_flows, axis=0)

np.save(config.flow_path + 'all_flows', all_flows)
print('All Flows Saved: ' + config.flow_path + 'all_flows.npy')
