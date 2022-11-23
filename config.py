import os
import glob
import shutil
import numpy as np
import cv2


task_name = 'violet'

input_path = './data/' + task_name + '/'
working_path = './workers/' + task_name + '/'

input_filenames = glob.glob(input_path + '*.png')
file_count = len(input_filenames)

H, W, C = cv2.imread(input_filenames[0]).shape
target_height = 256.0
target_width = float(target_height) * float(W) / float(H)
target_height = int(np.ceil(target_height / 8) * 8)
target_width = int(np.ceil(target_width / 8) * 8)

occ_length = 20

os.makedirs(working_path, exist_ok=True)

frame_path = working_path + 'frame/'
os.makedirs(frame_path, exist_ok=True)

flow_path = working_path + 'flow/'
os.makedirs(flow_path, exist_ok=True)

model_path = working_path + 'model/'
os.makedirs(model_path, exist_ok=True)

temp_path = working_path + 'temp/'
shutil.rmtree(temp_path, ignore_errors=True)
os.makedirs(temp_path, exist_ok=True)

number_of_objects = 2
number_of_frames = file_count

PALETTE = np.random.randint(low=0, high=256, size=(number_of_objects, 3)).astype(np.float32)

if number_of_objects >= 1:
    PALETTE[0] = 0

if number_of_objects >= 2:
    PALETTE[1] = np.array([255, 255, 255]).astype(np.float32)

if number_of_objects >= 3:
    PALETTE[2] = np.array([200, 40, 20]).astype(np.float32)

if number_of_objects >= 4:
    PALETTE[3] = np.array([20, 40, 200]).astype(np.float32)

key_flows = [-3, -1, +1, +3]


def key_flow_range(x):
    return [np.clip(x + i, 0, number_of_frames - 1) for i in key_flows]
