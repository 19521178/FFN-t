import scipy.io as sio
import numpy as np
import pandas as pd
import cv2 as cv 
import os
from itertools import islice
import math

gt_path = 'data/Tour20/Tour20-UserSummaries/'
save_path = 'data/Tour20/train/gt/'
suffixes_savepath = '_gt.mat'
videos_path = 'data/Tour20/Tour20-Videos/'
# info_path = 'data/Tour20/Tour20-Info.xlsx'
shots_info_path = 'data/Tour20/Tour20-Segmentation/'


# Get shot gt
def expand_element_dict(dict, key, value):
    if (key not in dict.keys()):
        dict[key] = [value]
    else:
        if value not in dict[key]:
            dict[key].append(value)
dict_shot_gt = {}
for path, subpaths_name, files_name in os.walk(gt_path):
    for file_name in files_name:
        file_path = os.path.join(path, file_name)
        df = pd.read_excel(file_path)
        keys = df.columns[-2:] # import_shot_index vs video_index
        # print(file_path, df[keys[0]].dropna().index.value_counts)
        for row_index in df[keys[0]].dropna().index.values:
            # video_index = int(df[keys[1]][row_index] - 1)
            shot_index = int(df[keys[0]][row_index]) - 1
            video_index = df[keys[1]][row_index] - 1
            video_name = df.columns[video_index]
            
            expand_element_dict(dict_shot_gt, video_name, shot_index)

# Get shot indices
dict_shot_indices = {}
def make_chunks(data, SIZE):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield [k for k in islice(it, SIZE)]

for path, subpaths_name, files_name in os.walk(shots_info_path):
    for file_name in files_name:
        video_name = file_name.split('_')[-1]
        category = video_name[:2]
        cap = cv.VideoCapture(os.path.join(videos_path, category, video_name+'.mp4'))
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r') as f:
            rows = [row[:-1] for row in f.readlines()]  # remove \n
            shots = [start_end for start_end in make_chunks(rows, 2)]
            for i in range(len(shots)):
                shot = shots[i]
                shots[i] = [int(shot[0]) - 1, int(shot[1]) - 1]
            # shots[-1][-1] = max(shots[-1][-1], num_frames-1)
            shots[-1][-1] = num_frames-1
            dict_shot_indices[video_name] = shots

# Update 1 for frames in importance shot
# importance shot is the shot tagged 1 by any judge
videos_name = sorted(dict_shot_gt.keys())
for video_name in videos_name:
    num_frame = int(dict_shot_indices[video_name][-1][-1]) + 1
    gt = np.zeros((num_frame,))
    importance_shots = dict_shot_gt[video_name]
    for shot_index in importance_shots:
        start, end = dict_shot_indices[video_name][int(shot_index)]
        for frame_index in range(int(start), int(end)+1, 1):
            gt[frame_index] = 1
    sio.savemat(save_path + video_name + suffixes_savepath, {'gt': gt})
