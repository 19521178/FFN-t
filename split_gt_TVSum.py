import scipy.io as sio
import numpy as np
import h5py
import cv2 as cv 
import os
from itertools import islice
import math

all_path = 'data/TVSum/train/gt/ydata-tvsum50.mat'
save_path = 'data/TVSum/train/gt/'
suffixes_savepath = '_gt.mat'
videos_path = 'data/TVSum/train/video/'

hf = h5py.File(all_path, 'r')

def convert_ints_2_str(array):
    array = array.reshape(-1)
    list_char = [chr(x) for x in array]
    return "".join(list_char)
ref_videos = hf['tvsum50/video']
shape = ref_videos.shape
value_videos = [convert_ints_2_str(np.array(hf[ref_videos[i][0]])) for i in range(shape[0])]

def make_chunks(data, SIZE):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield [k for k in islice(it, SIZE)]
ref_gt = hf['tvsum50/gt_score']
shape = ref_gt.shape
value_gt = []
for index_video in range(shape[0]):
    gt = np.array(hf[ref_gt[index_video][0]])[0]
    cap = cv.VideoCapture(videos_path+value_videos[index_video]+'.mp4')
    fps = math.ceil(cap.get(5))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if len(gt)<num_frames:
        last_value = gt[-1]
        fill_array = [last_value for _ in range(num_frames-len(gt))]
        gt = np.concatenate([gt, np.array(fill_array)])
        # gt = value_gt[index_video][0]
        # gt = np.concatenate([gt, np.array(fill_array)])
    value_gt.append([gt])
# value_gt = [np.array(hf[ref_gt[i][0]]) for i in range(shape[0])]


for index_video in range(len(value_videos)):
    
    gt = value_gt[index_video][0] # value_gt[index_video]: [[2.9, ...1.3]]
    cap = cv.VideoCapture(videos_path+value_videos[index_video]+'.mp4')
    fps = math.ceil(cap.get(5))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    # a shot durate 2s
    num_shots = int(num_frames/(fps*2))
    shots_gt_score = np.array([sample for sample in make_chunks(gt, fps*2)])

    shots_avg_score = [np.mean(shot) for shot in shots_gt_score]
    shots_avg_score = np.argsort(shots_avg_score) # ascade sort
    
    for index_shot in shots_avg_score[-int(num_shots/5):]:
        shots_gt_score[index_shot] = np.ones(len(shots_gt_score[index_shot]))
    for index_shot in shots_avg_score[:-int(num_shots/5)]:
        shots_gt_score[index_shot] = np.zeros(len(shots_gt_score[index_shot]))

    gt = np.concatenate(shots_gt_score)
    value_gt[index_video][0] = gt
    
    # true_pos_hotkey = [1 if (gt!=predict) else 0 for gt, predict in zip(value_gt[index_video][0], gt)]
    # true_pos_sum = sum(true_pos_hotkey)
    # print(true_pos_sum)
    # exit()
    




for video_name, gt in zip(value_videos, value_gt):
    sio.savemat(save_path + video_name + suffixes_savepath, {'gt': gt})