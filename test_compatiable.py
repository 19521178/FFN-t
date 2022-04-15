import numpy as np 
import os
import cv2 as cv
import scipy.io as sio

# video_path = 'data/Tour20/Tour20-Videos/'
# gt_path = 'data/Tour20/train/gt/'

video_path = 'data/TVSum/test/input/video/'
feat_path = 'data/TVSum/test/input/feat/'
gt_path = 'data/TVSum/test/input/gt/'

for path, subpath, files_name in os.walk(video_path):
    for file_name in files_name:
        video_name = file_name.split('.')[0]
        cap = cv.VideoCapture(os.path.join(path, file_name))
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        gt = sio.loadmat(gt_path + video_name + '_gt.mat')['gt'][0]
        
        # if num_frames != len(gt):
        #     print(os.path.join(path, file_name), num_frames, len(gt), num_frames == len(gt))
            
        num_feat = sio.loadmat(feat_path + video_name + '_alex_fc7_feat.mat')['Features'].shape[0]
        # if num_frames != len(gt) or num_frames !=num_feat or num_feat!=len(gt):
        print(os.path.join(path, file_name), num_frames, num_feat, len(gt))
        