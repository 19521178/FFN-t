# select a video for an episode. can be modified to randomly select.
import scipy.io as sio
import numpy as np

class Episode(object):

    def __init__(self, idx, num, name, feat_path,gt_path):
    # idx: episode index
    # num: total number of videos
    # name: video name
    # path: video and gt path
        self.index = idx
        self.name = name[idx%num]
        self.feat = []
        self.gt = []
        temp_1= sio.loadmat(feat_path+self.name+'_alex_fc7_feat.mat')
        self.feat = temp_1['Features'] #(n_frame, 1, 4096)
        dim_non = np.where(self.feat.shape == 1)
        self.feat = np.squeeze(self.feat, axis=(dim_non,))
        temp_2 = sio.loadmat(gt_path+self.name+'_gt.mat')
        self.gt = temp_2['gt']  # (1, n_frame)


    def get_size(self):
        num_frames = np.shape(self.gt)[1]
        return num_frames
