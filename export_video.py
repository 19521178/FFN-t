import cv2 as cv 
import numpy as np
import os
import scipy.io as sio

# from alive_progress import alive_bar

videos_path = 'data/TVSum/test/input/video/'
# videos_path = 'input/video/'
label_path = 'data/TVSum/test/output/summary/'
# label_path = 'data/TVSum/test/summary/'
# label is a hot-encoding vector of important frame in a video
out_path = 'data/TVSum/test/output/video/'

fourcc = cv.VideoWriter_fourcc(*'MP4V')
for video_name in sorted(os.listdir(videos_path)):
    print('Exporting '+ video_name, end=' | ')
    # prepare read and write video
    cap = cv.VideoCapture(videos_path+video_name)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    out = cv.VideoWriter(out_path+video_name.split('.')[0]+'.mp4', \
                        fourcc, fps, (width, height), isColor=True)
    
    # load label
    label = sio.loadmat(label_path + 'sum_' + video_name.split('.')[0] + '.mat')['summary']
    label = np.squeeze(label, (0, ))

    # with alive_bar(num_frames) as bar:
    #     for i in range(num_frames):
    #         if label[i]==1:
    #             out.write(cap.read()[1])
    #         bar()
    success, frame = cap.read()
    index = 0
    while success:
        if label[index]==1:
            out.write(frame)
        index += 1
        success, frame = cap.read()
    cap.release()
    out.release()
    print(' | Saved')