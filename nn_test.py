import tensorflow as tf
import numpy as np
from nn_layer import Layer
from nn import NeuralNet
from read_data import Episode
from agent import Agent
import scipy.io as sio
import os
import cv2 as cv
import math
import utils

# data path and ground truth path.
# feat_path = 'data/TVSum/test/input/feat/'
# gt_path ='data/TVSum/test/input/gt/'
# video_path = 'data/TVSum/test/input/video/'
# save_path = 'data/TVSum/test/output/summary/'

feat_path = 'data/Tour20/test/input/feat/'
gt_path ='data/Tour20/test/input/gt/'
video_path = 'data/Tour20/test/input/video/'
save_path = 'data/Tour20/test/output/summary/'

# names of test videos
test_name = [file.split('_gt.')[0] for file in os.listdir(gt_path)]
test_num = len(test_name)

# define neural network layout
l1 = Layer(4096,400,'relu', name='0')
l2 = Layer(400,200,'relu', name='1')
l3 = Layer(200,100,'relu', name='2')
l4 = Layer(100,25,'linear', name='3')
layers = [l1,l2,l3,l4]
learning_rate = 0.0002
loss_type = 'mean_square'
opt_type = 'RMSprop'

# eval
is_eval = True
hit_numbers = [i for i in range(1, 21, 1)]
score_per_hn = np.zeros(len(hit_numbers))
f1_scores = []
all_segment_num = 0

# load model
Q = NeuralNet(layers,learning_rate,loss_type, opt_type)
Q.recover('model/','Q_net_all_11_0_1000')

for i in range(test_num):
	video = Episode(i,test_num, test_name, feat_path, gt_path)
	video_name = test_name[i]
	frame_num = np.shape(video.feat)[0]

	summary = np.zeros(frame_num)
	id_curr = 0
	while id_curr < frame_num :
		action_value = Q.forward([video.feat[id_curr]])
		a_index = np.argmax(action_value[0])
		id_next = id_curr + a_index+1
		if id_next >frame_num-1 :
			break
		summary[id_next]=1
		id_curr = id_next
  
	# compute performance
	true_pos_hotkey = [1 if (gt==predict and gt==1) else 0 for gt, predict in zip(video.gt[0], summary)]
	true_pos_sum = sum(true_pos_hotkey)
	# print(true_pos_sum)
	# for i in range(len(hit_numbers)):
	# 	hit_number = hit_numbers[i]
	# 	if true_pos_sum>=hit_number:
	# 		score_per_hn[i] += 1
	
	pos_sum = np.sum(video.gt[0])
	predict_sum = np.sum(summary)
	recall = float(true_pos_sum)/float(pos_sum)
	precision = float(true_pos_sum)/float(predict_sum)
	fscore = 2*precision*recall/(precision+recall)
	f1_scores.append(fscore)
 
	cap = cv.VideoCapture(video_path+video_name+'.mp4')
	fps = math.ceil(cap.get(5))
	segment_duration = 2
	segment_len = fps * segment_duration

	segments = [segment for segment in utils.make_chunks(summary, segment_len)]
	segments_gt = [segment for segment in utils.make_chunks(video.gt[0], segment_len)]
	segments_len = len(segments)
	segment_score = [np.mean(segment) for segment in segments]
	segment_score_index = np.argsort(segment_score)
	
	all_segment_num += len(segment_score_index[-int(segments_len/5):])
	for segment_index in segment_score_index[-int(segments_len/5):]:
		
		count_frame_true = sum(segments_gt[segment_index])
		for i in range(len(hit_numbers)):
			hit_number = hit_numbers[i]
			if count_frame_true>=hit_number:
				score_per_hn[i] += 1
				# print(hit_number)
	name = save_path+'sum_'+video_name
	
	print(video_path+video_name+'.mp4', name)
	sio.savemat(name + '.mat',{'summary': summary})
print('number segment', all_segment_num)
for i in range(len(hit_numbers)):
    # print('Hit number: '+str(hit_numbers[i])+'\tdistribute: '+str(score_per_hn[i]/all_segment_num))
    print('Hit number: '+str(hit_numbers[i])+'\tdistribution: '+str(score_per_hn[i]/all_segment_num))
print('F1-score:', np.mean(f1_scores))
print('Test done.')
