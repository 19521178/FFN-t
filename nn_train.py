import tensorflow as tf
import numpy as np
import os
import time
from nn_layer import Layer
from nn import NeuralNet
from read_data import Episode
from agent import Agent

# data path and names.
# video_path = 'data/TVSum/train/video/'
# feat_path = 'data/TVSum/train/feat/'
# gt_path ='data/TVSum/train/gt/'
video_path = 'content/train/video/'
feat_path = 'content/train/feat/'
gt_path ='content/train/gt/'

# split 1
train_name = [file.split('.')[0] for file in os.listdir(video_path)]
train_num = len(train_name)

# define neural network layout
l1 = Layer(4096,400,'relu', name='0')
l2 = Layer(400,200,'relu', name='1')
l3 = Layer(200,100,'relu', name='2')
l4 = Layer(100,25,'linear', name='3')
layers = [l1,l2,l3,l4]
learning_rate = 0.0002
loss_type = 'mean_square'
opt_type = 'RMSprop'

# set Q learning parameters
batch_size = 128
exp_rate = 1
exp_low = 0.1
exp_decay = 0.00001
decay_rate = 0.8
max_eps = 50
save_per_eps = 10
savepath = 'model_R3_A25_S1_1013/'
filename = 'Q_net_all_11_0_800'

# if continue train 
is_continue_train = True
current_epoch = 1   #0 if is_continue_train false
load_filename = filename + '_' + str(current_epoch)

# define Q learning agent
agent = Agent(layers, batch_size, exp_rate,exp_low,exp_decay, learning_rate, decay_rate,savepath, load_filename, is_continue_train)

# Training process
for epoch in range(current_epoch, max_eps):
    start_time = time.time()
    recalls = []
    precisions = []
    fscores = []
    for video_index in range(train_num):
        index = epoch * train_num + video_index
        current_eps = Episode(index,train_num, train_name, feat_path, gt_path)

        agent.data_init(current_eps)
        agent.episode_run()

        pos = 0
        true_pos = 0
        summ = 0
        for i in range(current_eps.get_size()):
            if current_eps.gt[0][i]==1:
                pos = pos+1
            if current_eps.gt[0][i]==1 and agent.selection[i]==1:
                true_pos = true_pos+1   
            if agent.selection[i] ==1:
                summ = summ+1
        recall = float(true_pos)/float(pos)
        precision = float(true_pos)/float(summ)
        fscore = 2*precision*recall/(precision+recall)
        recalls.append(recall)
        precisions.append(precision)
        fscores.append(fscore)
        agent.data_reset()
    print('epoch:'+str(epoch+1)+', gt: '+str(pos)+', sum: '+str(summ)+', tp: '+ str(true_pos)+', r: '+str(np.mean(recalls))+', p: '+str(np.mean(precisions))+', f: '+str(np.mean(fscores))+', time: '+str(time.time()-start_time))
    if ((epoch+1)%save_per_eps==0):
        agent.save_model(filename+'_'+str(epoch+1))
        print('Save model at epoch '+ str(epoch+1))


# agent.save_model(filename)
