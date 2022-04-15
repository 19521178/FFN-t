from torchvision.models import alexnet
import torchvision.transforms as transforms
import torch
# from alive_progress import alive_bar

from PIL import Image
import numpy as np 
import scipy.io as sio
import cv2 as cv
import os

videos_path = 'data/TVSum/train/done_video/'
save_path = 'data/TVSum/train/feat/'
suffixes_feat = '_alex_fc7_feat.mat'

device = torch.device('cuda') # thêm

#load model alexnet
model = alexnet(pretrained=True)
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1]) #sửa
model.to(device)    # thêm

# transform image to be compatiable with alexnet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def reshape_img(img):
    pil_img = Image.fromarray(np.uint8(img)).convert('RGB')
    img_t = transform(pil_img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t



# read video 
videos_name = sorted(os.listdir(videos_path))
for video_name in videos_name: # nhớ sửa thành 25:
    print('Extracting '+ video_name, end=' | ')
    features = []
    capture = cv.VideoCapture(videos_path+video_name)
    
    # num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    # with alive_bar(num_frames) as bar:
    #     for i in range(num_frames):
    #         success, frame = capture.read()
    #         if success:
    #             reshape_frame = reshape_img(frame)
    #             feat = model.forward(reshape_frame)
    #             features.append(feat)
    #         bar()
     
    success, frame = capture.read()
    while success:
        with torch.no_grad():   # thêm
            reshape_frame = reshape_img(frame)
            reshape_frame = reshape_frame.to(device)    # thêm
            feat = model(reshape_frame)
            feat = feat.cpu().numpy()   # sửa
            features.append(feat)
            success, frame = capture.read()
    # features.pop()
    
    sio.savemat(save_path + video_name.split('.')[0] + suffixes_feat, {'Features': features})
    print('Saved')  
