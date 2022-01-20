# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, ft_net_dense, ft_net_swin, ft_net_NAS, PCB, PCB_test
from shutil import move
from tqdm import tqdm
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default=r'put_cameras\hala_sportowa',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ibn', action='store_true', help='use ibn.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()

# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.use_swin = config['use_swin']
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

if 'ibn' in config:
    opt.ibn = config['ibn']

str_ids = opt.gpu_ids.split(',')
name = opt.name
test_dir = opt.test_dir
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
if opt.use_swin:
    h, w = 224, 224
else:
    h, w = 256, 128

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = os.path.join(os.getcwd(), test_dir + '/pytorch')
test_dir = os.path.join(os.getcwd(), test_dir)
dirs = os.listdir(data_dir)
print("Present directories:")
print(dirs)
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in dirs}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=0) for x in dirs}
use_gpu = torch.cuda.is_available()
######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    global size_ind, sizes
    features = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders, ncols=50):
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
elif opt.use_swin:
    model_structure = ft_net_swin(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn )

if opt.PCB:
    model_structure = PCB(opt.nclasses)


model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if opt.PCB:
        model = PCB_test(model)
else:
        model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature

features = torch.FloatTensor()
avg_features = torch.FloatTensor()

labels = np.empty(0)
avg_labels = np.empty(0)

cams = np.empty(0)
avg_cams = np.empty(0)

with torch.no_grad():
    for subdir in dirs:
        print('Current directory: ', subdir)
        path = image_datasets[subdir].imgs
        cam, label = get_id(path)
        feature = extract_feature(model,dataloaders[subdir])
        
        features = torch.cat((features, feature))
        labels = np.append(labels, label)
        cams = np.append(cams, cam)

        # -----------------------------------
        # Calculating averages of features
        # -----------------------------------
        
        print("Calculating average of features...")
        size = np.amax(label)
        mean_feature = np.zeros((size,512))
        mean_label = np.arange(size)
        ind = 0
        old_ind = 0
        np_features = feature.numpy()
        np_label = np.array(label)
        counter = 0
        for i in range(size+1):
            ind_table = np.where(np_label == i)[0]
            if ind_table.size:
                ind = ind_table[-1]
                tab_slice = np_features[old_ind:ind+1]

                tab_slice = np.mean(tab_slice, axis=0)
                mean_feature[counter] = tab_slice
                mean_label[counter] = i
                counter += 1
                
                old_ind = ind
        

        slice_tab = np.arange(counter, len(mean_feature))
        mean_feature = np.delete(mean_feature, slice_tab, 0)
        slice_tab = np.arange(counter, len(mean_label))
        mean_label = np.delete(mean_label, slice_tab, 0)
        mean_cam = np.array(cam)
        slice_tab = np.arange(counter, len(mean_cam))
        mean_cam = np.delete(mean_cam, slice_tab, 0)

        mean_tensor = torch.from_numpy(mean_feature)

        avg_features = torch.cat((avg_features, mean_tensor))
        avg_labels =  np.append(avg_labels, mean_label)
        avg_cams = np.append(avg_cams, mean_cam)
        
    
    result = {'feature':features.numpy(), 'label':labels, 'cam':cams}
    filename = 'pytorch_result.mat'
    
    scipy.io.savemat(filename, result)
    move(filename , test_dir + "/" + filename)
    
    average_result = {'feature': avg_features.numpy(), 'label':avg_labels, 'cam':avg_cams}
    average_filename = 'pytorch_result_averages.mat'
    scipy.io.savemat(average_filename, average_result)
    move(average_filename, test_dir + "/" + average_filename)
 