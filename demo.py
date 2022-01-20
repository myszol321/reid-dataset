import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
# parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
# parser.add_argument('--test_dir',default='/content/drive/MyDrive/Person_reID_baseline_pytorch/Market-1501-v15.09.15/pytorch',type=str, help='./test_data')
parser.add_argument('--test_dir',default=r'C:\Users\Hubs\Documents\re-id-pytorch\Person_reID_baseline_pytorch\data\pytorch',type=str, help='./test_data')
parser.add_argument('--average',action='store_true')
opts = parser.parse_args()

data_dir = opts.test_dir
data_dir = os.path.join(os.getcwd(), data_dir)
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')

query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

if opts.average:
    mean_result = scipy.io.loadmat('pytorch_result_averages.mat')

    query_mean_feature = torch.FloatTensor(mean_result['query_f'])
    query_mean_label = mean_result['query_label'][0]
    gallery_mean_feature = torch.FloatTensor(mean_result['gallery_f'])
    gallery_mean_label = mean_result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()
if opts.average:
    query_mean_feature = query_mean_feature.cuda()
    gallery_mean_feature = gallery_mean_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

def sort_averages(qf, ql, gf, gl):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    return index
    
i = opts.query_index

index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
if opts.average:
    averages_index = sort_averages(query_mean_feature[i],query_mean_label[i],gallery_mean_feature,gallery_mean_label)
############################################
# Visualize the rank result
query_path, _ = image_datasets['query'].imgs[i]

#image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}

query_index = query_label[i]
if opts.average:
    mean_query_index = query_mean_label[i]

print('Top 10 images are as follow:')
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    print("Query path: ", query_path)
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_index:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')
        print(img_path)
        
    # Average ranking results ----------
    if opts.average:
        print("MEAN QUERY INDEX: ", mean_query_index)
        fig2 = plt.figure(figsize=(16,10))
        ax = plt.subplot(3, 6, 1)
        ax.axis('off')
        #print(query_label)
        query_inds = np.where(query_label==mean_query_index)[0]
        #print(query_inds)
        query_path, _ = image_datasets['query'].imgs[query_inds[0]]
        imshow(query_path, 'query \n(person id: ' + str(mean_query_index) + ')')
        print("Query path: ", query_path)
        for i in range(5):
            ax = plt.subplot(3,6,i+2)
            ax.axis('off')
            person_id = gallery_mean_label[averages_index][i]
            ax.set_title("Top " + str(i+1) + "\n(person id:" + str(person_id) + ")")
            print("PERSON", i+1, "INDEX:", person_id)
            inds = np.where(gallery_label==person_id)[0]
            size = 3
            img_path, _ = image_datasets['gallery'].imgs[inds[0]]
            imshow(img_path)
            print(img_path)
            if len(inds) < 3:
                size = len(inds)
            for j in range(1, size):
                ax = plt.subplot(3, 6, i+2+6*j)
                ax.axis('off')
                img_path, _ = image_datasets['gallery'].imgs[inds[j]]
                imshow(img_path)
                print(img_path)
            
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')


fig.savefig("results.png")
if opts.average:
    fig2.savefig("average_results.png")
