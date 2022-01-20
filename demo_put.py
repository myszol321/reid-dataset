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
parser.add_argument('--query_avg_index', type=int, help='test image average index')
parser.add_argument('--test_dir',default=r'put_cameras\hala_sportowa',type=str, help='./test_data')
parser.add_argument('--no_duplicates', action='store_true', help='if set, program will not display more than one photo of each person')
parser.add_argument('--scores', action='store_true', help='if set, program will calculate scores of every occurence of a person')
parser.add_argument('--query_cam_index', default=1, type=int, help='which camera will be query (the rest of cameras will be gallery)')
opts = parser.parse_args()

query_cam_index = opts.query_cam_index
query_cam_index = int(query_cam_index)
test_dir = opts.test_dir
data_dir = os.path.join(os.getcwd(), test_dir + '/pytorch')
dirs = os.listdir(data_dir)
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in dirs}

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

repetitions = {}
counter = 1
query_path = ''

filename = 'pytorch_result.mat'
average_filename = 'pytorch_result_averages.mat'
result = scipy.io.loadmat(test_dir + "/" + filename)
avg_result = scipy.io.loadmat(test_dir + "/" + average_filename)

features = torch.FloatTensor(result['feature'])
cams = result['cam'][0]
labels = result['label'][0]
avg_features = torch.FloatTensor(avg_result['feature'])
avg_cams = avg_result['cam'][0]
avg_labels = avg_result['label'][0]



#######################################################################
# sort the images (qf - feature of a query person, gf - gallery features)
def sort_img(qf, gf):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)  
    #from small to large
    index = index[::-1]
        
    return index


def calculate_len(lengths, index):
    bound = index
    if isinstance(index, np.ndarray):
        if index[0] < lengths[0]:
            return index
        else:
            bound = index[0]
           
    elif index < lengths[0]:
        return index
    result = 0
    i = 0
    
    while result<=bound:
        result += lengths[i]
        i += 1
    return index - (result - lengths[i-1])
        
def calculate_scores(index, bound):
    global gallery_scores
    for i in range(bound):
        cam_ind = int(gallery_cams[index[i]])
        label = int(gallery_labels[index[i]])
        mask1 = np.argwhere(avg_gallery_labels==label)
        mask2 = np.argwhere(avg_gallery_cams==cam_ind)
        ind = np.intersect1d(mask1, mask2)[0]
        gallery_scores[ind] += 1
    sorted_index = np.argsort(gallery_scores)
    # from max to min
    return sorted_index[::-1]
    

i = opts.query_index

#index = sort_img(query_feature[i], gallery_feature)





########################################################################
# Visualize the rank result
query_path = ''

#query_index = query_label[i]

dir_inds = []
counter = 1
counter2 = 0
for subdir in dirs:
    dir_inds.append(str(subdir))
        
cameras = np.unique(cams)
gallery_lengths = []

itr = 1
for cam in cameras:
    if itr == query_cam_index:
        qf_inds = cams==cam
        query_features = features[qf_inds]
        query_features = query_features.cuda()
        query_labels = labels[qf_inds]
        query_cams = cams[qf_inds]
        gf_inds = cams!=cam
        gallery_features = features[gf_inds]
        gallery_features = gallery_features.cuda()
        gallery_labels = labels[gf_inds]
        gallery_cams = cams[gf_inds]
        index = sort_img(query_features[i], gallery_features)
        
        # average values
        
        qf_inds = avg_cams == cam
        avg_query_features = avg_features[qf_inds]
        avg_query_features = avg_query_features.cuda()
        avg_query_labels = avg_labels[qf_inds]
        avg_query_cams = avg_cams[qf_inds]
            
        gf_inds = avg_cams != cam
        avg_gallery_features = avg_features[gf_inds]
        avg_gallery_features = avg_gallery_features.cuda()
        avg_gallery_labels = avg_labels[gf_inds]
        avg_gallery_cams = avg_cams[gf_inds]
        
        gallery_scores = np.zeros(len(avg_gallery_features))
        if opts.query_avg_index:
            j = opts.query_avg_index
            mean_query_index = avg_query_labels[j]
            averages_index = sort_img(avg_query_features[j], avg_gallery_features)
        if opts.scores:
            bound = 50
            sorted_scores = calculate_scores(index, bound)
    else:
        gallery_len = len(np.argwhere(cams==cam))
        gallery_lengths.append(gallery_len)
        repetitions[int(cam)] = []
    itr += 1
query_path, _ = image_datasets[dir_inds[query_cam_index-1]].imgs[i]
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    if opts.no_duplicates:
        fig = plt.figure(figsize=(16,5))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(query_path,'query\ncam ' + str(query_cam_index) + '\nperson ' + str(int(query_labels[i])))
        print("Query path: ", query_path)
        top_ten = 0
        i = 0
        print('Top 10 images are as follow (no duplicates):')
        while top_ten<10:
            ax = plt.subplot(1,11,top_ten+2)
            ax.axis('off')
            cam_ind = int(gallery_cams[index[i]])

            ind = calculate_len(gallery_lengths, index[i])
            img_path, _ = image_datasets[dir_inds[cam_ind-1]].imgs[ind]

            label = gallery_labels[index[i]]
            if not int(label) in repetitions[int(cam_ind)]:
                ax.set_title('cam ' + str(int(cam_ind)) + '\nperson ' + str(int(label)))
                imshow(img_path)
                print(img_path)
                repetitions[int(cam_ind)].append(int(label))
                top_ten += 1
            i += 1
    elif opts.scores:
        print('cam ' + str(query_cam_index) + ' person ' + str(int(query_labels[i])))
        fig = plt.figure(figsize=(16,5))
        ax = plt.subplot(1,11,1)
        ax.axis('off')

        imshow(query_path,'query\ncam ' + str(query_cam_index) + '\nperson ' + str(int(query_labels[i])))
        print("Query path: ", query_path)
        print('Top 10 images are as follow (scores):')
        for i in range(10):
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')
            person_id = int(avg_gallery_labels[sorted_scores[i]])
            cam_ind = int(avg_gallery_cams[sorted_scores[i]])
            ax.set_title("Top " + str(i+1) + "\nperson id:" + str(person_id) + "\ncam:" + str(cam_ind))
            print("PERSON", i+1, "INDEX:", person_id, "CAM:", cam_ind)
            mask1 = np.argwhere(gallery_labels==person_id)
            mask2 = np.argwhere(gallery_cams==cam_ind)
            inds = np.intersect1d(mask1, mask2)

            ind = calculate_len(gallery_lengths, inds)
            mid = (ind[0]+ind[-1])/2
            mid = int(mid)
            size = 3
            img_path, _ = image_datasets[dir_inds[int(cam_ind)-1]].imgs[mid]
            imshow(img_path)
            print(img_path)
    else:
        fig = plt.figure(figsize=(16,5))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(query_path,'query\ncam ' + str(query_cam_index) + '\nperson ' + str(int(query_labels[i])))
        print("Query path: ", query_path)
        print('Top 10 images are as follow (classic):')
        for i in range(10):
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')
            cam_ind = int(gallery_cams[index[i]])
            
            ind = calculate_len(gallery_lengths, index[i])
            img_path, _ = image_datasets[dir_inds[cam_ind-1]].imgs[ind]
            label = gallery_labels[index[i]]
            ax.set_title('cam ' + str(int(cam_ind)) + '\nperson ' + str(int(label)))
            imshow(img_path)
            print(img_path)
    if opts.no_duplicates:
        print("Avoided repetitions: ")
        for key in repetitions:
            print("Cam " + str(key) + ": person ids: " + str(repetitions[key]))
    # Average ranking results ----------
    if opts.query_avg_index:
        print("AVERAGE QUERY INDEX: ", mean_query_index)
        fig2 = plt.figure(figsize=(16,10))
        ax = plt.subplot(3, 6, 1)
        ax.axis('off')
        query_inds = np.where(query_labels==mean_query_index)[0]
        query_path, _ = image_datasets[dir_inds[query_cam_index-1]].imgs[query_inds[0]]
        imshow(query_path, 'query \nperson id: ' + str(mean_query_index) + '\ncam ' + str(query_cam_index))
        print("Query path: ", query_path)
        for i in range(5):
            ax = plt.subplot(3,6,i+2)
            ax.axis('off')
            person_id = int(avg_gallery_labels[averages_index[i]])
            cam_ind = int(avg_gallery_cams[averages_index[i]])
            ax.set_title("Top " + str(i+1) + "\nperson id:" + str(person_id) + "\ncam:" + str(cam_ind))
            print("PERSON", i+1, "INDEX:", person_id, "CAM:", cam_ind)
            
            mask1 = np.argwhere(gallery_labels==person_id)
            mask2 = np.argwhere(gallery_cams==cam_ind)
            inds = np.intersect1d(mask1, mask2)

            ind = calculate_len(gallery_lengths, inds)
            size = 3
            img_path, _ = image_datasets[dir_inds[int(cam_ind)-1]].imgs[ind[0]]
            imshow(img_path)
            print(img_path)
            if len(inds) < 3:
                size = len(inds)
            for j in range(1, size):
                ax = plt.subplot(3, 6, i+2+6*j)
                ax.axis('off')
                img_path, _ = image_datasets[dir_inds[int(cam_ind)-1]].imgs[ind[j]]
                imshow(img_path)
                print(img_path)
    
    
except Exception as e:
   print(str(e))

fig.savefig("results.png")
if opts.query_avg_index:
    fig2.savefig("average_results.png")
