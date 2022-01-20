import os
from shutil import copyfile
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Prepare')
parser.add_argument('--directory', default=r'put_cameras\hala_sportowa', type=str, help='Directory where pictures are stored')
opt = parser.parse_args()

directory = opt.directory
download_path = os.path.join(os.getcwd(), directory)
if not os.path.isdir(download_path):
    print('please change the download_path')
    print('Dir is: ', download_path)

dirs = os.listdir(download_path)
pytorch_dir = os.path.join(download_path, 'pytorch')
if not os.path.isdir(pytorch_dir):
    os.mkdir(pytorch_dir)
for sub_dir in dirs:
    path = os.path.join(download_path, sub_dir)
    pytorch_save_path = os.path.join(pytorch_dir, sub_dir)
    if not os.path.isdir(pytorch_save_path):
        os.mkdir(pytorch_save_path)
    print("Current directory:", sub_dir)
    for root, dirs, files in os.walk(path, topdown=True):
        for name in tqdm(files, ncols=50):
            if not name[-3:]=='jpg':
                continue
            ID = name.split('_')
            src_path = path + '/' + name
            dst_path = pytorch_save_path + '/' + ID[0]

            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)