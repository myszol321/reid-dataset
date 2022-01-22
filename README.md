# reid-dataset

## Getting started
CUDA Toolkit is essential to use this program. <br>
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2
### YOLOv4
You can use pre-trained model weights to directly use the model <br>
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT <br>
Put yolov4.weights file into the 'data' folder of this repository. <br>
You can use yolov4-tiny.weights, a smaller and faster model but less accurate, download weights here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

For more information of installation process please visit official repository: [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort) <br>

### reID network
To use reidentification algorithm you need to train the model. <br>
Example of a dataset which can be used to train model: <br>
[Market1501 Dataset](http://www.liangzheng.com.cn/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op) 

For more information of installation process please visit official repository: [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)

## Dataset
Idea of this project is to have set of cameras in which we expect people to re-appear.
Assume we have an object in which we have 3 cameras.
Place video sequences somewhere, for example in ./data/video/object_folder/ <br>
## Running program
Run YOLOv4 tracker for each of these sequences (example for camera c1s1 in object):
```bash
 object_tracker.py:
  --video: path to video (for example: './data/video/object/c1s1.mp4')
  --crop: add flag to save cropped detections to file
  --crop_dir: directory of an object ('object')
  --camera_index: name of a folder that will store cropped detections ('c1s1')
  --crop_rate: frequency of cropping detections (frame rate default:4)
  
  python object_tracker.py --video ./data/vide/object/c1s1.mp4 --crop --crop_dir object --camera_index c1s1
```
There are more arguments to tune, please check them in directly in script or in the authors repo. <br>
In result you have a folder of an object an in that folder we have folders (3) each for present camera:
 * ./object/
   * /c1s1/
   * /c2s2/
   * /c3s3/ 

It is important to name camera using this naming rule.
In every folder you will have cropped detections of the tracker. <br>
Next to group pictures:
```bash
  prepare_put.py:
  --directory: directory of an object (for this example: 'object')
  
  python prepare_put.py --directory object
```
The script will create folder in object directory ('pytorch') and will group pictures by person ID <br>
Extract features:
```bash
  test_put.py:
  --test_dir: directory of an object (for this example: 'object')
  
  python test_put.py --test_dir object
```
Again, in this script you can tune parameters of a model used to extract features. Please check script or visit authors repo. <br>
Extracted features will be saved in pytorch_results.mat and pytorch_results_averages.mat files. The second file contains average values of every folder instead of feature of every picture. <br>
Now you can directly use GUI:
```bash
  gui.py:
  --test_dir: same as above
  --query_cam_index: which camera will be used as query (indexing from 1, default=1)
  
  python gui.py --test_dir object --query_cam_index 3
```
Credits to theAIGuysCode and layumi for creating backbones of this repository:
 * [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort) <br>
 * [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)
