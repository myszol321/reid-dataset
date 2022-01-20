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

Credits to theAIGuysCode and layumi for creating backbones of this repository:
 * [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)
 * [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)
