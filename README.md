# yoloncv
(YOLO ON OPENCV)
Test yolo object detector model with opencv4.2 dnn
![Image text](https://raw.githubusercontent.com/colinlin1982/yoloncv/master/screenshot/WX20200229-214217.png)

Requirements
============
c++
1. opencv >= 4.2 (with dnn module)
2. cmake >= 3.13
python
1. opencv>=4.2

Usage
=====
test on a image:  
detect --image=/your/path/to/image.jpg --cfg=/your/path/to/config.data --thresh=0.5 --nms=0.6  
detect --image=/your/path/to/image.jpg --cfg=/your/path/to/config.data  
detect -i=/your/path/to/image.jpg -c=/your/path/to/config.data -t=0.5 -n=0.6  

test on jpg images in a dir:  
detect --dir=/your/path/to/dir --cfg=/your/path/to/config.data --thresh=0.5 --nms=0.6  
python testoncv.py -d /your/path/to/dir -c /your/path/to/config.data -t 0.5 -n 0.6

About config.data
=================
names  : path to your classes name list, such as coco.names, voc.names  
cfg    : path to your darknet network cfg file, such as yolov3.cfg, yolov3-tiny.cfg  
weights: path to your darknet network weights file, download from web or trained with your own dataset  
width  : network input width  
height : network input height  

Contact me
==========
QQ群：695860125

Thanks
======
https://github.com/spmallick/learnopencv/tree/master/ObjectDetection-YOLO
https://blog.csdn.net/reasonyuanrobot/article/details/89451198
