### Setup
```sh
# Activate Conda Environment
conda activate vitis-ai-caffe 
```
```sh
# Setup
source /workspace/alveo/overlaybins/setup.sh
```

## Tool to convert darknet model to caffe
The convert.py script supports yolov2 and yolov3 models. This tool does not support tiny-yolov3 model due to fractional stride in maxpool. For conversion, run the below command. 

> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

```sh
cd $VAI_ALVEO_ROOT/apps/yolo/darknet_to_caffe
#Download yolov3.weights and yolov3.cfg 
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
python convert.py yolov3.cfg yolov3.weights yolov3.prototxt yolov3.caffemodel
```

## Run the converted model on cpu using $VAI_ROOT/apps/yolo/detect.sh and visualize output on test_image_set. 

Before running the below command, download and prepare the data as mentioned in the $VAI_ALVEO_ROOT/apps/yolo/README.md file. 

```sh
cd $VAI_ALVEO_ROOT/apps/yolo
./detect.sh -t cpu_detect  \
        -m custom \
        -d test_image_set \
        --neth 608 --netw 608 \
        --network  darknet_to_caffe/yolov3.prototxt\
        --weights darknet_to_caffe/yolov3.caffemodel \
        --bias darknet_to_caffe/yolov3_biases.txt \
        --anchorcnt 3 \
        --yoloversion v3 \
        --nclasses 80 \
        --iouthresh 0.45 \
        --scorethresh 0.25 \
        --labels coco.names \
        --dump_results \
        --visualize \
        --results_dir cpu_results
```
The output images are stored in cpu_results folder. 

## Calculate accuracy (mAP) on CPU with val_set of 2000 images.

The steps for creating val2k folder is mentioned in the $VAI_ALVEO_ROOT/apps/yolo/README.md
```sh
cd $VAI_ALVEO_ROOT/apps/yolo
./detect.sh -t cpu_detect  \
        -m custom \
        -d val2k \
        -g labels/val2014 \
        --neth 608 --netw 608 \
        --network  darknet_to_caffe/yolov3.prototxt\
        --weights darknet_to_caffe/yolov3.caffemodel \
        --bias darknet_to_caffe/yolov3_biases.txt \
        --anchorcnt 3 \
        --yoloversion v3 \
        --nclasses 80 \
        --labels coco.names
```
## Calculate accuracy (mAP) on FPGA with val_set of 2000 images

```sh
cd $VAI_ALVEO_ROOT/apps/yolo
./detect.sh -t test_detect  \
        -m custom \
        -d val2k \
        -g labels/val2014 \
        --neth 608 --netw 608 \
        --network  darknet_to_caffe/yolov3.prototxt\
        --weights darknet_to_caffe/yolov3.caffemodel \
        --bias darknet_to_caffe/yolov3_biases.txt \
        --anchorcnt 3 \
        --yoloversion v3 \
        --nclasses 80 \
        --labels coco.names
```

