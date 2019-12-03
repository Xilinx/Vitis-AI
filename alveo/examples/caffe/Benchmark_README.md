# Imagenet Benchmarks
*(top-1/top-5)

|Network | CPU FP32* | CPU INT8* | FPGA INT8* | Resize method |
|--------|-----------|-----------|------------|---------------|
|inception_v1|65.68/87.03|65.4/86.72|65.35/86.75|Center crop 224x224 from resized image with shorter side=256|
|inception_v2|71.70/90.38|71.68/90.36|71.67/90.28|Center crop 224x224 from resized image with shorter side=256|
|inception_v3|74.63/92.04|74.22/91.82|74.25/91.81|Direct resize to 299x299|
|inception_v4|78.63/94.21|78.57/94.20|78.60/94.19|Direct resize to 299x299|
|resnet50_v1|75.14/92.12|74.10/91.64|74.42/91.75|Center crop 224x224 from resized image with shorter side=256|
|resnet50_v2|75.03/92.20|74.82/92.13|74.82/92.18|Center crop 224x224 from resized image with shorter side=256|
|squeezenet|57.09/79.95|55.41/79.19|55.81/79.20|Direct resize to 227x227|
|vgg16|71.11/89.95|70.90/89.91|71.03/89.93|Center crop 224x224 from resized image with shorter side=256|

# Performance

| Network                         | Latency mode (fps / PE)  | Throughput mode (fps / PE) |
|---------------------------------|--------------------|--------------------|
| inception_v1                    | 802.95 | 962.28  |
| inception_v2                    | 608.31 | 695.6   |
| inception_v3                    | 224.39 | 231.67  |
| inception_v4                    | 121.73 | 123.92  |
| resnet50_v1                     | 287.04 | 306.57  |
| resnet50_v2                     | 264.73 | 279.81  |
| squeezenet                      | 1056.3 | 1696.93 |
| VGG16                           | 101.73 | 119.08  |
| yolo v2 (608x608)               | 31.47 |  31.47   |
| yolo v2 prelu (608x608)         | 27.32 |  27.32   |
| inception_v2_ssd                | 34.97 |         |
  
*NOTE: alveo-u200 can fit up to 3 PEs, alveo-u250 can fit up to 4 PEs. Multiply the numbers in this table by the number of PEs in your system. PE refers to Processing Elements, or the number of DPU-v1 hardware kernels.


## Download the Imagenet Validation Set
To run a full Imagenet benchmark evaluation, one must have access to the Imagenet validation set (50,000 images).   
It is recommended to download these files and store them on your native machine.  
Go to the [download page](http://www.image-net.org/download-images), and download ILSVRC2012_img_val.tar. (You may need to register an account). Another potential solution is:  
`wget -c http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar`  

## 1. Running Caffe Classification Benchmark Models

### Bring the Imagnet Validation Set into containr
```
# Make sure to untar your downloaded validation set:
cd /<PATH>/<TO>/<IMAGENET>/
mkdir imagenet_val
tar -xvf ILSVRC2012_img_val.tar -C imagenet_val
# Make a copy of the data set for manipulation
cp -rf imagenet_val imagenet_val_224x224
# Make sure to give all users the permission to read and write
chmod -R a+rw imagenet_val
chmod -R a+rw imagenet_val_224x224
```

### center 224x224 crop from resized image with shorter side=256
```
cd imagenet_val_224x224
sudo apt install imagemagick
for name in *.JPEG; do
    convert -resize 256x256^ -gravity center -extent 224x224 $name $name
done
```
### Mount the images when launching the container
```
# When launching the container make sure to mount the downloaded imagenet images, using the below switch.
# -v /<PATH>/<TO>/<IMAGENET>/:/home/mluser/CK-TOOLS/ \
```

### Get the ground truth file
```
python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-aux
# the above command generates val.txt in $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux folder.
```

### Example
```
cd $VAI_ALVEO_ROOT/example/caffe
# Get the necessary models
python getModels.py
# Setup VAI Environment Variables
source $VAI_ALVEO_ROOT/overlaybins/setup.sh
```
In train_val prototxt change **source** and **root_folder** in image_data_param param

```
name: "GoogleNet"
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 224
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  image_data_param {
    source: "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt"
    root_folder: "/home/mluser/CK-TOOLS/imagenet_val_224x224/" #For direct_resize use "/home/mluser/CK-TOOLS/imagenet_val/" example inception_v3 network
    batch_size: 50
    shuffle : true
  }
}
```

### Prepare a model for inference
If you plan to work on several models, you can use the --output_dir switch to generate and store model artifacts in seperate directories. By default, output_dir is ./work
```
python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prepare --output_dir work
```
### Runing inference for 50K images
```
python run.py --validate --output_dir work --numBatches 1000

```

## 2. Running Caffe Inception_v2 SSD Model:
```
cd $VAI_ALVEO_ROOT/exampls/caffe/ssd-detect
```
### Data Preparation

Download VOC2007 dataset.

```
# Download the data.
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtest_06-Nov-2007.tar
#Generate ground truth file
python generate_gt_file.py
Note: VOC dataset contains 21 classes. But this model is trained with 19 classes (removed diningtable and train). If your model is having 21 classes please comment 40 line in generate_gt_file.py
```
```
The format of calib.txt used in calibration phase of vai_q is as follow:
#image_name fake_label_number  
000001.jpg 1
000002.jpg 1
000003.jpg 1
000004.jpg 1
000006.jpg 1

Please note that the label number is not actually used in calibration and arbitrary label number can be used.
```

```
# Get the necessary models
python $VAI_ALVEO_ROOT/exampls/caffe/getModels.py

# Setup VAI Environment Variables
source $VAI_ALVEO_ROOT/overlaybins/setup.sh

```

### Prepare a model for inference

```
python run_ssd.py --prototxt /opt/models/caffe/inception_v2_ssd/inception_v2_ssd_train.prototxt --caffemodel /opt/models/caffe/inception_v2_ssd/inception_v2_ssd.caffemodel --prepare
```

### Run Inference on entire dataset and calculate mAP
```
python run_ssd.py --prototxt xfdnn_auto_cut_deploy.prototxt --caffemodel quantize_results/deploy.caffemodel --labelmap_file labelmap_voc_19c.prototxt --test_image_root ./VOCdevkit/VOC2007/JPEGImages/ --image_list_file ./VOCdevkit/VOC2007/ImageSets/Main/test.txt --gt_file voc07_gt_file_19c.txt --validate
```


## 3. Running Yolo_v2 Model:

```
cd $VAI_ALVEO_ROOT/apps/yolo/
```
### Data Preparation
```
Download Images and lables
wget -c https://pjreddie.com/media/files/val2014.zip
unzip -q val2014.zip
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
```

### Run Inference on entire dataset and calculate mAP
```
./run.sh -p alveo-u200 -t test_detect -k v3 -b 8 -m yolo_v2_prelu_608 -g labels/val2014/ -d val2014/
```
