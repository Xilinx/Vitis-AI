
# Running Caffe pix2pix (maps_AtoB) Model:

### Setup

> **Note:** Skip, If you have already run the below steps.

Activate Conda Environment
  ```sh
  conda activate vitis-ai-caffe 
  ```

Setup the Environment

  ```sh
  source /workspace/alveo/overlaybins/setup.sh
  ```

## Data Preparation

Download maps-dataset from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz as follows
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

```
cd /workspace/alveo/examples/caffe/pix2pix/maps_AtoB/
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz
tar -xvf maps.tar.gz
rm maps.tar.gz
```

The folder is supposed to be as the following.  

```
/workspace/alveo/examples/caffe/pix2pix/maps_AtoB/maps/train
/workspace/alveo/examples/caffe/pix2pix/maps_AtoB/maps/val
```

The downloaded images have the combination of Cityscapes Semantic photo and label. 
To split Semantic photo and label, please run the following command lines.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/maps_AtoB/
$ python extract_label_aerial.py
```

This will generate two subfolders in val folder. 'photo' and 'label'. 
```
/workspace/alveo/examples/caffe/pix2pix/maps_AtoB/maps/val/photo
/workspace/alveo/examples/caffe/pix2pix/maps_AtoB/maps/val/label
```  


## Pix2Pix (maps_AtoB) model

Pix2pix is image to image translastion using GAN [1]


[1]	Phillip Isola Jun-Yan Zhu Tinghui Zhou Alexei A. Efros: Image-to-Image Translation with Conditional Adversarial Networks (2016), arXiv:1611.07004



maps_AtoB model translates aerial photo to map. 


We trained Pix2Pix (maps_AtoB) model with input size as [256,256,3].

After training the model, we quantized the model to deploy on FPGA.

To get the quantized Caffe model, run the following command lines. 

> **Note:** Skip, If you have already run the below steps.
```
cd /workspace/alveo/examples/caffe
python getModels.py
```

The Pix2Pix (maps_AtoB) model files would be located in '/workspace/alveo/examples/caffe/models/maps_AtoB' folder.

Copy the model files to 'pix2pix/maps_AtoB/quantize_results' with the following commands.
```
cd /workspace/alveo/examples/caffe/pix2pix/maps_AtoB
cp -R /workspace/alveo/examples/caffe/models/maps_AtoB ./quantize_results
```

You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'maps_AtoB/quantize_results' sub-folder.


## Run Inference model on CPU

To run the inference model on cpu for translating aerial photo to map, run the following commands.
```
cd /workspace/alveo/examples/caffe/pix2pix/maps_AtoB
python maps_AtoB_cpu.py --image <image-file>
For example, 
$ python maps_AtoB_cpu.py --image ./maps/val/photo/1.jpg
```
The generated semantic label image will be stored in 'test_output' sub-folder.


## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following commands

```
cd /workspace/alveo/examples/caffe/pix2pix/maps_AtoB
source run.sh deploy
```

All compiler files will be generated in 'work' sub folder.

xfdnn_deploy.prototxt (used to execute Caffe model on FPGA) will be generated at root folder.




## Run Inference model on FPGA 

To run the inference model on fpga with cityscape photo images, run the following commands.

```
cd /workspace/alveo/examples/caffe/pix2pix/maps_AtoB
python maps_AtoB_fpga.py --image <image-file>
For example, 
$ python maps_AtoB_fpga.py --image ./maps/val/photo/1.jpg
```
The generated semantic label image will be stored in 'test_output' sub-folder.
