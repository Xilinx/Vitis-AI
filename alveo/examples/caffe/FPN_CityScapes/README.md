# Running FPN CityScapes segmentation model on FPGA 


## activate Caffe conda environment

Please activate Caffe conda environment using the following commands.

```
$ conda activate vitis-ai-caffe
$ source /workspace/alveo/overlaybins/setup.sh
$ cd /workspace/alveo/examples/caffe/FPN_CityScapes
```


## Data Preparation

Download cityscapes-dataset from https://www.cityscapes-dataset.com/downloads/

You need to register for the website to download the dataset.

Please download 'cityscapes/frankfurt' folder.

```
/workspace/alveo/examples/caffe/FPN_CityScapes/cityscapes/frankfurt
```


## FPN CityScapes segmentation model

We trained FPN CityScapes model with input size as [256,512,3].

After training the model, we quantized the model to deploy on FPGA.

To get the quantized Caffe model, run the following command lines. 

```
$ cd /workspace/alveo/examples/caffe 
$ python getModels.py
```

The FPN CityScapes model files would be located in '/workspace/alveo/examples/caffe/models/FPN_CityScapes' folder.  


We need to copy the model files into 'FPN_CityScapes/quantize_results' sub-foloder using the following command lines.
```
$ cd /workspace/alveo/examples/caffe/FPN_CityScapes
$ mkdir quantize_results
$ cp -R /workspace/alveo/examples/caffe/models/FPN_CityScapes/*.* ./quantize_results/*.*
```
You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'FPN_CityScapes/quantize_results' sub-foloder.




## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following command line.
```
$ source run.sh deploy
```
All compiler files will be generated in 'work' sub folder.

And xfdnn_deploy.prototxt will be generated at root folder.

xfdnn_deploy.prototxt is to execute Caffe model on FPGA.



## Run Inference model on cpu

To run the inference model on cpu with 'cityscapes/frankfurt' images, please use the following command line.
```
$ python FPN_cpu.py 
```
The first 30 output images will be stored in cpu_output sub-folder.



## Run Inference model on FPGA 

To run the inference model on fpga with 'cityscapes/frankfurt' images, 

please use the following command line.

```
$ python FPN_fpga.py 
```
The first 30 segmentation output images will be stored in 'fpga_output' folder. 

This will also provide mean IOU between cpu output and fpga output for the first 30 images.

