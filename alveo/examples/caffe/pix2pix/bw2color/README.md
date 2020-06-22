# Running Caffe pix2pix (b/w to color) Model



## Data Preparation

The input image is supposed to be black/white image.

However, we need the format of the input image as 3d array with 3 channels. 

You can put same value for the 3 channels. 



## Pix2Pix (b/w to color) model

We trained Pix2Pix (b/w to color) model with input size as [256,256,3].

After training the model, we quantized the model to deploy on FPGA.

To get the quantized Caffe model, run the following command lines. 

```
$ cd /workspace/alveo/examples/caffe 
$ python getModels.py
```

The Pix2Pix (b/w to color) model files would be located in '/workspace/alveo/examples/caffe/models/bw2color' folder.  


We need to copy the model files into 'pix2pix/bw2color/quantize_results' sub-foloder using the following command lines.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/bw2color
$ mkdir quantize_results
$ cp -R /workspace/alveo/examples/caffe/models/bw2color/*.* ./quantize_results/*.*
```

You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'bw2color/quantize_results' sub-foloder.


## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following command line.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/bw2color
$ source run.sh deploy
```

All compiler files will be generated in 'work' sub folder.

And xfdnn_deploy.prototxt will be generated at root folder.

xfdnn_deploy.prototxt is to execute Caffe model on FPGA.


## Run Inference model on CPU

To run the inference model on cpu with b/w images, please use the following command line.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/bw2color
$ python bw2color_cpu.py --image imagefilename
```


## Run Inference model on FPGA 

To run the inference model on fpga with b/w images, please use the following command line.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/bw2color
$ python bw2color_fpga.py --image imagefilename
```




