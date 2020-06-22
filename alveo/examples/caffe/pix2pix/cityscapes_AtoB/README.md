
# Running Caffe pix2pix (cityscapes_AtoB) Model:


## Data Preparation

Download cityscapes-dataset from https://www.cityscapes-dataset.com/downloads/

Or Download cityscapes-dataset fro https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz

Please unpack the file inside of pix2pix/cityscapes_AtoB folder. 

The unpacked image files are supposed to be located as the following.  

The input image is supposed to be cityscape photo image.

```
/workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB/
```


## Pix2Pix (cityscapes_AtoB) model

We trained Pix2Pix (cityscapes_AtoB) model with input size as [256,256,3].

After training the model, we quantized the model to deploy on FPGA.

To get the quantized Caffe model, run the following command lines. 

```
$ cd /workspace/alveo/examples/caffe 
$ python getModels.py
```

The Pix2Pix (cityscapes_AtoB) model files would be located in '/workspace/alveo/examples/caffe/models/cityscapes_AtoB' folder.  


We need to copy the model files into 'pix2pix/cityscapes_AtoB/quantize_results' sub-foloder using the following command lines.
```
$ cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB
$ mkdir quantize_results
$ cp -R /workspace/alveo/examples/caffe/models/cityscapes_AtoB/*.* ./quantize_results/*.*
```

You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'cityscapes_AtoB/quantize_results' sub-foloder.



## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following command line.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB
$ source run.sh deploy
```

All compiler files will be generated in 'work' sub folder.

And xfdnn_deploy.prototxt will be generated at root folder.

xfdnn_deploy.prototxt is to execute Caffe model on FPGA.



## Run Inference model on CPU

To run the inference model on cpu with cityscape photo images, please use the following command line.
```
$ cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB
$ python cityscapes_AtoB_cpu.py --image imagefilename
```


## Run Inference model on FPGA 

To run the inference model on fpga with cityscape photo images, please use the following command line.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB
$ python cityscapes_AtoB_fpga.py --image imagefilename
```



