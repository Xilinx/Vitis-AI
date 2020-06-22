# Running Caffe pix2pix (facades_BtoA) Model:




## Data Preparation

Download facades dataset from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz

Please unpack the file inside of pix2pix/facades_BtoA folder. 

The unpacked image files are supposed to be located as the following.  

```
/workspace/alveo/examples/caffe/pix2pix/facades_BtoA/
```


## Pix2Pix (facades_BtoA) model

We trained Pix2Pix (facades_BtoA) model with input size as [256,256,3].

After training the model, we quantized the model to deploy on FPGA.

To get the quantized Caffe model, run the following command lines. 

```
$ cd /workspace/alveo/examples/caffe 
$ python getModels.py
```

The Pix2Pix (facades_BtoA) model files would be located in '/workspace/alveo/examples/caffe/models/facades_BtoA' folder.  


We need to copy the model files into 'pix2pix/facades_BtoA/quantize_results' sub-foloder using the following command lines.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/facades_BtoA
$ mkdir quantize_results
$ cp -R /workspace/alveo/examples/caffe/models/facades_BtoA/*.* ./quantize_results/*.*
```

You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'facades_BtoA/quantize_results' sub-foloder.



## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following command line.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/facades_BtoA
$ source run.sh deploy
```

All compiler files will be generated in 'work' sub folder.

And xfdnn_deploy.prototxt will be generated at root folder.

xfdnn_deploy.prototxt is to execute Caffe model on FPGA.




## Run Inference model on CPU

To run the inference model on cpu with facades images, please use the following command line.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/facades_BtoA
$ python facades_BtoA_cpu.py --image imagefilename
```


## Run Inference model on FPGA 

To run the inference model on fpga with facades images, please use the following command line.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/facades_BtoA
$ python facades_BtoA_fpga.py --image imagefilename
```




