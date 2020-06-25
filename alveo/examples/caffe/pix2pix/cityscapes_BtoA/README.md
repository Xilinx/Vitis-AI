
# Running Caffe pix2pix (cityscapes_BtoA) Model:

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

Download cityscapes-dataset from https://www.cityscapes-dataset.com/downloads/

The unpacked image files are supposed to be at the following location.
```
/workspace/alveo/examples/caffe/pix2pix/cityscapes_BtoA/
```

Alternatively you can download cityscapes-dataset from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz as follows
```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_BtoA/
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz
tar -xvf cityscapes.tar.gz
rm cityscapes.tar.gz
```

## Pix2Pix (cityscapes_BtoA) model

We trained Pix2Pix (cityscapes_BtoA) model with input size as [256,256,3].

After training the model, we quantized the model to deploy on FPGA.

To get the quantized Caffe model, run the following command lines. 

> **Note:** Skip, If you have already run the below steps.
```
cd /workspace/alveo/examples/caffe
python getModels.py
```

The Pix2Pix (cityscapes_BtoA) model files would be located in '/workspace/alveo/examples/caffe/models/cityscapes_BtoA' folder.

Copy the model files to 'pix2pix/cityscapes_BtoA/quantize_results' with the following commands.
```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_BtoA
mkdir quantize_results
cp -R /workspace/alveo/examples/caffe/models/cityscapes_BtoA/*.* ./quantize_results
```

You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'cityscapes_BtoA/quantize_results' sub-folder.



## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following commands

```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_BtoA
source run.sh deploy
```

All compiler files will be generated in 'work' sub folder.

xfdnn_deploy.prototxt (used to execute Caffe model on FPGA) will be generated at root folder.


## Run Inference model on CPU

To run the inference model on cpu with cityscape photo images, run the following commands.
```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_BtoA
python cityscapes_BtoA_cpu.py --image <image-file>
```


## Run Inference model on FPGA 

To run the inference model on fpga with cityscape photo images, run the following commands.

```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_BtoA
python cityscapes_BtoA_fpga.py --image <image-file>
```
