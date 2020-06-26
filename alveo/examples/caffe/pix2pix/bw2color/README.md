# Running Caffe pix2pix (B/W to color) Model

### Setup

> **Note:** Skip, If you have already run the below steps.

Activate Conda Environment
  ```sh
  conda activate vitis-ai-caffe 
  ```

## Data Preparation

The input image is supposed to be black/white image.

However, we need the format of the input image as 3d array with 3 channels. 

You can put same value for the 3 channels. 

Here we providied example script to convert RGB image to B/W image.

```
cd /workspace/alveo/examples/caffe 
python convert_rgb_bw.py --image <image-file>
```



## Pix2Pix (b/w to color) model

We trained Pix2Pix (B/W to color) model with input size as [256,256,3].

After training the model, we quantized the model to deploy on FPGA.

To get the quantized Caffe model, run the following commands.

> **Note:** Skip, If you have already run the below steps.
```
cd /workspace/alveo/examples/caffe 
python getModels.py
```

The Pix2Pix (B/W to color) model files would be located in '/workspace/alveo/examples/caffe/models/bw2color' folder.


We need to copy the model files into 'pix2pix/bw2color/quantize_results' sub-foloder using the following commands.

```
cd /workspace/alveo/examples/caffe/pix2pix/bw2color
cp -R /workspace/alveo/examples/caffe/models/bw2color ./quantize_results
```

You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'bw2color/quantize_results' sub-foloder.


## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following commands.

```
cd /workspace/alveo/examples/caffe/pix2pix/bw2color
source run.sh deploy
```

All compiler files will be generated in 'work' sub folder.

And xfdnn_deploy.prototxt will be generated at root folder.

xfdnn_deploy.prototxt is to execute Caffe model on FPGA.


## Run Inference model on CPU

To run the inference model on cpu to convert B/W image to color (RGB) image, please use the following commands.

```
cd /workspace/alveo/examples/caffe/pix2pix/bw2color
python bw2color_cpu.py --image <image-file>
```


## Run Inference model on FPGA 

To run the inference model on fpga to convert B/W image to color (RGB) image, please use the following commands.

```
cd /workspace/alveo/examples/caffe/pix2pix/bw2color
python bw2color_fpga.py --image <image-file>
```
