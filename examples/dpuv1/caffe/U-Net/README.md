# Running U-Net segmentation model on FPGA 


## Activate Caffe conda environment

Please activate Caffe conda environment using the following commands.

```
conda activate vitis-ai-caffe
source /workspace/alveo/overlaybins/setup.sh
cd /workspace/alveo/examples/caffe/U-Net
```



## Data Preparation

Download PhC-U373 dataset from http://celltrackingchallenge.net/2d-datasets/ 
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

The dataset has two subsets: '01' and '02'. 

The original paper [1] and we used '01' subset. 

The '01' folder has 115 light microscopic images as tif format.
The '01_ST/SEG' folder has its segmentation images as tif format. 

Please unzip the zip file of the PhC-U373 dataset inside of U-Net folder. 
The folder is supposed to be as the following.  

```
/workspace/alveo/examples/caffe/U-Net/PhC-C2DH-U373/01
/workspace/alveo/examples/caffe/U-Net/PhC-C2DH-U373/01_GT
/workspace/alveo/examples/caffe/U-Net/PhC-C2DH-U373/01_ST
/workspace/alveo/examples/caffe/U-Net/PhC-C2DH-U373/02
/workspace/alveo/examples/caffe/U-Net/PhC-C2DH-U373/02_GT
/workspace/alveo/examples/caffe/U-Net/PhC-C2DH-U373/02_ST
```

For training/interference, we need to make two subfolders. 'Img' and 'Seg'. 

```
cd /workspace/alveo/examples/caffe/U-Net
python convert_dataset.py
```

All the microscopic images will be copied to 'Img' folder after converting to png format.

All the segmentation images will be copied to 'Seg' folder after renaming the files to the same filename in 'Img' folder.  


[1]	Olaf Ronneberger, Philipp Fischer, and Thomas Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation (2015), arXiv:1505.04597



## U-Net model

The original paper used Caffe framework to train the model.

In this work, we trained the U-Net network using Keras framework.

For detail information about U-Net architecture and training procedure, please refer to unet_whitepaper.

We trained U-Net model with input size as (256, 256).

We splited the dataset as train dataset and test dataset. The train dataset has 80 files and test dataset has 35 files. 

After training the model, we converted Keras model to Caffemodel.

To get the pre-trained Caffe model, run the following command lines. 

```
cd /workspace/alveo/examples/caffe 
python getModels.py
```

The U-Net model files would be located in '/workspace/alveo/examples/caffe/models/U-Net' folder.  


We need to copy the model files into 'U-Net/float' sub-foloder using the following command lines.
```
cd /workspace/alveo/examples/caffe/U-Net
cp -R /workspace/alveo/examples/caffe/models/U-Net ./float
```
You can find unet_U373_256.prototxt and unet_U373_256.caffemodel in 'U-Net/float' sub-foloder.



## Run Caffe model on CPU

To run the Caffe model on CPU with the test images, please use the following command line.

```
cd /workspace/alveo/examples/caffe/U-Net
python unet_caffe_cpu.py 
```

This will provide mean IOU for the test dataset.

Also, it will generate sample output image in 'U-Net/test_output' sub-folder.





## Quantization

To run the Caffemodel on FPGA, the Caffemodel need to be quantized using the following command. 

```
DECENT_DEBUG=1 vai_q_caffe quantize -model ./float/dummy_256.prototxt -weights ./float/unet_U373_256.caffemodel -input_blob "input_1" -method 1 -calib_iter 100
```

‘-method’ is the option for quantization, 0: Non-Overflow, 1: Min-Diff. The default is 1. 



After 100 iteration of calibrating input images, quantize_info.txt, deploy.prototxt, and deploy.caffemodel would be generated at 'quantize_results' sub-folder. 

quantize_train_test.prototxt and quantize_train_test.caffemodel would be also generated in 'quantize_results' folder. 

The quantize_train_test files are a kind of simulation version of quantization which can be exectued on CPU. 


Using any text editor, the input block of the deploy.prototxt should be updated as the following.

```
layer {
  name: "input_1"
  type: "Input"
  top: "input_1"
#  transform_param {
#    scale: 0.0078431
#    crop_size: 256
#  }
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 256
      dim: 256
    }
  }
}
```

## Run Caffe model of quantization on CPU

You can check the output of deploy.prototxt and deploy.caffemodel using unet_caffe_cpu.py.
Using any text editor, open unet_caffe_cpu.py and modify as following.

```
#model_def = './float/unet_U373_256.prototxt'
#model_weights = './float/unet_U373_256.caffemodel'

model_def = './quantize_results/deploy.prototxt'
model_weights = './quantize_results/deploy.caffemodel'
```
Then, run the following command to get the sample output image and mean IOU for the test dataset.
```
python unet_caffe_cpu.py 
```

In addition, You can check the output of quantize_train_test.prototxt and quantize_train_test.caffemodel using unet_caffe_cpu.py.
Using any text editor, the input block of the quantize_train_test.prototxt should be updated as the following.
```
#layer {
#  name: "input_1"
#  type: "ImageData"
#  top: "input_1"
#  top: "label"
#  include {
#    phase: TRAIN
#  }
#  transform_param {
#    scale: 0.0078431
#    crop_size: 256
#  }
#  image_data_param {
#    source: "/workspace/alveo/examples/caffe/U-Net/U373_list.txt"
#    batch_size: 1
#    shuffle: false
#    root_folder: "/workspace/alveo/examples/caffe/U-Net/PhC-C2DH-U373/Img/"
#  }
#}
layer {
  name: "input_1"
  type: "Input"
  top: "input_1"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 256
      dim: 256
    }
  }
}
```
Then, open unet_caffe_cpu.py and modify as following.

```
#model_def = './float/unet_U373_256.prototxt'
#model_weights = './float/unet_U373_256.caffemodel'

model_def = './quantize_results/quantize_train_test.prototxt'
model_weights = './quantize_results/quantize_train_test.caffemodel'
```



## Compilation and Partitioning

The quantized caffemodel need to be compiled and partitioned at your local drive using the following command line after updating the input block of deploy.prototxt as instructed as above.

```
source run.sh deploy
```

All compiler files will be generated in 'work' sub folder.

And xfdnn_deploy.prototxt will be generated at root folder.

xfdnn_deploy.prototxt is to execute Caffe model on FPGA.



## Run Inference model on FPGA 

To run the inference model on FPGA with the test images, please use the following command line.

```
python unet_caffe_fpga.py 
```

This will provide mean IOU on FPGA for the test dataset.

Also, it will generate sample output image on FPGA in 'test_output' folder.
