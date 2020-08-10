# Running FPN CityScapes segmentation model on FPGA 


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

Download cityscapes-dataset (leftImg8bit_trainvaltest.zip) from https://www.cityscapes-dataset.com/downloads/
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

You need to register for the website to download the dataset.


The unpacked image files are supposed to be at the following location.

```
/workspace/alveo/examples/caffe/FPN_CityScapes/leftImg8bit/val/frankfurt
```

> **Note:** This model was trained using cityscapes/frankfurt dataset. The alternative dataset might not provide correct output.

Alternatively you can download cityscapes-dataset from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz as follows
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.
```
cd /workspace/alveo/examples/caffe/FPN_CityScapes/
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz
tar -xvf cityscapes.tar.gz
rm cityscapes.tar.gz
```

The folder is supposed to be as the following.  

```
/workspace/alveo/examples/caffe/FPN_CityScapes/cityscapes/train
/workspace/alveo/examples/caffe/FPN_CityScapes/cityscapes/val
```

The downloaded images have the combination of Cityscapes Semantic photo and label. 
To split Semantic photo and label, please run the following command lines.

```
$ cd /workspace/alveo/examples/caffe/FPN_CityScapes/
$ python extract_label_semantic.py
```

This will generate two subfolders in val folder. 'photo' and 'label'. 
```
/workspace/alveo/examples/caffe/FPN_CityScapes/cityscapes/val/photo
/workspace/alveo/examples/caffe/FPN_CityScapes/cityscapes/val/label
```  



## FPN CityScapes segmentation model

We trained FPN CityScapes model with input size as [256,512,3].

After training the model, we quantized the model to deploy on FPGA.

> **Note:** Skip, If you have already run the below steps.

To get the quantized Caffe model, run the following command lines. 

```
cd /workspace/alveo/examples/caffe 
python getModels.py
```

The FPN CityScapes model files would be located in '/workspace/alveo/examples/caffe/models/FPN_CityScapes' folder.  


We need to copy the model files into 'FPN_CityScapes/quantize_results' sub-foloder using the following command lines.
```
cd /workspace/alveo/examples/caffe/FPN_CityScapes
cp -R /workspace/alveo/examples/caffe/models/FPN_CityScapes ./quantize_results
```
You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'FPN_CityScapes/quantize_results' sub-foloder.


## Run Inference model on CPU


To run the inference model on cpu with 'cityscapes/frankfurt' images, please update FPN_cpu.py using any text editor as following.

```
line97, img_path = './leftImg8bit/val/frankfurt/'
ling98, #img_path = './cityscapes/val/photo/'
```

Then, use the following command line.
```
python FPN_cpu.py 
```


To run the inference model on cpu with 'cityscapes/val/photo' images, please use the following command line.
```
python FPN_cpu.py 
```


The first 30 output images will be stored in 'cpu_output' sub-folder.



## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following command line.
```
source run.sh deploy
```
All compiler files will be generated in 'work' sub folder.

And xfdnn_deploy.prototxt will be generated at root folder.

xfdnn_deploy.prototxt is to execute Caffe model on FPGA.





## Run Inference model on FPGA 

To run the inference model on fpga with 'cityscapes/frankfurt' images, please update FPN_fpga.py using any text editor as following.

```
line124, img_path = './leftImg8bit/val/frankfurt/'
ling125, #img_path = './cityscapes/val/photo/'
```

Then, use the following command line.
```
python FPN_fpga.py 
```

To run the inference model on fpga with 'cityscapes/val/photo' images, please use the following command line.

```
python FPN_fpga.py 
```
The first 30 segmentation output images will be stored in 'fpga_output' folder. 

This will also provide mean IOU between cpu output and fpga output for the first 100 images.

