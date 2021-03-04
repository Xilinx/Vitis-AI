
# Running Caffe pix2pix (facades_BtoA) Model:



### Setup

> **Note:** Skip, If you have already run the below steps.

Activate Conda Environment
  ```sh
  conda activate vitis-ai-caffe
  ```

Setup the Environment

  ```sh
  source /vitis_ai_home/setup/alveo/u200_u250/overlaybins/setup.sh
  ```

## Data Preparation

Download facades-dataset from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz as follows
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.
```
cd ${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA/
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz
tar -xvf facades.tar.gz
rm facades.tar.gz
```

The folder is supposed to be as the following.

```
${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA/facades/test
${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA/facades/train
${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA/facades/val
```

The downloaded images have the combination of Architectural photo and label.
To split Architectural photo and label, please run the following command lines.

```
cd ${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA/
python extract_label_facades.py
```

This will generate two subfolders in val folder. 'photo' and 'label'.
```
${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA/facades/val/photo
${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA/facades/val/label
```


## Pix2Pix (facades_BtoA) model

Pix2pix is image to image translastion using GAN [1]


[1]	Phillip Isola Jun-Yan Zhu Tinghui Zhou Alexei A. Efros: Image-to-Image Translation with Conditional Adversarial Networks (2016), arXiv:1611.07004



Facades_BtoA model translates Architectural labels to photo.

We trained Pix2Pix (facades_BtoA) model with input size as [256,256,3].

After training the model, we quantized the model to deploy on FPGA.

To get the quantized Caffe model, run the following command lines.

> **Note:** Skip, If you have already run the below steps.
```
cd ${VAI_HOME}/examples/DPUCADX8G/caffe
python getModels.py
```

The Pix2Pix (facades_BtoA) model files would be located in '${VAI_HOME}/examples/DPUCADX8G/caffe/models/facades_BtoA' folder.

Copy the model files to 'pix2pix/facades_BtoA/quantize_results' with the following commands.
```
cd ${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA
cp -R ${VAI_HOME}/examples/DPUCADX8G/caffe/models/facades_BtoA ./quantize_results
```

You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'facades_BtoA/quantize_results' sub-folder.


## Run Inference model on CPU

To run the inference model on cpu to translate Architectural label to photo image, run the following commands.
```sh
cd ${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA
python facades_BtoA_cpu.py --image <image-file>

[sample]
python facades_BtoA_cpu.py --image ./facades/val/label/1.jpg
```
The generated Architectural photo image will be stored in 'test_output' sub-folder.



## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following commands

```
cd ${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA
source run.sh deploy
```

All compiler files will be generated in 'work' sub folder.

xfdnn_deploy.prototxt (used to execute Caffe model on FPGA) will be generated at root folder.




## Run Inference model on FPGA

To run the inference model on fpga to translate Architectural label to photo image, run the following commands.

```sh
cd ${VAI_HOME}/examples/DPUCADX8G/caffe/pix2pix/facades_BtoA
python facades_BtoA_fpga.py --image <image-file>

[sample]
python facades_BtoA_fpga.py --image ./facades/val/label/1.jpg
```
