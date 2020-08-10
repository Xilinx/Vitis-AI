
# Running Caffe pix2pix (cityscapes_AtoB) Model:

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
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

The unpacked image files are supposed to be at the following location.

```
/workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB/
```

Alternatively you can download cityscapes-dataset from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz as follows
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.
```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB/
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz
tar -xvf cityscapes.tar.gz
rm cityscapes.tar.gz
```

The folder is supposed to be as the following.  

```
/workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB/cityscapes/train
/workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB/cityscapes/val
```

The downloaded images have the combination of Cityscapes Semantic photo and label. 
To split Semantic photo and label, please run the following command lines.

```
$ cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB/
$ python extract_label_semantic.py
```

This will generate two subfolders in val folder. 'photo' and 'label'. 
```
/workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB/cityscapes/val/photo
/workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB/cityscapes/val/label
```  


## Pix2Pix (cityscapes_AtoB) model

Pix2pix is image to image translastion using GAN [1]


[1]	Phillip Isola Jun-Yan Zhu Tinghui Zhou Alexei A. Efros: Image-to-Image Translation with Conditional Adversarial Networks (2016), arXiv:1611.07004



Cityscapes_AtoB model translates photo to semantic label. 



We trained Pix2Pix (cityscapes_AtoB) model with input size as [256,256,3].

After training the model, we quantized the model to deploy on FPGA.

To get the quantized Caffe model, run the following command lines. 

> **Note:** Skip, If you have already run the below steps.
```
cd /workspace/alveo/examples/caffe
python getModels.py
```

The Pix2Pix (cityscapes_AtoB) model files would be located in '/workspace/alveo/examples/caffe/models/cityscapes_AtoB' folder.

Copy the model files to 'pix2pix/cityscapes_AtoB/quantize_results' with the following commands.
```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB
cp -R /workspace/alveo/examples/caffe/models/cityscapes_AtoB ./quantize_results
```

You can find deploy.prototxt, deploy.caffemodel, and quantize_info.txt in 'cityscapes_AtoB/quantize_results' sub-folder.



## Run Inference model on CPU


To run the inference model on cpu for translating photo to semantic label, run the following commands.

```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB
python cityscapes_AtoB_cpu.py --image <image-file>
For example, 
$ python cityscapes_AtoB_cpu.py --image ./cityscapes/val/photo/1.jpg
```

The generated semantic label image will be stored in 'test_output' sub-folder.


## Compilation and Partitioning


The quantized caffemodel need to be compiled and partitioned at your local drive using the following commands

```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB
source run.sh deploy
```

All compiler files will be generated in 'work' sub folder.

xfdnn_deploy.prototxt (used to execute Caffe model on FPGA) will be generated at root folder.





## Run Inference model on FPGA 

To run the inference model on fpga for translating photo to semantic label, run the following commands.

```
cd /workspace/alveo/examples/caffe/pix2pix/cityscapes_AtoB
python cityscapes_AtoB_fpga.py --image <image-file>
For example, 
$ python cityscapes_AtoB_fpga.py --image ./cityscapes/val/photo/1.jpg
```
The generated semantic label image will be stored in 'test_output' sub-folder.
