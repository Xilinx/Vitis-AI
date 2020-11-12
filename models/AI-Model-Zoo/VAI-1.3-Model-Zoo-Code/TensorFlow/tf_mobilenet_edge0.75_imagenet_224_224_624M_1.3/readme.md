### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)
6. [Quantize](#quantize)

### Installation
1. Environment requirement 
   tensorflow 1.15
   
2. Installation
   - Create virtual envrionment and activate it:
   ```shell
   conda create -n tf_cls python=3.6
   conda activate tf_cls
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```

### Preparation

1.Prepare datset.
  
  ImageNet dataset link: [ImageNet](http://image-net.org/download-images) 
  
  ```
  a.Users need to download the ImageNet dataset by themselves as it needs registration. The script of get_dataset.sh can not automatically download the dataset. 
  b.The downloaded ImageNet dataset needs to be organized as the following format:
    1) Create a folder of "data" along the "code" folder. Put the validation data in +data/validation.
    2) Data from each class for validation set needs to be put in the folder:
    +data/validation
         +validation/n01847000 
         +validation/n02277742
         +validation/n02808304
         +... 
  ```
  
2.Preprocess dataset.

  ```shell
  cd code/gen_data
  # run get_dataset.sh and preprocess the dataset for model requirement.
  bash get_dataset.sh 
  ```
  
  ```
  The generated data directory structure after preprocessing is like as follows:
  +data/Imagenet/   
       +Imagenet/val_dataset #val images. 
       +Imagenet/val.txt #val gt file.
  
  gt file is like as follows: 
    ILSVRC2012_val_00000001.JPEG 65
    ILSVRC2012_val_00000002.JPEG 970
    ILSVRC2012_val_00000003.JPEG 230
    ILSVRC2012_val_00000004.JPEG 809
    ILSVRC2012_val_00000005.JPEG 516
    
  # Users also can use their own scripts to preprocess the dataset as the above format.
  ```

### Train/Eval

1. Evaluate pb model.
    ```shell
    cd code/test
    bash run_eval_pb.sh
    ```

### performance

|MobilenetEdge_0.75 | Ckpt on Imagenet| Pb on Imagenet|
|--------|---|---|
|Recall_1(%)|72.01|72.01|
|Recall_5(%)|90.812|90.812|

### Model_info

1.  data preprocess
  ```
  data channel order: RGB(0~255)                  
  input = input / 255
  crop: crop the central region of the image with an area containing 87.5% of the original image.
  resize: 224 * 224 (tf.image.resize_bilinear(image, [height, width], align_corners=False)) 
  input = 2*(input - 0.5)                          
  ``` 
2. node information
  ```
  The name of input node: 'input:0'
  The name of output node: 'MobilenetEdgeTPU/Predictions/Reshape_1:0'
  ```

### Quantize
1- Quantize tool installation
  See [vai_q_tensorflow](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Quantizer/vai_q_tensorflow)

2- Quantize workspace
  See [quantize](./code/quantize/)
