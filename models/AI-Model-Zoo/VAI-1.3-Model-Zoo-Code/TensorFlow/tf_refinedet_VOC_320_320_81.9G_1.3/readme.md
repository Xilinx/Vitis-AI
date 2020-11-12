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
   conda create -n tf_ssd python=3.6
   conda activate tf_ssd
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```
### Preparation

1. dataset describe.
  ```
  dataset includes VOC2007-test image files.
  image file: put images in data/VOC/images.
  ```
2. prepare testdatset.
   
   ```
   cd code/gen_data

   # download voc2007 test images and put in data/VOC/images. 

   bash get_dataset.sh 
   ```

### Train/Eval
1. Demo.

  ```shell
  cd code/test
  python demo.py --demo-image "path/to/image"
  ```
2. Evaluate pb model.
  ```shell
  cd code/test
  python test.py
  ```

### performance

|MAP|voc2007 testdataset|
|----|----|
|11point|80.15%|


### Model_info

1. data preprocess
  ```
  1. data channel order: BGR(0~255)                  
  2. resize: 320 * 320 (cv2.resize(), INTER_LINEAR)
  3. image = image - [103.94, 116.78, 123.68] 
  ``` 
2. node information
  ```
  The name of input node: 'image:0'
  the name of output node: 'arm_cls:0', 'arm_loc:0', 'odm_cls:0', 'odm_loc:0'.
  ```

### Quantize
1- Quantize tool installation
  See [vai_q_tensorflow](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Quantizer/vai_q_tensorflow)

2- Quantize workspace
  See [quantize](./code/quantize/)
