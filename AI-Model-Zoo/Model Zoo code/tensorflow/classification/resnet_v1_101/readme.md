### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. environment requirement 
   tensorflow 1.9+(<1.14)
   others libs found in code/conda_config/enveriment.yaml.
   
2. conda quickly install(optional).
  ```shell
  conda env create -f code/conda_config/enveriment.yaml
  ```

### Preparation

1. dataset describe.
  ```
  dataset includes image file and groundtruth file.
  image file: put train/val images.
  groundtruth file: format as "imagename label"
  ```
2. prepare testdatset.
   
   ```
   cd code/gen_data

   # check soft link vaild or 
   # user download validation dataset, rename it with "validation".

   # dataset put or generated in data fileholder.
   # validation dataset fortmat: A big file(validation) includes 1000 subfiles, echo subfile put same class images.  
   # run get_dataset.sh, then will generate required dataset for model.

   bash get_dataset.sh 
   ```

### Train/Eval
1. Evaluate pb model.
  ```shell
  cd code/test
  bash run_eval_pb.sh
  ```

### performance

|Acc |Claimed on Imagenet| Ckpt on Imagenet| Pb on Imagenet|
|----|----|---|---|
|Recall_1(%)|75.2|75.202|75.202|
|Recall_5(%)|92.2|92.194|92.194|

### Model_info

1.  data preprocess
  ```
  data channel order: RGB(0~255)                  
  resize: short side reisze to 256 and keep the aspect ratio.(tf.image.resize_bilinear(image, [height, width], align_corners=False))
  center crop: 224 * 224
  input = input - [123.68, 116.78, 103.94] 
  ```
2. node information

  ```
  The name of input node: 'input:0'
  The name of output node: 'resnet_v1_101/predictions/Reshape_1:0'
  ```
