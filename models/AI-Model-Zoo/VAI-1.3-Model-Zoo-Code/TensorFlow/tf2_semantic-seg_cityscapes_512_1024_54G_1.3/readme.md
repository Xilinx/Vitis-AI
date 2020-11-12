### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Environment requirement 
   tensorflow 2.3
   
2. Installation
   - Create virtual envrionment and activate it:
   ```shell
   conda create -n tf2.3 python=3.6
   conda activate tf2.3
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```

### Preparation

1.Prepare datset.
  
  Cityscapes dataset link: [Cityscapes](https://www.cityscapes-dataset.com/) 
  
  ```
  a.Users need to download the Cityscapes dataset by themselves as it needs registration. The script of get_dataset.sh can not automatically download the dataset. 
  b.The downloaded Cityscapes dataset needs to be organized as the following format:
    1) Create a folder of "data" along the "code" folder. Put the processed data in +data/.

  ```

### Preparation

1. dataset describle.
  ```
  dataset includes image file and groundtruth file.
  image file: put train/val images.
  groundtruth file: put train/val labels
  ```
2. prepare datset.

  ```shell
  cd code/

#check dataset soft link or                                                    \
    user download cityscapes dataset(                                          \
        https: // www.cityscapes-dataset.com/downloads)
#grundtruth folder : gtFine_trainvaltest.zip[241MB]
#image folder : leftImg8bit_trainvaltest.zip[11GB]
#put the grundtruth folder and image folder in  `data` directory.
#run get_data.sh, then will generate required dataset for model.

  bash get_data.sh 

  * `data` directory structure like:
     + cityscapes_20cls
       + train_images
       + train_masks
       + val_images
       + val_masks

#Users also can use their own scripts to preprocess the dataset as the above   \
    format.
  ```

### Train/Eval
1. Evaluate h5 model.
  ```shell
  bash run_eval.sh
  ```

2. Training model.
  ```shell
  bash run_train.sh
  ```
### Quantization
**vai_q_tensorflow2** is required, see 
[Vitis AI User Document](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html#documentation) for installation guide.

1. Quantize model.
  ```shell
  bash run_quantize.sh
  ```

2. Evaluate quantized model.
  ```shell
  bash run_quantize_eval.sh
  ```

3. Dump quantized golden results.
  ```shell
  bash run_quantize_dump.sh
  ```


### performance

| Method | Input size | FLOPs | Val. mIoU |
|--------|------------|-----------|-------|
| ERFNet | 512x1024 | 54G | 52.98% |

### Model_info

1.  data preprocess
  ```
  data channel order: RGB(0~255)                  
  resize: reisze image to H*W = 512*1024.(tf.image.resize_bilinear(image, [height, width], align_corners=False))
  input = input - [123.68, 116.78, 103.94] 
  input /= 255.0
  ```
