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
  
   dataset link: [Nuclei Cell](https://www.kaggle.com/advaitsave/tensorflow-2-nuclei-segmentation-unet) 
  
  ```
  a. Download the Nuclei Cell dataset by themselves as it needs registration.
  b. Create a folder of "data" along the "code" folder. Put the downloadeed data in +data/.

  ```

### Preparation

1. prepare datset.

  ```shell
  cd code/

  # check dataset soft link or user download Nuclei Cell dataset

  * `data` directory structure like:
    + nuclei_data 
       + stage1_test
       + stage1_train (only this one is enough)
       + stage2_test_final
  ```

### Train/Eval
1. Test h5 model.
  ```shell
  bash run_test.sh
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

| Method | Input size | FLOPs |Val. mIoU | 
|--------|------------|-----------|----------|
| UNet | 128x128 | 5.31G | 39.68% |

### Model_info

1.  data preprocess
  ```
  data channel order: RGB(0~255)                  
  resize: reisze image to H*W = 128*128.(tf.image.resize_bilinear(image, [height, width], align_corners=False))
  ```
