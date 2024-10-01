# MobileNet v3 

### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Classic Image Classification
   - Trained on ImageNet dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | TensorFlow2                             |
| Prune Ratio        | 0%                                      |
| FLOPs              | 132M                                    |
| Input Dims (H W C) | 224,224,3                               |
| FP32 Accuracy      | 0.6756/0.8728 (top1/top5)               |
| INT8 Accuracy      | 0.6513/0.8543 (top1/top5)               |
| Train Dataset      | ImageNet ILSVRC2012 Train               |
| Test Dataset       | ImageNet ILSVRC2012 Val                 |
| Supported Platform | GPU                                     |
  

### Paper and Architecture 

1. Network Architecture: MobileNet v3
 
2. Paper link: https://arxiv.org/abs/1905.02244
  
  
### Dataset Preparation

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
  
  
### Use Guide

1. Evaluate pb model.
  ```shell
  cd code/test
  bash run_eval_by_images_h5.sh
  ```

### Quantization

**vai_q_tensorflow2** is required, see 
[Vitis AI User Document](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html#documentation) for installation guide.

1. Quantize h5 model.
  ```shell
  cd code/quantize
  bash run_quantize_by_images_h5.sh
  ```

2. Evaluate quantized h5 model.
  ```shell
  cd code/quantize
  bash run_quantize_eval_by_images_h5.sh
  ```

3. Dump quantized golden results.
  ```shell
  cd code/quantize
  bash run_quantize_dump_by_images_h5.sh
  ```

    
### License

Non-Commercial Use Only

For details, please refer to **[AMD license agreement for non commercial models](https://github.com/Xilinx/Vitis-AI/blob/master/model_zoo/AMD-license-agreement-for-non-commercial-models.md)**


### Note

Data preprocess

  ```
  data channel order: RGB(0~255)                  
  resize: height=0.875*img_height, width=0.875*img_width
  center crop: 224 * 224
  input = input - 127.5
  input = input /255.0 * 2 (rescale to [-1, 1])
  ```
   

