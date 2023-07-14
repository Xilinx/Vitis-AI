# PSMnet


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Disparity Estimation
   - Trained on SceneFlow dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 68%                                     |
| FLOPs              | 696G                                    |
| Input Dims         | 576,960,3                               |
| FP32 Accuracy      | 1.064 EPE                               |
| INT8 Accuracy      | 1.049 EPE                               |
| Train Dataset      | SceneFlow                               |
| Test Dataset       | SceneFlow                               |
| Supported Platform | GPU                                     |
  

### Paper and Architecture 

1. Network Architecture: PSMnet
 
2. Paper link: https://arxiv.org/abs/1803.08669
  
  
### Dataset Preparation

1. Dataset description

    ```
    Usage of Scene Flow dataset (https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
    Download RGB cleanpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa. Organize them as follows:
    - FlyingThings3D_release/frames_cleanpass
    - FlyingThings3D_release/disparity
    - driving_release/frames_cleanpass
    - driving_release/disparity
    - monkaa_release/frames_cleanpass
    - monkaa_release/disparity
    Put them in the data/ folder (or soft link).
    ```

2. Dataset diretory structure
   ```
   + data
     + driving_release
     + FlyingThings3D_release
     + monkaa_release
    ```

	

### Use Guide

Note:
1) The code supports both single-gpu and multi-gpu configs.
2) The code supports two quantization strategies: post-training quantization (PTQ) and quantization-aware training (QAT)
3) For sanity check, you can just train 1-batch and test 1-batch data.

```shell
cd code
```

1. Train unpruned floating model
    ```shell
    # Use single GPU
    bash train_psmnet_single.sh
    # Use multi GPUs
    bash train_psmnet_multi.sh
    ```

2. Test unpruned floating model
    ```shell
    bash test_psmnet.sh
    ```

3. Test pruned model
    ```shell
    bash test_psmnet_prune.sh
    ```

4. Post-Training Quantization
    - Use single GPU and fixed batch size = 1 for this step. Set env of DUMP_XMODEL=0 for quantization and DUMP_XMODEL=1 for dump xmodel in test_psmnet_ptq.sh.
    ```shell
    # Calibration: --qat 0 --quant_mode calib --dump_xmodel 0 --device gpu --quant_dir ptq_result --deploy 0
    bash test_psmnet_ptq.sh
    # Test quantized model: --qat 0 --quant_mode test --dump_xmodel 0 --device gpu --quant_dir ptq_result --deploy 0
    bash test_psmnet_ptq.sh
    # Dump xmodel: --qat 0 --quant_mode test --dump_xmodel 1 --device cpu --quant_dir ptq_result --deploy 0
    bash test_psmnet_ptq.sh
    ```

5. Quantization-Aware Training
    - Set env of DUMP_XMODEL=0 for quantization and DUMP_XMODEL=1 for dump xmodel.
    ```shell
    # Training: --qat 1 --quant_mode calib --dump_xmodel 0 --device gpu --quant_dir qat_result --deploy 0
    # Use single GPU
    bash train_psmnet_qat_single.sh
    # Use multi GPUs
    bash train_psmnet_qat_multi.sh
    # Convert quantizated model to deployable model and test (Use 1 GPU and fixed batch size = 1 for this step): --qat 1 --quant_mode test --dump_xmodel 0 --device gpu --quant_dir qat_result --deploy 1
    bash test_psmnet_qat.sh
    # Dump xmodel: --qat 1 --quant_mode test --dump_xmodel 1 --device cpu --quant_dir qat_result --deploy 1
    bash test_psmnet_qat.sh
    ```
	

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

Data preprocess infomation
  
  - Load image: skimage.io.imread()                  
  - Normalize image: torchvision.transforms.Normalize(), mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
  - Resize image: torch.nn.functional.interpolate(), height = 576, width = 960, mode = 'bilinear' 
  
  
  
