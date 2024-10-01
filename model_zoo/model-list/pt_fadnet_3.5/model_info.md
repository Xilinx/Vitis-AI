# FADnet


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
| Prune Ratio        | 0%                                      |
| FLOPs              | 441G                                    |
| Input Dims         | 576,960,3                               |
| FP32 Accuracy      | 0.926 EPE                               |
| INT8 Accuracy      | 1.169 EPE                               |
| Train Dataset      | SceneFlow                               |
| Test Dataset       | SceneFlow                               |
| Supported Platform | GPU                                     |
  

### Paper and Architecture 

1. Network Architecture: FADnet
 
2. Paper link: https://arxiv.org/abs/2003.10758
  
  
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

1. Visulization
    ```shell
    # Run demo
    cd code
    bash run_demo.sh
    ```

2. Evaluation
    ```shell
    # Evaluate the floating model. 
    cd code
    bash run_eval.sh
    ```

3. Training
    ```shell
    # Train the floating model. Please note that the training script uses four devices of 0, 1, 2, 3 by default. 
    # If you want to increase or decrease the GPU device used, please modify it and the conf file in code/exp_configs/.
    cd code
    bash run_train.sh
    ```

4. Quantization
    ```shell
    # Start quantization-aware training and then generate the quantized deployable model.  
    cd code
    bash run_qat.sh
    ```

5. Deployment
    ```shell
    # Evaluate the quantized deployable model and then generate the quantized xmodel.
    cd code
    bash run_deploy.sh
	```

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

1. Data preprocess infomation
  
  - Load image: skimage.io.imread()                  
  - Normalize image: torchvision.transforms.Normalize(), mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
  - Resize image: torch.nn.functional.interpolate(), height = 576, width = 960, mode = 'bilinear' 

2. Channel normalization and image warping operations are removed in our modified FADNet for model deployment on FPGA.
  
