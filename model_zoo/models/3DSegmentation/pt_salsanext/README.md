# Contents

- [Contents](#contents)
- [Model Description](#model-description)
  - [Description](#description)
  - [Paper](#paper)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
  - [Structure](#structure)
  - [Inference Process](#inference-process)
- [Performance](#performance)
- [Links](#links)
- [Vitis AI Model Zoo Homepage](#vitis-ai-model-zoo-homepage)

# Model Description

## Description
Salsanext is a deep learning model designed for semantic segmentation tasks in autonomous driving applications. 
It efficiently and accurately labels each pixel of an input image with a corresponding semantic class, enabling the 
perception system of self-driving cars to understand and interpret the surrounding environment.
## Paper

 Cortinhal, Tiago, George Tzelepis, and Eren Erdal Aksoy. 
 "Salsanext: Fast, uncertainty-aware semantic segmentation of lidar point clouds for autonomous driving." 
 arXiv preprint arXiv:2003.03653 (2020). Link: https://arxiv.org/abs/2003.03653

# Model Architecture
The architecture of SalsaNext consists of two main components: the SalsaNet and the UNet. 
The SalsaNet is responsible for feature extraction from the input point cloud, capturing both local and contextual information. 
The UNet is a fully convolutional network that takes the features from SalsaNet and performs multi-scale fusion to generate dense 
semantic segmentation predictions. The model utilizes skip connections and employs a decoder-like structure to capture spatial 
information effectively.
# Dataset

Dataset for testing: Semantic Kitti. The Semantic Kitti dataset is a widely used benchmark dataset for evaluating semantic 
segmentation algorithms in the context of autonomous driving. It consists of high-resolution point cloud sequences collected 
from a Velodyne LiDAR sensor mounted on a moving vehicle. 

Link to download the  dataset: http://www.semantic-kitti.org/

# Features

The notable features of the Salsanext model:

1. **Uncertainty estimation**: The model incorporates a Monte Carlo Dropout technique to estimate the uncertainty associated with its predictions. 
2. **Contextual information integration**: SalsaNext leverages contextual information from neighboring points to enhance the segmentation accuracy. 
3. **Multi-scale fusion**: The UNet component of SalsaNext incorporates skip connections and performs multi-scale fusion, enabling the model to capture features at different scales. 
4. **Fast inference speed**: SalsaNext is designed to provide real-time semantic segmentation of LiDAR point clouds.
5. **Regularization techniques**: The model employs techniques such as dropout and batch normalization to mitigate overfitting and improve generalization capabilities.

# Environment Requirements

Before running the model inference, make sure that the latest version of
[Vitis-AI](https://xilinx.github.io/Vitis-AI/docs/install/install.html) is installed and the host computer fully supports
Xilinx FPGA/ACAP and the appropriate accelerator is installed correctly, e.g. Xilinx VCK5000 Versal.

# Quick Start

Follow the [Quick Start guide](../../../README.md#quick-start) in the main Model Zoo README:

1. Install the Vitis-AI
2. Run the docker container
3. Download test data
4. Run the inference of the model

# Script Description

## Structure

```text
pt_salsanext                  # model name  
├── artifacts                 # artifacts - will be created during the inference process
│ ├── inference               # folder with results values of inference and evaluation
│ │ ├── performance           # model productivity measurements
│ │ ├── results               # model inference results files
│ │ └── vaitrace              # vaitrace profiling performance reports
│ └── models                  # folder with model meta and .xmodel executable files
├── scripts                   # scripts for model processing 
│ ├── inference.sh            # model inference
│ ├── performance.sh          # model performance report
│ └── vaitrace.sh             # model profiling via vaitrace
├── config.env                # model configuration - env variables
└── README.md
```

## Inference Process

- Native inference - follow the [Quick Start guide](../../../README.md#quick-start) in the main Model Zoo. <br>
  To run the inference [at the step 8](../../../README.md#quick-start) for the salsanext model:
   ```
   bash scripts/inference.sh \
       $MODEL_FOLDER/artifacts/models/salsanext_pt/salsanext_pt.xmodel /workspace/Vitis-AI-Library/samples/3DSegmentation/salsanext_input/holder.jpg
   ``` 
  >   **Warning**
  >   In the directory of the test image, there **must** be the following extra 3D point cloud information:
    ```text
    salsanext_input               # folder with the image
    ├── holder.jpg                # test image
    ├── scan_x.txt                # x-component of the points
    ├── scan_y.txt                # y-component of the points
    ├── scan_z.txt                # z-component of the points
    └── scan_remission.txt        # intensity of the laser beam reflected from the objects in the environment
    ```
- AMD Server


# Performance

- You can profile the model using [vaitrace](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Starting-a-Simple-Trace-with-vaitrace) perfomance report:
  To run the Vaitrace, use: 
   ```
   # Format: bash scripts/vaitrace.sh <MODEL_PATH> <TEST_IMAGE_PATH>
   # where:
   # <MODEL_PATH> - The path to the model file .xmodel
   # <TEST_IMAGE_PATH> - The path to the image to be processed via vaitrace. 
   # The report files will be stored in the $MODEL_FOLDER/artifacts/inference/vaitrace folder
   # Example: 
   
   bash scripts/vaitrace.sh $MODEL_FOLDER/artifacts/models/salsanext_pt/salsanext_pt.xmodel /workspace/Vitis-AI-Library/samples/3DSegmentation/salsanext_input/holder.jpg
   ```
  >   **Warning**
  >   In the directory of the test image, there **must** be the following extra 3D point cloud information:
    ```text
    salsanext_input               # folder with the image
    ├── holder.jpg                # test image
    ├── scan_x.txt                # x-component of the points
    ├── scan_y.txt                # y-component of the points
    ├── scan_z.txt                # z-component of the points
    └── scan_remission.txt        # intensity of the laser beam reflected from the objects in the environment
    ```

- To get performance metrics (FPS, E2E, DPU_MEAN), use:
  ```bash
  # Format: bash scripts/performance.sh <MODEL_PATH> [<image paths list>]
  # where:
  # <MODEL_PATH> - the absolute path to the .xmodel
  # [<image paths list>] - space-separated list of image absolute paths
  # Alternatively, you can pass --dataset option with the folder where images are stored.
  # Example:

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/salsanext_pt/salsanext_pt.xmodel /workspace/Vitis-AI-Library/samples/3DSegmentation/salsanext_input/holder.jpg
  ```
  >   **Warning**
  >   In the directory of the test image, there **must** be the following extra 3D point cloud information:
    ```text
    salsanext_input               # folder with the image
    ├── holder.jpg                # test image
    ├── scan_x.txt                # x-component of the points
    ├── scan_y.txt                # y-component of the points
    ├── scan_z.txt                # z-component of the points
    └── scan_remission.txt        # intensity of the laser beam reflected from the objects in the environment
    ```


# Links

- Salsanext model review (PapersWithCode): https://paperswithcode.com/paper/salsanext-fast-semantic-segmentation-of-lidar/review/
- SemanticKitti dataset: http://www.semantic-kitti.org/
- 3D Semantic segmentation benchmark on the SemanticKitti (PapersWithCode): https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=salsanext-fast-semantic-segmentation-of-lidar

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
