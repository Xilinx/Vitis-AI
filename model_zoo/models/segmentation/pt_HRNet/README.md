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
- [Quality](#quality)
- [Performance](#performance)
- [Links](#links)
- [Vitis AI Model Zoo Homepage](#vitis-ai-model-zoo-homepage)

# Model Description

## Description
HRNet is a deep learning model designed for visual recognition tasks such as object detection and segmentation. 
## Paper

 Wang, Jingdong, et al. "Deep high-resolution representation learning for visual recognition." 
 IEEE transactions on pattern analysis and machine intelligence 43.10 (2020): 3349-3364. 
 Link: https://arxiv.org/abs/1908.07919v2

# Model Architecture
HRNet's architecture consists of parallel multi-resolution streams that process the input image at different levels of spatial resolution. 
Unlike traditional convolutional neural networks that downsample the resolution early in the network, 
HRNet maintains high-resolution feature maps throughout its processing stages. 
It employs a high-to-low resolution fusion strategy, where features from lower resolution streams are upsampled 
and fused with features from higher resolution streams to preserve fine-grained details.
# Dataset

Dataset for testing: CityScapes. The Cityscapes dataset is a widely used benchmark dataset for semantic understanding of urban street scenes. 
It focuses on high-quality pixel-level annotations for various urban objects, including cars, pedestrians, roads, buildings, traffic signs, and more. 

Link to download the  dataset: https://www.cityscapes-dataset.com/

# Features

The notable features of the HRNet model:

1. **Multi-resolution processing**: HRNet processes the input image at multiple resolutions simultaneously, allowing it to capture both global context and fine-grained details.
2. **High-resolution representation learning**: By maintaining high-resolution feature maps, HRNet preserves fine details that are crucial for accurate visual recognition.
3. **High-to-low resolution fusion**: HRNet employs a fusion strategy to combine features from different resolution streams, enabling the integration of both local and global information.
4. **Scale-aware training**: HRNet incorporates scale-aware training techniques to effectively handle objects of different scales, enhancing its ability to detect and segment objects of varying sizes.

# Environment Requirements

Before running the model inference, make sure that the latest version of
[Vitis-AI](https://xilinx.github.io/Vitis-AI/3.5/html/docs/install/install.html) is installed and the host computer fully supports
Xilinx FPGA/ACAP and the appropriate accelerator is installed correctly, e.g. Alveo V70.

# Quick Start

Follow the [Quick Start guide](../../../README.md#quick-start) in the main Model Zoo README:

1. Install the Vitis-AI
2. Run the docker container
3. Download test data
4. Run the inference of the model

# Script Description

## Structure

```text
pt_HRNet                      # model name  
├── artifacts                 # artifacts - will be created during the inference process
│ ├── inference               # folder with results values of inference and evaluation
│ │ ├── performance           # model productivity measurements
│ │ ├── quality               # model quality measurements
│ │ ├── results               # model inference results files
│ │ └── vaitrace              # vaitrace profiling performance reports
│ └── models                  # folder with model meta and .xmodel executable files
├── scripts                   # scripts for model processing 
│ ├── inference.sh            # model inference
│ ├── performance.sh          # model performance report
│ ├── quality.sh              # model quality report
│ └── setup_venv.sh           # virtual environment creation
├── src                       # python supporting scripts
│ └── quality.py              # quality metric calculation
├── config.env                # model configuration - env variables
├── README.md
└── requirements.txt          # requirements for the virtual environment
```

## Inference Process

- Native inference - follow the [Quick Start guide](../../../README.md#quick-start) in the main Model Zoo
- AMD Server

# Quality

Use the following script:

```bash
  # Format: bash scripts/quality.sh <inference_result> <ground_truth> [--batch] [--dataset]
  # where:
  #  inference_result  - Path to the inference result image or folder.
  #  ground_truth      - Path to the ground truth image or folder
  # --batch            - Evaluate a dataset (default: individual images)
  # --dataset          - Evaluate Cityscapes dataset
  # The metric values will be stored in the artifacts/inference/quality/metrics.txt file
  # Example:
  
  bash scripts/quality.sh $MODEL_FOLDER/artifacts/inference/results/ /workspace/Vitis-AI-Library/samples/segmentation/images/ --dataset
```

# Performance

- You can profile the model using [vaitrace](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Starting-a-Simple-Trace-with-vaitrace) perfomance report,
  the script and format described in the [Quick Start guide](../../../README.md#vaitrace) in the main Model Zoo.
- To get performance metrics (FPS, E2E, DPU_MEAN), use:
  ```bash
  # Format: bash scripts/performance.sh <MODEL_PATH> [<image paths list>]
  # where:
  # <MODEL_PATH> - the absolute path to the .xmodel
  # [<image paths list>] - space-separated list of image absolute paths
  # Alternatively, you can pass --dataset option with the folder where images are stored.
  # Example:

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/HRNet_pt/HRNet_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/segmentation/images/
  ```


# Links

- HRNet Architecture (PapersWithCode): https://paperswithcode.com/method/hrnet
- Cityscapes dataset: https://www.cityscapes-dataset.com/
- Panoptic Feature Pyramid Networks: https://arxiv.org/abs/1901.02446
- Semantic segmentation benchmark on the Cityscapes (PapersWithCode): https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
