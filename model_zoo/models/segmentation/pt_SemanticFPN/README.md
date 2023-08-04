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

SemanticFPN (Semantic Feature Pyramid Network) is a widely used model in the task of semantic segmentation.

## Paper

 Shifeng Zhang, Longyin Wen, et al. "Semantic Feature Pyramid for Real-Time Semantic Segmentation.", 2018.

# Model Architecture
The Semantic FPN model builds upon the Feature Pyramid Network (FPN) by incorporating a top-down pathway and lateral connections.
It combines multi-scale spatial details with semantic information, allowing for better integration of context and 
improved accuracy in semantic segmentation tasks. By leveraging these architectural enhancements, 
SemanticFPN achieves more precise object segmentation by propagating semantic information across different scales in the feature pyramid.

# Dataset

Dataset for testing: CityScapes. The Cityscapes dataset is a widely used benchmark dataset for semantic understanding of urban street scenes. 
It focuses on high-quality pixel-level annotations for various urban objects, including cars, pedestrians, roads, buildings, traffic signs, and more. 

Link to download the  dataset: https://www.cityscapes-dataset.com/

# Features

The notable features of the SemanticFPN:

1. **Strong feature representation**
    By combining multi-scale features and semantic information, SemanticFPN creates a rich representation that captures 
    both local details and global context, leading to more robust object segmentation.
2. **Adaptability to different backbone networks**: 
    The model can be combined with various backbone networks, such as ResNet or VGGNet, providing flexibility 
    to choose the most suitable architecture for the specific dataset.
3. **Transfer learning**: 
    The pre-trained weights of the backbone network can be used as a starting point for training 
    the SemanticFPN model, enabling transfer learning and reducing the amount of training data required.
4. **End-to-end training**:The model can be trained end-to-end, allowing for joint optimization of the backbone network, 
    FPN, and the semantic segmentation head.
5. **Real-time performance**: The model is designed to achieve real-time semantic segmentation, making it suitable for 
    applications that require fast and efficient processing of images or videos.

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
pt_SemanticFPN                # model name  
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/SemanticFPN_Mobilenetv2_pt/SemanticFPN_Mobilenetv2_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/segmentation/images/
  ```


# Links

- Cityscapes dataset: https://www.cityscapes-dataset.com/
- Panoptic Feature Pyramid Networks: https://arxiv.org/abs/1901.02446
- Going deeper with convolutions (Inception): https://arxiv.org/pdf/1409.4842
- Semantic segmentation benchmark on the Cityscapes (PapersWithCode): https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
