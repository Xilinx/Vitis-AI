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

ResNet18 vehicle classification model. ResNet-18 is a popular variant of the Residual Neural Network (ResNet) architecture, 
which is widely used for various computer vision tasks, including vehicle classification.

## Paper
 Watkins, Rohan, Nick Pears, and Suresh Manandhar. "Vehicle classification using ResNets, localisation and spatially-weighted pooling." 
 arXiv preprint arXiv:1810.10329 (2018). Link - https://arxiv.org/abs/1810.10329

# Model Architecture
ResNet-18 is a deep convolutional neural network architecture designed for image classification tasks, including vehicle classification. 
It consists of a series of convolutional layers with 3x3 filters, followed by four sets of residual blocks. 
Each residual block contains convolutional layers and a skip connection that bypasses the layers, allowing the network to learn residual mappings. 
Average pooling is applied after each set of blocks, and the final output is obtained through a fully connected layer with softmax activation. 
ResNet-18's key features include residual connections for training deeper networks, its relatively shallow depth of 18 layers, 
pretraining capabilities for transfer learning, and its high accuracy in image classification benchmarks.
# Dataset

Dataset for testing: CompCars. The CompCars dataset contains comprehensive annotations and images of vehicles captured 
from different viewpoints and under varying conditions. <br>
Link to download the  dataset: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

# Features

The notable features of the model:

1. **Residual Neural Networks (ResNets)**.
2. **Localization**.
3. **Spatially-Weighted Pooling**.

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
pt_vehicle-make-classification  # model name  
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
  # --batch          - Evaluate a dataset (default: individual images)
  # --dataset        - Evaluate CompCars dataset
  # The metric values will be stored in the artifacts/inference/quality/metrics.txt file
  # Example:
  
  bash scripts/quality.sh $MODEL_FOLDER/artifacts/inference/results/ /workspace/Vitis-AI-Library/samples/vehicleclassification/vehicle_images/ --batch
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
  
  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/vehicle_make_resnet18_pt/vehicle_make_resnet18_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/vehicleclassification/vehicle_images/
  ```


# Links

- CompCars dataset: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
- EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML.
Google Research blog post: https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
- Going deeper with convolutions (Inception): https://arxiv.org/pdf/1409.4842
- Benchmark on the CompCars (PapersWithCode): https://paperswithcode.com/dataset/compcars

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
