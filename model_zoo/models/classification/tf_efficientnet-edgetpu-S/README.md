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

EfficientNet-EdgeTPU-S model for image classification customized for deployment on Google TPU. These networks are closely related to EfficientNets
that achieves state-of-the-art performance by efficiently scaling network dimensions, resulting in a balance between model size and accuracy. 

## Paper
 Gupta, Suyog, and Mingxing Tan. "EfficientNet-EdgeTPU: Creating accelerator-optimized neural networks with AutoML." Google AI Blog 2.1 (2019).
 Link - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

# Model Architecture
EfficientNet is a family of convolutional neural network (CNN) models designed to achieve state-of-the-art performance on various computer vision tasks while maintaining computational efficiency.
The EfficientNet models use a compound scaling method that uniformly scales the network's depth, width, and resolution.
This approach allows the models to achieve excellent accuracy by finding an optimal balance between model size and performance.
The AutoML MNAS framework was utilized to develop EfficientNet-EdgeTPU by incorporating specially optimized building blocks 
into the neural network search space. These building blocks were designed to maximize efficiency when executing on the EdgeTPU 
neural network accelerator architecture.
# Dataset

Dataset for testing: ImageNet. The ImageNet dataset is a large-scale visual database widely used in the image classification and object recognition tasks. <br>
The dataset categories cover a wide range of objects, animals, scenes, and everyday items. Each image in the dataset is annotated with a single label indicating the object or concept it represents.
Link to download the  dataset: https://www.image-net.org/

# Features

The notable features of the model:

1. The model introduces a new compound scaling method that uniformly scales the width, depth, and resolution of the network using a single scaling parameter.
2. The architecture consists of stacked layers, including convolutional layers, pooling layers, and fully connected layers.
3. The model is customized from the original EfficientNet for deployment on Google TPU.

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
tf_efficientnet-edgetpu-S  # model name  
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
  
  bash scripts/quality.sh $MODEL_FOLDER/artifacts/inference/results/ /workspace/Vitis-AI-Library/samples/classification/images/ --batch
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
  
  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/efficientNet-edgetpu-S_tf/efficientNet-edgetpu-S_tf.xmodel --dataset /workspace/Vitis-AI-Library/samples/classification/images/
  ```


# Links

- EfficientNets: https://arxiv.org/abs/1905.11946
- ImageNet dataset: https://www.image-net.org/
- Classification: New Annotations, Experiments, and Results: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7830427/pdf/sensors-21-00596.pdf
- EfficientNet (PapersWithCode): https://paperswithcode.com/method/efficientnet

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
