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

ResNet50 is a deep convolutional neural network model that has achieved significant advancements in image classification tasks. 
It addresses the challenge of training deep networks by introducing residual connections, enabling the successful training of models with 50 layers.

## Paper

He, Kaiming, et al. "Deep residual learning for image recognition. arXiv 2015." 
arXiv preprint arXiv:1512.03385 14 (2015).  Link: https://arxiv.org/abs/1512.03385

# Model Architecture

The architecture of ResNet50 consists of 50 layers, including convolutional layers, pooling layers, fully connected layers, 
and shortcut connections. It follows a "building block" structure where each block contains multiple convolutional layers 
with batch normalization and ReLU activation, along with a skip connection that bypasses the block. These skip connections 
help propagate the gradients and enable training of deeper networks.

# Dataset

Dataset for testing: ImageNet. The ImageNet dataset is a large-scale visual database widely used in the image classification and object recognition tasks. <br>
The dataset categories cover a wide range of objects, animals, scenes, and everyday items. Each image in the dataset is annotated with a single label indicating the object or concept it represents.
Link to download the  dataset: https://www.image-net.org/

# Features

The notable features of the ResNet50 model:

1. **Residual Connections**: The introduction of residual connections in ResNet50 allows the network to learn residual mappings,
which helps in training deeper models more effectively.
2. **Skip Connections**: The skip connections in ResNet50 allow the network to learn residual mappings. 
3. **Pre-Activation Residual Units**: The building blocks in ResNet50 follow the pre-activation residual unit design,
which places batch normalization and ReLU activation before each convolutional layer. This helps in reducing "vanishing/exploding gradients" problem.
4. **Deep Architecture**: With its 50 layers, ResNet50 has a deep architecture that enables it to capture intricate features and patterns in images. 
5. **Pre-trained Model**: ResNet50 is often used as a pre-trained model, meaning it has been trained on a large dataset (e.g., ImageNet). This pre-training enables transfer learning, where the model can be fine-tuned on smaller datasets.

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
pt_resnet50                   # model name  
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
  # --dataset        - Evaluate ImageNet dataset
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/resnet50_pruned_0_6_pt/resnet50_pruned_0_6_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/classification/images/
  ```

# Links

- ImageNet dataset: https://www.image-net.org/
- Pytorch documentation, ResNet50: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
- ResNet50 overview: https://iq.opengenus.org/resnet50-architecture/
- Deep residual learning for image recognition: https://arxiv.org/abs/1512.03385


# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
