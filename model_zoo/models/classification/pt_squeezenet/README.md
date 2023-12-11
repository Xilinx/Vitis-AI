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

SqueezeNet is a compact deep neural network architecture designed for efficient image classification tasks. 
It aims to achieve a balance between model size and performance by drastically reducing the number of parameters while maintaining competitive accuracy. 

## Paper

Iandola, Forrest N., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size." 
arXiv preprint arXiv:1602.07360 (2016).  Link: 


# Model Architecture

SqueezeNet's architecture focuses on reducing the number of parameters by employing a combination of strategies such as 1x1 convolutional filters (also known as pointwise convolutions) and aggressive downsampling. The primary innovation lies in the "fire" modules, which consist of a combination of 1x1 and 3x3 convolutions that expand and then squeeze the data, hence the name "SqueezeNet." The 1x1 convolutions help to mix and reduce the number of channels, while the 3x3 convolutions capture spatial features.

# Dataset

Dataset for testing: ImageNet. The ImageNet dataset is a large-scale visual database widely used in the image classification and object recognition tasks. <br>
The dataset categories cover a wide range of objects, animals, scenes, and everyday items. Each image in the dataset is annotated with a single label indicating the object or concept it represents.
Link to download the  dataset: https://www.image-net.org/

# Features

The notable features of the Squeezenet model:

1. **Model Size:** SqueezeNet achieves a compact model size by heavily relying on 1x1 convolutions, which significantly reduces the number of parameters compared to traditional architectures.
2. **High Performance**: Despite its small size, SqueezeNet maintains competitive accuracy on image classification tasks like ImageNet, thanks to its efficient design and well-balanced use of convolutional filters.
3. **Real-time Inference**: SqueezeNet is well-suited for real-time applications due to its low computational demands, making it applicable in scenarios where low-latency predictions are crucial.
4. **Transfer Learning**: While originally designed for image classification, SqueezeNet's compact architecture makes it suitable as a starting point for transfer learning on other tasks, particularly when computational resources are limited.
5. **Embedded Systems**: SqueezeNet's efficiency makes it an attractive option for deployment on resource-constrained devices like smartphones, IoT devices, and edge computing devices.

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
pt_squeezenet                 # model name  
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/squeezenet_pt/squeezenet_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/classification/images/
  ```

# Links

- ImageNet dataset: https://www.image-net.org/
- SqueezeNet original paper: https://arxiv.org/abs/1602.07360
- SqueezeNet explained: https://paperswithcode.com/method/squeezenet
- Deep residual learning for image recognition: https://arxiv.org/abs/1512.03385


# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
