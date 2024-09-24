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

The SSD-ResNet34 detector is a model designed for object detection tasks based on the COCO dataset.
It combines the ResNet34 architecture with the Single Shot MultiBox Detector (SSD) framework
to achieve accurate and efficient object detection.

## Paper

Liu, Wei et al. "Speed/accuracy trade-offs for modern convolutional object detectors."
arXiv preprint arXiv:1611.10012 – 2016. Link: http://arxiv.org/abs/1611.10012/

# Model Architecture

The SSD-ResNet34 model architecture is a fusion of the ResNet34 backbone network and the SSD framework.
The ResNet34 serves as the feature extractor, consisting of multiple convolutional layers with residual connections.
The SSD framework adds additional convolutional layers on top of the ResNet34 to generate a set of default bounding boxes at different scales and aspect ratios. These bounding boxes are then refined to accurately localize objects.

# Dataset

Dataset for testing: COCO. The COCO dataset is a widely used benchmark dataset in the field of object detection.
It focuses on high-quality pixel-level annotations for various urban objects, including cars, pedestrians, roads, buildings, traffic signs, and more.

Link to download the dataset: https://cocodataset.org/

# Features

The notable features of the SSD-ResNet34 detector:

1. Utilizes the ResNet34 architecture as the backbone for robust feature extraction.
2. Implements the SSD framework for efficient object detection by generating default bounding boxes and predicting class probabilities for each box.
3. Achieves high accuracy in object detection tasks while maintaining real-time processing speeds.
4. Handles objects at different scales and aspect ratios effectively through multi-scale feature maps and anchor boxes.

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
tf_mlperf_resnet34            # model name  
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
  
  bash scripts/quality.sh $MODEL_FOLDER/artifacts/inference/results/ /workspace/Vitis-AI-Library/samples/yolov4/images/ --dataset
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/yolov4_leaky_512_tf/yolov4_leaky_512_tf.xmodel --dataset /workspace/Vitis-AI-Library/samples/yolov4/images/
  ```

# Links

- COCO dataset: https://cocodataset.org/
- Object detection benchmark on the COCO (PapersWithCode): https://paperswithcode.com/sota/object-detection-on-coco

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
s
