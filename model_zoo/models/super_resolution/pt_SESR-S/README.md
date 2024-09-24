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

SESR-S (Single Image Super Resolution with Recursive Squeeze and Excitation Networks) is an advanced model that pushes 
the boundaries of single image super-resolution by effectively exploiting recursive architecture and squeeze-and-excitation 
modules to generate high-quality, high-resolution images from their low-resolution counterparts.


## Paper

Cheng, Xi, et al. "SESR: Single image super resolution with recursive squeeze and excitation networks." 
2018 24th International conference on pattern recognition (ICPR). IEEE, 2018. 
Link: https://ieeexplore.ieee.org/abstract/document/8546130

# Model Architecture

The model consists of multiple stages, each responsible for progressively refining the image resolution. 
At the core of SESR is the recursive architecture, where the output of each stage is fed back into the network as an input 
for the next stage. This recursive process allows the model to iteratively refine the details and generate high-resolution images.

One of the key components of SESR is the squeeze-and-excitation module. 
This module focuses on capturing channel-wise dependencies within the network by adaptively recalibrating feature maps. 
It consists of two main operations: squeezing and exciting. The squeezing operation aggregates global information from the 
feature maps by applying global average pooling. The exciting operation utilizes learned parameters to generate channel-wise 
weights that are applied to the feature maps. This mechanism enables the model to emphasize important features and suppress less relevant ones, 
enhancing the overall image quality.

# Dataset

Dataset for testing: DIV2K. The DIV2K dataset is a popular benchmark dataset for image super-resolution.
It consists of 800 high-quality, high-resolution images divided into training and validation sets. 
These images cover a wide range of scenes and contain different types of content, making it suitable for evaluating super-resolution algorithms.

Link to download the  dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/

# Features

The notable features of the SESR-S model:

1. **Recursive Architecture**: The model utilizes a recursive approach where the output of each stage is fed back into the network as input for the next stage, allowing for iterative refinement of image resolution.
2. **Squeeze-and-Excitation Modules**: SESR incorporates squeeze-and-excitation modules to capture channel-wise dependencies and recalibrate feature maps, emphasizing important features and suppressing less relevant ones.
3. **Deep Convolutional Neural Networks**: The model leverages the power of deep CNNs to learn the mapping between low-resolution and high-resolution image spaces, enabling accurate and detailed super-resolution results.
4. **Local and Global Dependency Capture**: SESR combines recursive architecture and squeeze-and-excitation modules to capture both local and global dependencies within the image, enhancing overall image quality.

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
pt_SESR-S                     # model name 
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

To evaluate the model inference results, you may compute [PNSR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) metric.
Use the following script:

```bash
  # Format: bash scripts/quality.sh <DATASET_FOLDER> <INFERENCE_FOLDER>
  # where:
  # <DATASET_FOLDER> - The path of folder where original dataset is stored.
  # <INFERENCE_FOLDER> - The path of folder where results of model inference is stored.
  # The metric values will be stored in the artifacts/inference/quality/psnr.txt file
  # Example:
  
  bash scripts/quality.sh /workspace/Vitis-AI-Library/samples/rcan/images/ $MODEL_FOLDER/artifacts/inference/results/
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/SESR_S_pt/SESR_S_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/rcan/images/
  ```


# Links

- The DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- Plug-and-Play Image Restoration with Deep Denoiser Prior: https://arxiv.org/pdf/2008.13751.pdf
- Learning Deep CNN Denoiser Prior for Image Restoration: https://arxiv.org/pdf/1704.03264.pdf


# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
