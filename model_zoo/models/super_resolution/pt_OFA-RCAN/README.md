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

The OFA-RCAN (Omnidirectional Feature Aggregation and Recursive Channel Attention Networks) is a deep learning model 
designed for single-image super-resolution tasks. It leverages omnidirectional feature aggregation and recursive channel 
attention mechanisms to effectively enhance the resolution and details of low-resolution images.

## Paper

Cai, Han, et al. "Once-for-all: Train one network and specialize it for efficient deployment." 
arXiv preprint arXiv:1908.09791 (2019). Link: https://arxiv.org/abs/1908.09791

# Model Architecture

The architecture of OFA-RCAN consists of two main components: the Omnidirectional Feature Aggregation module and 
the Recursive Channel Attention module. The Omnidirectional Feature Aggregation module captures multi-scale features 
by integrating multiple receptive fields, enabling the model to extract rich spatial information. 
The Recursive Channel Attention module incorporates recursive connections and channel attention mechanisms to refine 
feature representations and selectively enhance important features for high-resolution reconstruction.

# Dataset

Dataset for testing: DIV2K. The DIV2K dataset is a benchmark dataset consisting of 2,000 diverse high-quality images with a resolution of 2K. 
It is specifically designed for evaluating single-image super-resolution models

Link to download the  dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/

# Features

The notable features of the OFA-RCAN model:

1. **Omnidirectional Feature Aggregation**: The model incorporates multiple receptive fields to capture features at different scales, allowing it to effectively extract spatial information.
2. **Recursive Channel Attention**: By using recursive connections and channel attention mechanisms, the model iteratively refines feature representations and selectively enhances important features.
3. **Efficient and Scalable**: Despite its high performance, the model is designed to be computationally efficient and scalable, making it practical for real-time and large-scale super-resolution applications.
4. **Generalization**: The model exhibits good generalization capabilities, allowing it to perform well on a wide range of images and diverse super-resolution scenarios.

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
pt_OFA-RCAN                   # model name 
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/ofa_rcan_latency_pt/ofa_rcan_latency_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/rcan/images/
  ```


# Links

- Xilinx article: review of OFA technique: https://www.xilinx.com/developer/articles/advantages-of-using-ofa.html
- PapersWithCode - OFA technique: https://paperswithcode.com/method/ofa
- DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- Plug-and-Play Image Restoration with Deep Denoiser Prior: https://arxiv.org/pdf/2008.13751.pdf
- Learning Deep CNN Denoiser Prior for Image Restoration: https://arxiv.org/pdf/1704.03264.pdf

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
