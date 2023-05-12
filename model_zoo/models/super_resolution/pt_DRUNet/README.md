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
    - [Script Parameters](#script-parameters)
    - [Data Preprocessing](#data-preprocessing)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage)
        - [Result](#result)
    - [Inference Process](#inference-process)
        - [Usage](#usage-1)
        - [Result](#result-1)
- [Performance](#performance)
    - [Evaluation Performance](#evaluation-performance)
    - [Inference Performance](#inference-performance)
- [Links](#links)
- [Vitis AI Model Zoo Homepage](#vitis-ai-model-zoo-homepage)

# Model Description

## Description

A Dilated-Residual U-Net Deep Learning Network for Image Denoising. It combines the strengths of the U-Net architecture 
and dilated convolutions, along with residual connections, to effectively remove noise from images 
while preserving important image details.

## Paper

Krishna Devalla, Sripad, et al. "DRUNET: A Dilated-Residual U-Net Deep Learning Network to Digitally Stain Optic
Nerve Head Tissues in Optical Coherence Tomography Images." arXiv e-prints (2018): 
[arXiv-1803](https://arxiv.org/abs/1803.00232).

# Model Architecture
A Dilated-Residual U-Net architecture leverages a combination of U-Net structure, dilated convolutions, 
residual connections, and skip connections. This combination enables the model to capture both local and global 
contextual information, propagate gradients effectively, and recover fine details, ultimately leading 
to high-quality image denoising results.

# Dataset
Dataset for testing: CBSD68. The CBSD68  dataset is a widely used benchmark dataset for image denoising. CBSD stands for "Color and Binary Shape Database".
It consists of 68 grayscale images with various scenes and objects. 

Link to download the  dataset: https://github.com/clausmichele/CBSD68-dataset

# Features
The notable features of the Dilated-Residual U-Net:
1. **U-Net Architecture**: Enables contextual information capture and spatial detail recovery.
2. **Dilated Convolutions**: Captures both local and global contextual information.
3. **Residual Connections**: Efficiently propagates gradients and preserves important features.
4. **Skip Connections**: Merges high-resolution and low-resolution features to recover fine details.
5. **Multi-Scale Information Fusion**: Considers information at multiple scales for effective denoising.
# Environment Requirements
- Docker: Install Docker on your host system.
- Hardware Platform: prepare Xilinx FPGA/ACAP and the appropriate accelerator should be installed correctly, e.g.
[Xilinx VCK5000 Versal](https://xilinx.github.io/Vitis-AI/docs/board_setup/board_setup_vck5000.html).
- Model Files: Include the Vitis-AI model files inside the Docker image. You can either download them from the Vitis-AI model zoo or add your own trained model files to the image. 
Make sure the model files are accessible within the Docker container.
- Input Data: Prepare the input data you want to use for inference and ensure it is accessible within the Docker container.

# Quick Start

Follow the [Quick Start guide](../../../README.md#quick-start) in the main Model Zoo README:
install the Vitis-AI, run the docker container, download test data and then run the inference.

# Script Description

## Structure

```text
.
├── artifacts                   # models binaries and inference result files
└── scripts                     
    ├── config.env              # environment variables setting       
    └── inference.sh            # bash script for running model inference
```
## Script Parameters
Inference:
```text
model_path                   # The path to the model binary .xmodel
filepaths                    # The list of files for inference
results_folder               # The directory where model's inference results are stored.
```
## Data Preprocessing

## Evaluation Process

### Usage

### Result

## Inference Process

### Usage

### Result

# Performance

## Evaluation Performance

## Inference Performance

# Links

- The Berkeley Segmentation Dataset and Benchmark (original): https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- CBSD68-dataset for image denoising benchmarks: https://github.com/clausmichele/CBSD68-dataset
- Plug-and-Play Image Restoration with Deep Denoiser Prior: https://arxiv.org/pdf/2008.13751.pdf
- Learning Deep CNN Denoiser Prior for Image Restoration: https://arxiv.org/pdf/1704.03264.pdf

# Vitis AI Model Zoo Homepage

Please check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
