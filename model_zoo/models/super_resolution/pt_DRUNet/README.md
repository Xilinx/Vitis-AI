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

A Dilated-Residual U-Net Deep Learning Network for Image Denoising.

## Paper

Krishna Devalla, Sripad, et al. "DRUNET: A Dilated-Residual U-Net Deep Learning Network to Digitally Stain Optic
Nerve Head Tissues in Optical Coherence Tomography Images." arXiv e-prints (2018): 
[arXiv-1803](https://arxiv.org/abs/1803.00232).

# Model Architecture

# Dataset

The dataset for testing: CBSD68.

Link to download the original dataset: https://github.com/clausmichele/CBSD68-dataset

# Features

# Environment Requirements

# Quick Start

1. Follow the Quick Start Prerequisites chapter in the model_zoo README:
install the Vitis-AI, run the docker container and download test data.
2. Go to the model's folder:
```bash
cd /workspace/model_zoo/models/super_resolution/pt_DRUNet
```
3. Make a folder to save artifacts:
```bash
bash scripts/make_artifacts_folder.sh
```
4. Download the model files for specific device and device configuration:
```bash
cd /workspace/model_zoo
python downloader.py

# A command line interface will be provided for downloading model files.

# In the first input you need to specify the base framework and the model name.
# Example of the first input:
# input: pt drunet

# Then select the desired device configuration.
# Example of the second input:
# input num: 7

# As a result you will download the .tar.gz archive with model files.
```
5. Move and unzip the downloaded model:
```bash
mv drunet_pt-vck5000-DPUCVDX8H-8pe-r3.0.0.tar.gz models/super_resolution/pt_DRUNet/artifacts/models/
cd models/super_resolution/pt_DRUNet/
tar -xzvf artifacts/models/drunet_pt-vck5000-DPUCVDX8H-8pe-r3.0.0.tar.gz -C artifacts/models/
```
6. Set environment variables for a specific device and device configuration inside the docker container:
```bash
# source /vitis_ai_home/board_setup/<DEVICE_NAME>/setup.sh <DEVICE_CONFIGURATION>
# where:
# <DEVICE_NAME> - the name of current device
# <DEVICE_CONFIGURATION> - selected device configuration

# Example:
source /vitis_ai_home/board_setup/vck5000/setup.sh DPUCVDX8H_8pe_normal
```
7. Run the inference on files:
```bash
# bash inference.sh <MODEL_PATH> [<image paths list>]
# where:
# <MODEL_PATH> - the absolute path to the .xmodel
# [<image paths list>] - space-separated list of image absolute paths

# Example
bash scripts/inference.sh \
    /workspace/model_zoo/models/super_resolution/pt_DRUNet/artifacts/models/drunet_pt/drunet_pt.xmodel \
    /workspace/Vitis-AI-Library/samples/rcan/images/1.png /workspace/Vitis-AI-Library/samples/rcan/images/2.png \
    /workspace/Vitis-AI-Library/samples/rcan/images/3.png
```
8. Results of the inference you will find in the folder: `artifacts/inference`

# Script Description

## Structure

## Script Parameters

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
