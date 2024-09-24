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
pt_DRUNet                     # model name 
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

## Comparison

- Original paper results of mean PNSR metric: <br>
  <table style="undefined;table-layout: fixed; width: 472px">
    <colgroup>
    <col style="width: 59.444444px">
    <col style="width: 46.444444px">
    <col style="width: 77.444444px">
    <col style="width: 49.444444px">
    <col style="width: 62.444444px">
    <col style="width: 55.444444px">
    <col style="width: 60.444444px">
    <col style="width: 60.444444px">
    </colgroup>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Noise<br>level</th>
        <th>DRUNet</th>
        <th><a href="https://github.com/Ding-Liu/NLRN" target="_blank" rel="noopener noreferrer">NLRN</a></th>
        <th><a href="https://github.com/hsijiaxidian/FOCNet" target="_blank" rel="noopener noreferrer">FOCNet</a></th>
        <th><a href="https://github.com/cszn/IRCNN" target="_blank" rel="noopener noreferrer">IRCNN</a></th>
        <th><a href="https://github.com/cszn/FFDNet" target="_blank" rel="noopener noreferrer">FFDNet</a></th>
        <th><a href="https://github.com/cszn/DnCNN" target="_blank" rel="noopener noreferrer">DnCNN</a></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="3">BSD68</td>
        <td>15</td>
        <td>31.91</td>
        <td>31.88</td>
        <td>31.83</td>
        <td>31.63</td>
        <td>31.63</td>
        <td>31.73</td>
      </tr>
      <tr>
        <td>25</td>
        <td>29.48</td>
        <td>29.41</td>
        <td>29.38</td>
        <td>29.15</td>
        <td>29.19</td>
        <td>29.23</td>
      </tr>
      <tr>
        <td>50</td>
        <td>26.59</td>
        <td>26.47</td>
        <td>26.50</td>
        <td>26.19</td>
        <td>26.29</td>
        <td>26.23</td>
      </tr>
    </tbody>
    </table>

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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/drunet_pt/drunet_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/rcan/images/
  ```


# Links

- The Berkeley Segmentation Dataset and Benchmark (original): https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- CBSD68-dataset for image denoising benchmarks: https://github.com/clausmichele/CBSD68-dataset
- Plug-and-Play Image Restoration with Deep Denoiser Prior: https://arxiv.org/pdf/2008.13751.pdf
- Learning Deep CNN Denoiser Prior for Image Restoration: https://arxiv.org/pdf/1704.03264.pdf
- Benchmarks based on the CBSD68-dataset with SOTA solutions: https://paperswithcode.com/dataset/cbsd68

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
