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
ENet is a deep neural model for real-time semantic segmentation tasks, which involves classifying each pixel in an image 
into predefined categories. It offers a lightweight and efficient solution, enabling real-time inference on resource-constrained 
devices such as embedded systems and mobile devices.
## Paper

 Paszke, Adam, et al. "Enet: A deep neural network architecture for real-time semantic segmentation." 
 arXiv preprint arXiv:1606.02147 (2016). Link: https://arxiv.org/abs/1606.02147

# Model Architecture
The ENet architecture is built upon an encoder-decoder framework. The encoder consists of a series of convolutional blocks
that gradually reduce the spatial dimensions of the input image while capturing high-level features. 
The decoder, on the other hand, employs a set of upsampling and convolutional layers to produce a pixel-wise segmentation map 
with the same dimensions as the input image.
# Dataset

Dataset for testing: CityScapes. The Cityscapes dataset is a widely used benchmark dataset for semantic understanding of urban street scenes. 
It focuses on high-quality pixel-level annotations for various urban objects, including cars, pedestrians, roads, buildings, traffic signs, and more. 

Link to download the  dataset: https://www.cityscapes-dataset.com/

# Features

The notable features of the ENet model:

1. **Efficient architecture**: ENet is designed to be computationally efficient, making it suitable for real-time applications even on low-power devices.
2. **Lightweight model size**: ENet has a small model size, enabling easy deployment and reducing memory footprint.
3. **Skip connections**: The architecture incorporates skip connections between the encoder and decoder, allowing for the fusion of high-resolution features with the upsampled features, leading to more accurate segmentation results.
4. **Spatial and channel-wise attention**: ENet utilizes spatial and channel-wise attention mechanisms to enhance the discriminative power of the model, improving segmentation accuracy.
5. **Regularization techniques**: The model employs techniques such as dropout and batch normalization to mitigate overfitting and improve generalization capabilities.

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
pt_ENet                       # model name  
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
  
  bash scripts/quality.sh $MODEL_FOLDER/artifacts/inference/results/ /workspace/Vitis-AI-Library/samples/segmentation/images/ --dataset
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/ENet_ciityscapes_pt/ENet_ciityscapes_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/segmentation/images/
  ```


# Links

- ENet Architecture (PapersWithCode): https://paperswithcode.com/paper/enet-a-deep-neural-network-architecture-for
- Cityscapes dataset: https://www.cityscapes-dataset.com/
- Panoptic Feature Pyramid Networks: https://arxiv.org/abs/1901.02446
- Semantic segmentation benchmark on the Cityscapes (PapersWithCode): https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
