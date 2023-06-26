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
The tf2 ERFNet is a deep learning model specifically designed for efficient and accurate semantic segmentation of images. 
It is based on the original ERFNet architecture and has been implemented using TensorFlow 2.
## Paper
E. Romera, J. M. Álvarez, L. M. Bergasa and R. Arroyo, "ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation," 
in IEEE Transactions on Intelligent Transportation Systems, vol. 19, no. 1, pp. 263-272, Jan. 2018, doi: 10.1109/TITS.2017.2750080.
Link: https://ieeexplore.ieee.org/abstract/document/8063438

# Model Architecture

The architecture of the tf2 ERFNet consists of an encoder-decoder structure with a lightweight design aimed at real-time processing.
The encoder module utilizes a series of convolutional layers with progressively increasing receptive fields to capture hierarchical features. 
The decoder module consists of upsampling and convolutional layers to generate pixel-wise predictions.

# Dataset

Dataset for testing: CityScapes. The Cityscapes dataset is a widely used benchmark dataset for semantic understanding of urban street scenes. 
It focuses on high-quality pixel-level annotations for various urban objects, including cars, pedestrians, roads, buildings, traffic signs, and more. 

Link to download the  dataset: https://www.cityscapes-dataset.com/

# Features

The notable features of the ERFNet:
1. **Efficient Resource Filter (ERF)**: The ERF module in the encoder helps capture multi-scale contextual information while maintaining computational efficiency. 
2. **Encoder-Decoder Structure**: The architecture follows an encoder-decoder structure, where the encoder captures high-level features and the decoder performs upsampling and pixel-wise predictions. This enables accurate semantic segmentation. 
3. **Dilated Convolutions**: Dilated convolutions are used in the ERFNet architecture to increase the receptive field of the network without sacrificing spatial resolution. This helps in capturing both local and global contextual information. 
4. **Skip Connections**: Skip connections are incorporated in the decoder module to combine low-level and high-level features. These connections aid in preserving spatial details and improve the overall segmentation performance. 
5. **Lightweight Design**: The model is designed to be computationally efficient, making it suitable for real-time applications on devices with limited computational resources. 

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
tf2_ERFNet                    # model name  
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/semantic_seg_citys_tf2/semantic_seg_citys_tf2.xmodel --dataset /workspace/Vitis-AI-Library/samples/segmentation/images/
  ```


# Links

- Cityscapes dataset: https://www.cityscapes-dataset.com/
- Panoptic Feature Pyramid Networks: https://arxiv.org/abs/1901.02446
- Going deeper with convolutions (Inception): https://arxiv.org/pdf/1409.4842
- Semantic segmentation benchmark on the Cityscapes (PapersWithCode): https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
