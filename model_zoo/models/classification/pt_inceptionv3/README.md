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

Inception V3 is a convolutional neural network (CNN) model developed by Google. It is designed for image classification
tasks and has achieved state-of-the-art performance on various benchmark datasets.

## Paper

Szegedy, Christian, et al. "Rethinking the Inception Architecture for Computer Vision." Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2818-2826. Link: https://arxiv.org/abs/1512.00567

# Model Architecture

The architecture of Inception V3 is based on the concept of "Inception modules," which are stacked together to form a deep neural network.
These modules consist of parallel convolutional layers with different filter sizes, allowing the model to capture information at multiple scales.
Inception V3 also incorporates techniques like factorized convolutions and dimensionality reduction to improve computational efficiency.

# Dataset

Dataset for testing: ImageNet. The ImageNet dataset is a large-scale visual database widely used in the image classification and object recognition tasks. <br>
The dataset categories cover a wide range of objects, animals, scenes, and everyday items. Each image in the dataset is annotated with a single label indicating the object or concept it represents.
Link to download the  dataset: https://www.image-net.org/

# Features

The notable features of the Inception V3 model:

1. **Inception modules**: The model employs a series of Inception modules that consist of 1x1, 3x3, and 5x5 convolutions,
   as well as pooling operations. These modules enable the model to learn hierarchical representations at different scales
   and capture both local and global information.
2. **Factorized convolutions**: Inception V3 uses factorized convolutions, which split the standard convolution
   into two smaller convolutions, reducing the number of parameters and computational cost while maintaining model performance.
3. **Auxiliary classifiers**: The model incorporates auxiliary classifiers at intermediate layers, aiding in training by providing additional gradients.
   This helps to avoid the vanishing gradient problem and improve gradient flow through the network.
4. **Pretrained weights**: Inception V3 is often used as a transfer learning model due to its availability of pretrained
   weights on large-scale image classification datasets like ImageNet. These pretrained weights can be fine-tuned on specific tasks, allowing for effective transfer of knowledge.

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
pt_inceptionv3                # model name  
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/inception_v3_pruned_0_5_pt/inception_v3_pruned_0_5_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/classification/images/
  ```

# Links

- Inception-V3 overview (PapersWithCode): https://paperswithcode.com/method/inception-v3
- ImageNet dataset: https://www.image-net.org/
- Going deeper with convolutions (Inception): https://arxiv.org/pdf/1409.4842

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
