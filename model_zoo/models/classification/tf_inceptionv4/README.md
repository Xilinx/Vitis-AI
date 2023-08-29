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

Inception-v4 is a deep convolutional neural network architecture designed for image classification. 
It is an evolution of the Inception family of models, known for their innovative use of multiple kernel sizes and parallel 
convolutions to capture features at different scales within a single layer. 

## Paper

Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." 
Proceedings of the AAAI conference on artificial intelligence. Vol. 31. No. 1. 2017.  
Link: https://ojs.aaai.org/index.php/aaai/article/view/11231

# Model Architecture

Inception-V4's architecture follows the principles of the Inception family, which emphasizes the use of multi-scale convolutions.
It employs a combination of 1x1, 3x3, and 5x5 convolutional filters, along with pooling layers, to capture various levels of detail. 
Additionally, the architecture incorporates auxiliary classifiers at intermediate stages, aiding in training 
by combating the vanishing gradient problem. 
Inception-V4 also benefits from factorized convolutions, where large convolutions are decomposed into smaller ones, reducing the computational burden. Architectural improvements such as residual connections and improved factorization contribute to enhanced feature extraction capabilities.

# Dataset

Dataset for testing: ImageNet. The ImageNet dataset is a large-scale visual database widely used in the image classification and object recognition tasks. <br>
The dataset categories cover a wide range of objects, animals, scenes, and everyday items. Each image in the dataset is annotated with a single label indicating the object or concept it represents.
Link to download the  dataset: https://www.image-net.org/

# Features

The notable features of the Inception-V4 model:
1. **Factorized Convolutions**: Large convolutions are factorized into smaller ones, reducing computational complexity.
2. **Residual Connections**: Integration of residual connections helps alleviate the vanishing gradient problem and enables training of deeper networks.
3. **Stem Network**: A specialized initial network module, the "stem," processes input images before feeding them into the main network, enabling efficient feature extraction.
4. **Reduction Blocks**: These blocks utilize 3x3 and 5x5 convolutions with pooling to reduce spatial dimensions and computational load while preserving important features.
5. **Auxiliary Classifiers**: Intermediate auxiliary classifiers aid in training by providing additional gradient flow paths during backpropagation.

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
tf_inceptionv4                # model name  
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/inception_v4_2016_09_09_tf/inception_v4_2016_09_09_tf.xmodel --dataset /workspace/Vitis-AI-Library/samples/classification/images/
  ```

# Links

- ImageNet dataset: https://www.image-net.org/
- Inception-V4 original paper: https://ojs.aaai.org/index.php/aaai/article/view/11231
- Inception-V4 explained: https://paperswithcode.com/method/inception-v4
- Deep residual learning for image recognition: https://arxiv.org/abs/1512.03385


# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
