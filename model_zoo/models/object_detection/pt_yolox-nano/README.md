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

YOLOX-nano is a lightweight variant of the YOLOX object detection model. It is designed for real-time object detection tasks, 
particularly on resource-constrained devices such as embedded systems and mobile devices. Despite its compact size, 
YOLOX-nano maintains competitive accuracy and achieves remarkable inference speed.

## Paper

 Ge, Zheng, et al. "Yolox: Exceeding yolo series in 2021." <br>
 arXiv preprint arXiv:2107.08430 (2021). Link: https://arxiv.org/abs/2107.08430

# Model Architecture
The architecture of YOLOX-nano is based on the You Only Look Once (YOLO) family of object detection models. 
It follows a one-stage detection pipeline, where a single convolutional neural network (CNN) simultaneously predicts object 
bounding boxes and class probabilities. YOLOX-nano incorporates several design strategies, including the Darknet backbone, 
a Spatial Attention Module (SAM), and a Detect Head, to enhance feature representation, spatial attention, and detection performance.

# Dataset

Dataset for testing: COCO. The COCO dataset is a widely used benchmark dataset in the field of object detection. 
It focuses on high-quality pixel-level annotations for various urban objects, including cars, pedestrians, roads, buildings, traffic signs, and more. 

Link to download the dataset: https://cocodataset.org/

# Features

The notable features of the YOLOX-nano model:

1. **Backbone**: YOLOX-nano adopts the Darknet backbone, which consists of a series of convolutional layers followed by downsampling operations. 
2. **Spatial Attention Module (SAM)**: YOLOX-nano incorporates a Spatial Attention Module to enhance the model's capability to attend to important spatial regions in the feature maps. 
3. **Detect Head**: YOLOX-nano utilizes a Detect Head module responsible for predicting object bounding boxes and class probabilities.
4. **Scaled-YOLOX**: YOLOX-nano follows the Scaled-YOLOX paradigm, which involves progressively decreasing the input resolution during training and inference. 

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
pt_yolox-nano                 # model name  
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
  
  bash scripts/quality.sh $MODEL_FOLDER/artifacts/inference/results/ /workspace/Vitis-AI-Library/samples/yolovx/images/ --dataset
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

  bash scripts/performance.sh $MODEL_FOLDER/artifacts/models/yolovx_nano_pt/yolovx_nano_pt.xmodel --dataset /workspace/Vitis-AI-Library/samples/yolovx/images/
  ```


# Links

- YOLOX: Exceeding YOLO Series in 2021: https://arxiv.org/abs/2107.08430
- COCO dataset: https://cocodataset.org/
- Object detection benchmark on the COCO (PapersWithCode): https://paperswithcode.com/sota/object-detection-on-coco

# Vitis AI Model Zoo Homepage

Check the official Vitis AI Model Zoo [homepage](https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo).
