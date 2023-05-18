<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


# Vitis AI Model Zoo

As of the 3.0 release of Vitis AI, the Model Zoo documentation and performance benchmarks have migrated to Github.IO.  **[YOU MAY ACCESS THE MODEL ZOO DOCUMENTATION ONLINE](https://xilinx.github.io/Vitis-AI/docs/workflow-model-zoo)** or **[OFFLINE](../docs/docs/workflow-model-zoo.html)**.

## Quick Start

### Prerequisites

1. Before starting, make sure that the host computer fully supports Xilinx FPGA/ACAP and the appropriate accelerator
is installed correctly, e.g.
[Xilinx VCK5000 Versal](https://xilinx.github.io/Vitis-AI/docs/board_setup/board_setup_vck5000.html).
Or you can use an already configured server on [vmaccel.com](https://www.vmaccel.com/).
2. Install the latest [Vitis-AI](https://xilinx.github.io/Vitis-AI/docs/install/install.html).
3. Go to the Vitis-AI repo:
```bash
# cd <Vitis-AI install path>/Vitis-AI
# where:
# <Vitis-AI install path> - the path where Vitis-AI was installed

# Example:
cd ~/Vitis-AI
```
4. Start the Vitis AI Docker:
```bash
# ./docker_run.sh xilinx/vitis-ai-<Framework>-<Arch>:latest
# where:
# <Framework>, <Arch> - deep learning framework and target architecture,
# more info in the Vitis-AI installation instruction

# Example:
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
```
5. Download the test data:
```bash
bash model_zoo/scripts/download_test_data.sh
```
6. Make folders and subfolders to store artifacts:
```bash
cd model_zoo
bash scripts/make_artifacts_folders.sh
```

### Native Inference

1. Follow the [Prerequisites chapter](#prerequisites): install the Vitis-AI, run the docker container, 
download test data, make folders to store artifacts.
2. All the following commands must be run inside the Vitis-AI container.
3. Select one of the available models from `model_zoo/models` and set the environment variable with the absolute
path to the model:
```bash
# MODEL_FOLDER="$(pwd)"/model_zoo/models/<application>/<model name>
# where:
# "$(pwd)" - the absolute path to the current folder inside the container, e.g.: /workspace
# <application> - type or application of the model
# <model name> - the name of the specific model

# Example:
MODEL_FOLDER="$(pwd)"/model_zoo/models/super_resolution/pt_DRUNet
```
4. Download model files for the specific device and device configuration:
```bash
cd /workspace/model_zoo
python downloader.py

# A command line interface will be provided for downloading model files.

# In the first input you need to specify the base framework and the specific model name.
# Example of the first input:
# input: pt drunet

# Then select the desired device configuration.
# Example of the second input:
# input num: 7

# As a result you will download the .tar.gz archive with model files.
# Example: drunet_pt-vck5000-DPUCVDX8H-8pe-r3.0.0.tar.gz
```
5. Move and unzip the downloaded model:
```bash
# Example:
mv drunet_pt-vck5000-DPUCVDX8H-8pe-r3.0.0.tar.gz $MODEL_FOLDER/artifacts/models/
tar -xzvf $MODEL_FOLDER/artifacts/models/drunet_pt-vck5000-DPUCVDX8H-8pe-r3.0.0.tar.gz -C $MODEL_FOLDER/artifacts/models/
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
7. Go to the specific model's folder inside the `model_zoo`:
```bash
cd $MODEL_FOLDER
```
8. Run the inference on files:
```bash
# bash inference.sh <MODEL_PATH> [<image paths list>]
# where:
# <MODEL_PATH> - the absolute path to the .xmodel
# [<image paths list>] - space-separated list of image absolute paths
# Alternatively, you can pass --dataset option with the folder where images for the inference are stored.
# Example 1: 
bash scripts/inference.sh \
    $MODEL_FOLDER/artifacts/models/drunet_pt/drunet_pt.xmodel \
    /workspace/Vitis-AI-Library/samples/rcan/images/1.png /workspace/Vitis-AI-Library/samples/rcan/images/2.png \
    /workspace/Vitis-AI-Library/samples/rcan/images/3.png
# Example 2: 
bash scripts/inference.sh \
    $MODEL_FOLDER/artifacts/models/drunet_pt/drunet_pt.xmodel \
    --dataset /workspace/Vitis-AI-Library/samples/rcan/images
```
9. Results of the inference will be stored in the folder: `artifacts/inference`.
10. To get the [vaitrace](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Starting-a-Simple-Trace-with-vaitrace) performance report, you can use:
  ```bash
  # Format: bash scripts/vaitrace.sh <MODEL_PATH> <TEST_IMAGE_PATH>
  # where:
  # <MODEL_PATH> - The path to the model file .xmodel
  # <TEST_IMAGE_PATH> - The path to the image to be processed via vaitrace.
  # The report files will be stored in the $MODEL_FOLDER/artifacts/inference/vaitrace folder
  # Example: 

  bash scripts/vaitrace.sh $MODEL_FOLDER/artifacts/models/drunet_pt/drunet_pt.xmodel /workspace/Vitis-AI-Library/samples/rcan/images/2.png
  ```

## Contributing

We welcome community contributions. When contributing to this repository, first discuss the change you wish to make via:

-  [GitHub Issues](https://github.com/Xilinx/Vitis-AI/issues)
-  [Vitis AI Forums](https://forums.xilinx.com/t5/AI-and-Vitis-AI/bd-p/AI)
-  <a href="mailto:xilinx_ai_model_zoo@xilinx.com">Email</a>

You can also submit a pull request with details on how to improve the product. Prior to submitting your pull request, ensure that you can build the product and run all the demos with your patch. In case of a larger feature, provide a relevant demo.
