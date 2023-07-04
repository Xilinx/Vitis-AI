<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


# Vitis AI Model Zoo

As of the 3.5 release of Vitis AI, the Model Zoo documentation and performance benchmarks have migrated to Github.IO.  [**YOU MAY ACCESS THE MODEL ZOO DOCUMENTATION ONLINE**](https://xilinx.github.io/Vitis-AI/docs/workflow-model-zoo) or [**OFFLINE**](../docs/docs/workflow-model-zoo.html).

## Model Zoo structure

```
Vitis-AI/model_zoo
├── images
├── model-list                      # list of all availible models with yaml configuration
├── models                          # model cards with code and all details 
│ ├── super_resolution              # task name
│ │ ├── pt_DRUNet                   # model name
│ │ │ ├── config.env                # model configuration - env variables
│ │ │ ├── artifacts                 # artifacts - will be created during the inference process
│ │ │ │ ├── inference               # folder with results values of inference and evaluation
│ │ │ │ │ ├── performance           # model productivity measurements
│ │ │ │ │ ├── quality               # model quality measurements
│ │ │ │ │ ├── results               # model inference results files
│ │ │ │ │ └── vaitrace              # vaitrace profiling performance reports
│ │ │ │ └── models                  # folder with model meta and .xmodel executable files
│ │ │ ├── scripts                   # scripts for model processing 
│ │ │ │ ├── inference.sh            # model inference
│ │ │ │ ├── performance.sh          # model performance report
│ │ │ │ ├── quality.sh              # model quality report
│ │ │ │ └── setup_venv.sh           # virtual environment creation
│ │ │ ├── src                       # python supporting scripts
│ │ │ │ └── quality.py              # quality metric calculation
│ │ │ ├── README.md
│ │ │ └── requirements.txt          # requirements for the virtual environment
│ │ ...
│ ├── semantic_segmentation
│ ...  
├── scripts                         # common scripts for all models 
├── AMD-license-agreement-for-non-commercial-models.md
├── downloader.py                   # python script for model files download
└── README.md
```
## Quick Start

### Prerequisites

1. Before starting, make sure that the host computer fully supports Xilinx FPGA/ACAP and the appropriate accelerator
is installed correctly, e.g.
[Xilinx VCK5000 Versal](https://xilinx.github.io/Vitis-AI/docs/board_setup/board_setup_vck5000.html).
Or you can use an already configured server on [vmaccel.com](https://www.vmaccel.com/).
2. Install the latest [Vitis-AI](https://xilinx.github.io/Vitis-AI/docs/install/install.html).
3. Go to the Vitis-AI repo: 
    ```
    # cd <Vitis-AI install path>/Vitis-AI
    # where:
    # <Vitis-AI install path> - the path where Vitis-AI was installed
    cd ~/Vitis-AI
    ```
4. Start the Vitis AI Docker: 
   ```
   # ./docker_run.sh xilinx/vitis-ai-<Framework>-<Arch>:latest
   # where:
   # <Framework>, <Arch> - deep learning framework and target architecture,
   # more info in the Vitis-AI installation instruction
   # Example:
   ./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
   ```
5. Download the test data: 
   ```
   bash model_zoo/scripts/download_test_data.sh
   ```
6. Make folders and subfolders to store artifacts: 
   ```
   cd model_zoo
   bash scripts/make_artifacts_folders.sh
   ```

### Native Inference

1. Follow the [Prerequisites chapter](#prerequisites): install the Vitis-AI, run the docker container, 
download test data, make folders to store artifacts.
2. All the following commands must be run inside the Vitis-AI container.
3. Select one of the available models from `model_zoo/models` and set the environment variable with the absolute
path to the model: 
   ```
   # MODEL_FOLDER="$(pwd)"/model_zoo/models/<application>/<model name>
   # where:
   # "$(pwd)" - the absolute path to the current folder inside the container, e.g.: /workspace
   # <application> - type or application of the model
   # <model name> - the name of the specific model
   # Example:
   MODEL_FOLDER="$(pwd)"/model_zoo/models/super_resolution/pt_DRUNet
   ```
4. Download model files for the specific device and device configuration:  
   ```
   cd /workspace/model_zoo
   python downloader.py
   # A command line interface will be provided for downloading model files
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
   ``` 
   # Example
   mv drunet_pt-vck5000-DPUCVDX8H-8pe-r3.0.0.tar.gz $MODEL_FOLDER/artifacts/models/
   tar -xzvf $MODEL_FOLDER/artifacts/models/drunet_pt-vck5000-DPUCVDX8H-8pe-r3.0.0.tar.gz -C $MODEL_FOLDER/artifacts/models/
   ```
6. Set environment variables for a specific device and device configuration inside the docker container:  
   ```
   # source /vitis_ai_home/board_setup/<DEVICE_NAME>/setup.sh <DEVICE_CONFIGURATION>
   # where:
   # <DEVICE_NAME> - the name of current device
   # <DEVICE_CONFIGURATION> - selected device configuration
   # Example:
   source /vitis_ai_home/board_setup/vck5000/setup.sh DPUCVDX8H_8pe_normal
   ```
7. Go to the specific model's folder inside the `model_zoo`:  
   ```
   cd $MODEL_FOLDER
   ```
8. Run the inference on files:  
   ```
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

## Vaitrace
You may profile the model performance using [Vaitrace](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/vaitrace-Usage) instrument.

> **Warning**
> To run the Vaitrace inside the docker container, you should have a **root permission**! <br>
> To the `Vitis-AI/docker_run.sh` script, add the following patch: 
   ```diff
   @@ -89,6 +71,7 @@ docker_run_params=$(cat <<-END
        -e USER=$user -e UID=$uid -e GID=$gid \
        -e VERSION=$VERSION \
        -v $DOCKER_RUN_DIR:/vitis_ai_home \
   +    -v /sys/kernel/debug:/sys/kernel/debug  --privileged=true \
        -v $HERE:/workspace \
        -w /workspace \
        --rm \
   ```

To run the Vaitrace, use: 
   ```
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
-  [Vitis AI Forums](https://support.xilinx.com/s/topic/0TO2E000000YKY9WAO/vitis-ai-ai?language=en_US)
-  <a href="mailto:xilinx_ai_model_zoo@amd.com">Email</a>

You can also submit a pull request with details on how to improve the product. 

Prior to submitting your pull request, ensure that you can build the product and run all the demos with your patch. In case of a larger feature, provide a relevant demo.
