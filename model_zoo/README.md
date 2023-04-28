<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


# Vitis AI Model Zoo

As of the 3.0 release of Vitis AI, the Model Zoo documentation and performance benchmarks have migrated to Github.IO.  **[YOU MAY ACCESS THE MODEL ZOO DOCUMENTATION ONLINE](https://xilinx.github.io/Vitis-AI/docs/workflow-model-zoo)** or **[OFFLINE](../docs/docs/workflow-model-zoo.html)**.

# Quick Start Prerequisites

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

## Contributing

We welcome community contributions. When contributing to this repository, first discuss the change you wish to make via:

-  [GitHub Issues](https://github.com/Xilinx/Vitis-AI/issues)
-  [Vitis AI Forums](https://forums.xilinx.com/t5/AI-and-Vitis-AI/bd-p/AI)
-  <a href="mailto:xilinx_ai_model_zoo@xilinx.com">Email</a>

You can also submit a pull request with details on how to improve the product. Prior to submitting your pull request, ensure that you can build the product and run all the demos with your patch. In case of a larger feature, provide a relevant demo.
