<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

## Vitis AI Runtime v3.0

# Introduction

Vitis AI Run time enables applications to use the unified high-level runtime API for both data center and embedded applications. Therefore, making cloud-to-edge deployments seamless and efficient.
The Vitis AI Runtime API features are:
* Asynchronous submission of jobs to the accelerator
* Asynchronous collection of jobs from the accelerator
* C++ and Python implementations
* Support for multi-threading and multi-process execution

In this release, VART are fully open source except the Python interfaces and DPUCADX8G interfaces. It contains the following 4 modules.
* unilog
* xir
* target_factory
* vart

Vitis AI Runtime directory structure introduction
--------------------------------------------------

```
vai_runtime
├── plugin-samples
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── cmake.sh
│   ├── README.md
│   └── samples
├── quick_start_for_embedded.md
├── quick_start_for_rnn.md
├── README.md
├── target_factory
│   ├── build.bat
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── cmake.sh
│   ├── Copyright.txt
│   ├── expired_targets
│   ├── include
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.py
│   ├── src
│   ├── targets
│   └── test
├── unilog
│   ├── build.bat
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── cmake.sh
│   ├── Copyright.txt
│   ├── include
│   ├── meta.yaml
│   ├── python
│   ├── README.md
│   ├── src
│   └── test
├── vart
│   ├── async-runner
│   ├── buffer-object
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── cmake.sh
│   ├── Copyright.txt
│   ├── dpu-controller
│   ├── dpu-runner
│   ├── dummy-runner
│   ├── mem-manager
│   ├── README.md
│   ├── rnn-runner
│   ├── runner
│   ├── softmax-runner
│   ├── softmax-runner-cpu
│   ├── trace
│   ├── trace-logging
│   ├── util
│   ├── xrnn-runner
│   └── xrt-device-handle
└── xir
    ├── 3rd-party
    ├── build.bat
    ├── cmake
    ├── CMakeLists.txt
    ├── cmake.sh
    ├── Copyright.txt
    ├── doc
    ├── include
    ├── README.md
    ├── src
    ├── test
    └── tools
```
# Quick Start
For edge users, click 
[Quick Start For Embedded](./quick_start_for_embedded.md) to get started quickly. 

For rnn users, click 
[Quick Start For RNN](./quick_start_for_rnn.md) to get started quickly.
