<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Runtime v2.5</h1>
    </td>
 </tr>
 </table>

# Introduction
Vitis AI Run time enables applications to use the unified high-level runtime API for both cloud and edge. Therefore, making cloud-to-edge deployments seamless and efficient.
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
VART
├── plugin-samples
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── cmake.sh
│   ├── README.md
│   └── samples
├── quick_start_for_edge.md
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
[Quick Start For Edge](./quick_start_for_edge.md) to get started quickly. 

For rnn users, click 
[Quick Start For RNN](./quick_start_for_rnn.md) to get started quickly.
