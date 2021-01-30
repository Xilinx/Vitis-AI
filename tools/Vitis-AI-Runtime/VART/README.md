<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Runtime v1.3.1</h1>
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
├── README.md
├── quick_start_for_cloud.md
├── quick_start_for_edge.md
├── target_factory
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── cmake.sh
│   ├── Copyright.txt
│   ├── include
│   ├── README.md
│   ├── src
│   ├── targets
│   └── test
├── unilog
│   ├── build.sh
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── cmake.sh
│   ├── Copyright.txt
│   ├── include
│   ├── meta.yaml
│   ├── README.md
│   ├── src
│   └── test
├── vart
│   ├── buffer-object
│   ├── build.sh
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── cmake.sh
│   ├── conda
│   ├── config
│   ├── Copyright.txt
│   ├── cpu-runner
│   ├── dpu-controller
│   ├── dpu-runner
│   ├── dpuv1-runner
│   ├── mem-manager
│   ├── meta.yaml
│   ├── README.md
│   ├── runner
│   ├── sim-runner
│   ├── util
│   └── xrt-device-handle
└── xir
    ├── cmake
    ├── CMakeLists.txt
    ├── cmake.sh
    ├── Copyright.txt
    ├── doc
    ├── include
    ├── meta.yaml
    ├── README.md
    ├── src
    ├── test
    └── tools
```
# Quick Start
For edge users, click 
[Quick Start For Edge](./quick_start_for_edge.md) to get started quickly. 

For cloud users, click 
[Quick Start For Alveo](./quick_start_for_cloud.md) to get started quickly.

For rnn users, click 
[Quick Start For RNN](./quick_start_for_rnn.md) to get started quickly.