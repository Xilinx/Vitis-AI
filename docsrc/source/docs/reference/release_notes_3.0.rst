====================
Release Notes 3.0
====================


Documentation and Github Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Migrated core documentation to `Github.IO <https://xilinx.github.io/Vitis-AI/>`__
- Incorporated offline HTML documentation for air-gapped users
- Restructured user documentation
- Restructured repository directory structure for clarity and ease-of-use

Docker Containers and GPU Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Migrated from multi-framework to per framework Docker containers
- Enabled Docker ROCm GPU support for quantization and pruning

Model Zoo
~~~~~~~~~

- Added 24 new models and deprecated 28 models for a total of 130 models
- Added super resolution 4x, as well as 2D and 3D semantic segmentation for Medical applications
- Added BERT models for Data Center NLP applications
- Added MLPerf models: 3D-Unet, FAMBench: MaskRCNN, superbenchmark: BERT (tiny, base, large)
- Added new YOLO variants: YoloX, v5, v6
- Added EfficientNet-lite
- Ease-of-use enhancements, including replacing markdown-format performance tables with a downloadable Model Zoo spreadsheet
- Added 60 PyTorch and TensorFlow models for AMD EPYC CPUs, targeting deployment with ZenDNN
- Added models to support AMD GPU architectures, supporting ROCm and MLGraphX

CNN Quantizer
~~~~~~~~~~~~~
TBA

RNN Quantizer
~~~~~~~~~~~~~
TBA

Compiler
~~~~~~~~

- Added support for new operators, including: strided_slice, cost volume, correlation 1D & 2D, argmax, group conv2d, reduction_max, reduction_mean
- Added support for Versal AI Edge DPUCV2DX8G
- Focused effort to improve the intelligibility of error and partitioning messages

Optimizer
~~~~~~~~~
TBA

Runtime, Library, Profiler
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for VEK280 and V70 targets (Early Access)
- Added support for ONNX runtime, with eleven examples
- Added four new model libraries to the Vitis AI Library and support for thirteen additional models
- Focused effort to improve the intelligibility of error messages
- Added Profiler support for DPUCV2DX8G (VEK280 Early Access)
- Added Profiler support for Versal DDR bandwidth profiling

DPU IP - Zynq Ultrascale+
~~~~~~~~~~~~~~~~~~~~~~~~~
- Upgraded to enable Vivado and Vitis 2022.2 release
- Added support for 1D and 2D Correlation, Argmax and Max
- Reduced resource utilization
- Timing closure improvements

DPU IP - Versal AIE Targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Upgraded to enable Vivado and Vitis 2022.2 release
- Added support for 1D and 2D Correlation
- Added support for Argmax and Max along the channel dimension
- Added support for Cost-Volume
- Reduced  resource utilization
- Timing closure improvements

DPU IP - Versal AIE-ML Targets (Versal AI Edge)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Early access release supporting early adopters with an early, unoptimized AIE-ML DPU
- Supports most 2D operators (currently does not support 3D operators)
- Batch size support from 1~13
- Supports more than 70 Model Zoo models

DPU IP - CNN - Alveo Data Center
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Upgraded to enable Vitis 2022.2 release
- Timing closure improvements via scripts supplied for .xo workflows

DPU IP - CNN - V70 Data Center
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Early access release supporting early adopters with an unoptimized DPU
- Supports most 2D operators (currently does not support 3D operators)
- Batch size 13 support
- Supports more than 70 Model Zoo models

DPU IP - Non-CNN - Data Center
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Integrate WeGO with the Vitis-AI Quantizer to enable on-the-fly quantization and improve easy-of-use
- Introduced serialization and deserialization with the WeGO flow to offer the capability of building once and running anytime
- Incorporated AMD ZenDNN into WeGO, enabling additional optimization for AMD EPYC CPU targets
- Improve WeGO robustness to offer a better developer experience and support a wider range of models

Whole-Application Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TBA

AI Kernel Scheduler
~~~~~~~~~~~~~~~~~~~
TBA

Third-party Workflows
~~~~~~~~~~~~~~~~~~~~~
TBA