Release Notes 3.0
====================

Documentation and Github Repository
-----------------------------------
- Migrated core documentation to `Github.IO <https://xilinx.github.io/Vitis-AI/>`__.
- Incorporated offline HTML documentation for air-gapped users.
- Restructured user documentation.
- Restructured repository directory structure for clarity and ease-of-use.

Docker Containers and GPU Support
----------------------------------
- Migrated from multi-framework to per framework Docker containers.
- Enabled Docker ROCm GPU support for quantization and pruning.

Model Zoo
~~~~~~~~~
- Updated Model Zoo with commentary regarding dataset licensing restrictions
- Added 14 new models and deprecated 28 models for a total of 130 models
- Added super resolution 4x, as well as 2D and 3D semantic segmentation for Medical applications
- Optimized models for benchmarks:
	- MLPerf: 3D-Unet
	- FAMBench: MaskRCNN
- Provides optimized backbones supporting YoloX, v4, v5, v6 and EfficientNet-Lite
- Ease-of-use enhancements, including replacing markdown-format performance tables with a downloadable Model Zoo spreadsheet
- Added 72 PyTorch and TensorFlow models for AMD EPYC |trade| CPUs, targeting deployment with ZenDNN
- Added models to support AMD GPU architectures based on ROCm and MLGraphX

TensorFlow 2 CNN Quantizer
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Based on TensorFlow 2.10
- Updated the Model Inspector to for improved accuracy of partitioning results expected from the DPU compiler.
- Added support for datatype conversions for float models, including FP16, BFloat16, FP32, and double.
- Added support for exporting quantized ONNX format models (to support the ONNX Runtime).
- Added support for new layer types including SeparableConv2D and PReLU.
- Added support for unsigned integer quantization.
- Added support for automatic modification of input shapes for models with variable input shapes.
- Added support to align the input and output quantize positions for Concat and Pooling layers.
- Added error codes and improved the readability of the error and warning messages.
- Various bug fixes.

TensorFlow 1 CNN Quantizer
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Separated the quantizer code from the TensorFlow code, making it a plug-in module to the official TensorFlow code base.
- Added support for exporting quantized ONNX format models (to support the ONNX Runtime).
- Added support for datatype conversions for float models, including FP16, BFloat16, FP32 and double.
- Added support for additional operations, including Max, Transpose, and DepthToSpace.
- Added support for aligning input and output quantize positions of Concat and Pooling operations.
- Added support for automatic replacement of Softmax with DPU-accelerated Softmax.
- Added error codes and improved the readability of the error and warning messages.
- Various bug fixes.

PyTorch CNN Quantizer
~~~~~~~~~~~~~~~~~~~~~
- Support PyTorch 1.11 and 1.12.
- Support exporting torch script format quantized model.
- QAT supports exporting trained model to ONNX and torch script.
- Support FP16 model quantization.
- Optimized Inspector to support more pattern types, and backward compatible of device assignment.
- Cover more PyTorch operators: more than 560 types of PyTorch operators are supported.
- Enhanced parsing to support control flow parsing.
- Enhanced message system with more useful message text.
- Support fusing and quantization of BatchNorm without affine calculation.

Compiler
~~~~~~~~
- Added support for new operators, including: strided_slice, cost volume, correlation 1D & 2D, argmax, group conv2d, reduction_max, reduction_mean
- Added support for Versal |trade| AIE-ML architectures DPUCV2DX8G (V70 and Versal AI Edge)
- Focused effort to improve the intelligibility of error and partitioning messages

PyTorch Optimizer
~~~~~~~~~~~~~~~~~
- Added support for fine-grained model pruning (sparsity)
- OFA support for convolution layers with kernel sizes = (1,3) and dialation
- OFA support for ConvTranspose2D
- Added pruning configuration that allows users to specify pruning hyper-parameters
- Specific exception types are defined for each type of error
- Enhanced parallel model analysis with increased robustness
- Support for PyTorch 1.11 and 1.12

TensorFlow 2 Optimizer
~~~~~~~~~~~~~~~~~~~~~~
- Added support for Keras ConvTranspose2D, Conv3D, ConvTranspose3D
- Added support TFOpLambda operator
- Added pruning configuration that allows users to specify pruning hyper-parameters
- Specific exception types are defined for each type of error
- Added support for TensorFlow 2.10

Runtime and Library
~~~~~~~~~~~~~~~~~~~
- Added support for Versal AI Edge VEK280 evaluation kit and Alveo |trade| V70 accelerator cards (Early Access)
- Added support for ONNX runtime, with eleven ONNX-specific examples
- Added four new model libraries to the Vitis |trade| AI Library and support for fifteen additional models
- Focused effort to improve the intelligibility of error messages

Profiler
~~~~~~~~
- Added Profiler support for DPUCV2DX8G (VEK280 Early Access)
- Added Profiler support for Versal DDR bandwidth profiling

DPU IP - Zynq Ultrascale+ DPUCZDX8G
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Upgraded to enable Vivado |trade| and Vitis 2022.2 release
- Added support for 1D and 2D Correlation, Argmax and Max
- Reduced resource utilization
- Timing closure improvements

DPU IP - Versal AIE Targets DPUCVDX8G
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Upgraded to enable Vivado and Vitis 2022.2 release
- Added support for 1D and 2D Correlation
- Added support for Argmax and Max along the channel dimension
- Added support for Cost-Volume
- Reduced  resource utilization
- Timing closure improvements

DPU IP - Versal AIE-ML Targets DPUCV2DX8G (Versal AI Edge)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Early access release supporting early adopters with an early, unoptimized AIE-ML DPU
- Supports most 2D operators (currently does not support 3D operators)
- Batch size support from 1~13
- Supports more than 90 Model Zoo models

DPU IP - CNN - Alveo Data Center DPUCVDX8H 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Upgraded to enable Vitis 2022.2 release
- Timing closure improvements via scripts supplied for .xo workflows

DPU IP - CNN - V70 Data Center DPUCV2DX8G
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Early access release supporting early adopters with an unoptimized DPU
- Supports most 2D operators (currently does not support 3D operators)
- Batch size 13 support
- Supports more than 70 Model Zoo models

Legacy Alveo DPU Support
~~~~~~~~~~~~~~~~~~~~~~~~
- Vitis AI support for the DPUCAHX8H/DPUCAHX8H-DWC IP, and Alveo™ U50LV and U55C cards was discontinued with the release of Vitis AI 3.0. The final release to support these targets was Vitis AI 2.5.0.


WeGO
~~~~
- Integrated WeGO with the Vitis-AI Quantizer to enable on-the-fly quantization and improve easy-of-use
- Introduced serialization and deserialization with the WeGO flow to offer the capability of building once and running anytime
- Incorporated AMD ZenDNN into WeGO, enabling additional optimization for AMD EPYC CPU targets
- Improve WeGO robustness to offer a better developer experience and support a wider range of models

Known Issues
------------
- Bitstream loading error occurs when the AIE-ML DPU application running on the VEK280 kit is interrupted manually
- HDMI not functional for the early access VEK280 image. The issue will be fixed in the next release

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
   
   
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc.