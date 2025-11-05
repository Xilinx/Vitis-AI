Release Notes 3.5
=================

Version Compatibility
---------------------

Vitis |trade| AI v3.5 and the DPU IP released with the v3.5 branch of this repository are verified as compatible with Vitis, Vivado |trade|, and PetaLinux version 2023.1. If you are using a previous release of Vitis AI, you should review the :ref:`version compatibility matrix <version-compatibility>` for that release.


Documentation and Github Repository
-----------------------------------
- Merged UG1333 into UG1414
- Streamlined UG1414 to remove redundant content
- Streamlined UG1414 to focus exclusively on core tool usage.  Core tools such as the Optimizer, Quantizer and Compiler are now being utilized across multiple targets (ie Ryzen |trade| AI, EPYC |trade|) and this change seeks to make UG1414 more portable to these targets
- Migrated Adaptive SoC and Alveo specific content from UG1414 to Github.IO
- New Github.IO Toctree structure
- Integrated VART Runtime APIs in Doxygen format

Docker Containers and GPU Support
----------------------------------
- Removed Anaconda dependency from TensorFlow 2 and PyTorch containers in order to address Anaconda commercial license requirements
- Updated Docker container to disable Ubuntu 18.04 support (which was available in Vitis AI but not officially supported).  This was done to address `CVE-2021-3493 <https://nvd.nist.gov/vuln/detail/CVE-2021-3493>`__.																																   

Model Zoo
---------
- Added more classic models without modification such as YOLO series and 2D Unet 
- Provided model info card for each model and Jupyter Notebook tutorials for new models
- New copyleft repo for GPL license models

ONNX CNN Quantizer
------------------
- Initial release
- This is a new quantizer that supports the direct PTQ quantization of ONNX models for DPU. It is a plugin built for the ONNXRuntime native quantizer.
- Support for power-of-two quantization with both QDQ and QOP format.
- Support for Non-overflow and Min-MSE quantization methods.
- Support for various quantization configurations in power-of-two quantization in both QDQ and QOP format.
- Support for signed and unsigned configurations.
- Support for symmetry and asymmetry configurations.
- Support for per-tensor and per-channel configurations.
- Support for ONNX models in excess of 2GB.
- Support for the use of the CUDAExecutionProvider for calibration in quantization.
 
PyTorch CNN Quantizer
---------------------
- Support for Pytorch 1.13 and 2.0
- Support for mixed precision quantization, float32/float16/bfloat16/intx
- Support for bit-wise accuracy cross check between quantizer and ONNX-runtime
- Split and chunk operators were automatically converted to slicing
- Dict input/output supports for model forward function
- Keywords argument supports for model forward function
- Support for matmul subroutine
- Added support for BFP data type quantization
- QAT supports training on mutiple GPUs
- QAT supports operations with multiple inputs or outputs

TensorFlow 2 CNN Quantizer
--------------------------
- Updated to support for Tensorflow 2.12 and Python 3.8.
- Support for quantizing subclass models.
- Support for mix precision, supports layer-wise data type configuration, supports float32, float16, bfloat16, and int quantization. 
- Support for BFP datatypes, and add a new quantize strategy called 'bfp'.
- Support for quantize Keras nested models.
- Experimental support for quantizing the frozen pb format model in TensorFlow 2.x.
- Added a new 'gpu' quantize strategy which uses float scale quantization and is used in GPU deployment scenarios.
- Support for exporting the quantized model to frozen pb format or onnx format.
- Support for exporting the quantized model with power-of-two scales to frozen pb format with "FixNeuron" inside, to be compatible with some compilers with pb format input.
- Support for splitting Avgpool and Maxpool with large kernel sizes into smaller kernel sizes.

Bug Fixed:
1.	Fixed a gradient bug in the 'pof2s_tqt' quantize strategy.
2.	Fixed a bug of quantization position change introduced by the fast fine-tuning process after the PTQ.
3.	Fixed a graph transformation bug when a TFOpLambda op has multiple inputs.

TensorFlow 1 CNN Quantizer
--------------------------
- Support for fast fine-tuning that improves PTQ accuracy.
- Support for folding Reshape and ResizeNearestNeighbor operators.
- Support for splitting Avgpool and Maxpool with large kernel sizes into smaller kernel sizes.
- Support for quantizing Sum, StridedSlice, and Maximum operators.
- Support for setting the input shape of the model, which is useful in the deployment of models with undefined input shapes.
- Support for setting the opset version in exporting onnx format.

Bug Fixed:
1.	Fixed a bug where the AddV2 operation is misinterpreted as a BiasAdd.

Compiler
--------
- New operators supported: Broadcast add/mul, Bilinear downsample, Trilinear downsample, Group conv2d, Strided-slice
- Performance improved on XV2DPU
- Error message improved
- Compilation time speed up

PyTorch Optimizer
-----------------
- Removed requirement for license purchase
- Migrated to Github open-source
- Support for PyTorch 1.11, 1.12 and 1.13
- Support for pruning of grouped convolution
- Support for setting the number of channels to be a multiple of the specified number after pruning

TensorFlow 2 Optimizer
----------------------
- Removed requirement for license purchase
- Migrated to Github open-source
- Support for TensorFlow 2.11 and 2.12
- Support for pruning of tf.keras.layers.SeparableConv2D
- Fixed tf.keras.layers.Conv2DTranspose pruning bug
- Support for setting the number of channels to be a multiple of the specified number after pruning

Runtime
-------
- Supports Versal AI Edge VEK280 evaluation kit
- Buffer optimized for multi-batches to improve performance 
- Added new tensor buffer interface to enhance zero copy

Vitis ONNX Runtime Execution Provider (VOE)
-------------------------------------------
- Support for ONNX Opset version 18, ONNX Runtime 1.16.0 and ONNX version 1.13
- Support for both C++ and Python APIs(Python version 3)
- Support for Vitis AI EP and other EPs to work together to deploy the model
- Provided Onnx examples based on C++ and Python APIs
- Vitis AI EP is open sourced and upstreamed to ONNX public repo on Github

Library
-------
- Added three new model libraries and support for five additional models

Model Inspector
---------------
- Added support for DPUCV2DX8G

Profiler
--------
- Added Profiler support for DPUCV2DX8G

DPU IP - Versal AIE-ML Targets DPUCV2DX8G (Versal AI Edge / Core)
-----------------------------------------------------------------------------
- General access release for the Versal AI Edge device VE2802, Versal AI Core device VC2802 and Alveo V70 card
- Configurable from C20B1 to C20B14
- Support most 2D operators based on models in the Model Zoo

DPU IP - Zynq Ultrascale+ DPUCZDX8G
-----------------------------------
- No DPU IP updates in  3.5 release
- No DPU reference design updates in 3.5 release
- No pre-built board image updates in 3.5 release

DPU IP - Versal AIE Targets DPUCVDX8G
-------------------------------------
- No DPU IP updates in 3.5 release
- No DPU reference design updates in 3.5 release
- No pre-built board image updates in 3.5 release																					 

DPU IP - CNN - Alveo Data Center DPUCVDX8H
------------------------------------------
- No DPU IP updates in 3.5 release
- No DPU reference design  updates in 3.5 release
- No pre-built board image updates in 3.5 release																					 

WeGO
------------------------------------------
- Support for Alveo V70 DPU GA release. 
- Support for PyTorch 1.13.1 and TensorFlow r2.12. 
- Enhanced WeGO-Torch to support PyTorch 2.0 as a preview feature.
- Introduced new C++ API that supports for WeGO-Torch 
- Implemented WeGO-TF1 and WeGO-TF2 as out-of-tree plugins.

Known Issues
------------
- To be announced ASAP

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
   
   
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc.
