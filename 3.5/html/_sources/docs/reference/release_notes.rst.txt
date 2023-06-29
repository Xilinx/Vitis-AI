Release Notes 3.5
=================

Version Compatibility
---------------------

Vitis |trade| AI v3.5 and the DPU IP released with the v3.5 branch of this repository are verified as compatible with Vitis, Vivado |trade|, and PetaLinux version 2023.1. If you are using a previous release of Vitis AI, you should review the :ref:`version compatibility matrix <version-compatibility>` for that release.


Documentation and Github Repository
-----------------------------------
- Merged UG1313 into UG1414
- Streamlined UG1414 to remove redundant content
- Streamlined UG1414 to focus exclusively on core tool usage.  Core tools such as the Optimizer, Quantizer and Compiler are now being utilized across multiple targets (ie Ryzen |trade| AI, EPYC |trade|) and this change seeks to make UG1414 more portable to these targets
- Migrated Adaptable SoC and Alveo specific content from UG1414 to Github.IO
- New Github.IO Toctree structure
- Integrated VART Runtime APIs in Doxygen format

Docker Containers and GPU Support
----------------------------------
- Removed Anaconda dependency from TensorFlow 2 and PyTorch containers in order to address Anaconda commercial license requirements
- Updated Docker container to disable Ubuntu 18.04 support (which was available in Vitis AI but not officially supported).  This was done to address `CVE-2021-3493 <https://nvd.nist.gov/vuln/detail/CVE-2021-3493>`__.																																   

Model Zoo
---------
- Add more classic models without modification such as YOLO series and 2D Unet 
- Provide model info card for each model and Jupyter Notebook tutorials for new models
- New copyleft repo for GPL license models

ONNX CNN Quantizer
------------------
- Initial release
- This is a new quantizer that supports the direct PTQ quantization of ONNX models for DPU. It is a plugin built for the ONNXRuntime native quantizer.
- Supports power-of-two quantization with both QDQ and QOP format.
- Supports Non-overflow and Min-MSE quantization methods.
- Supports various quantization configurations in power-of-two quantization in both QDQ and QOP format.
- Supports signed and unsigned configurations.
- Supports symmetry and asymmetry configurations.
- Supports per-tensor and per-channel configurations.
- Supports ONNX models in excess of 2GB.
- Supports the use of the CUDAExecutionProvider for calibration in quantization.
 
PyTorch CNN Quantizer
---------------------
- Pytorch 1.13 and 2.0 support
- Mixed precision quantization support, supporting float32/float16/bfloat16/intx mixed quantization
- Support of bit-wise accuracy cross check between quantizer and ONNX-runtime
- Split and chunk operators were automatically converted to slicing
- Dict input/output support for model forward function
- Keywords argument support for model forward function
- Matmul subroutine support
- Add support for BFP data type quantization
- QAT supports training on mutiple GPUs
- QAT supports operations with multiple inputs or outputs

TensorFlow 2 CNN Quantizer
--------------------------
- Updated to support Tensorflow 2.12 and Python 3.8.
- Adds support for quantizing subclass models.
- Adds support for mix precision, supports layer-wise data type configuration, supports float32, float16, bfloat16, and int quantization. 
- Adds support for BFP datatypes, and add a new quantize strategy called 'bfp'.
- Adds support to quantize Keras nested models.
- Adds experimental support for quantizing the frozen pb format model in TensorFlow 2.x.
- Adds a new 'gpu' quantize strategy which uses float scale quantization and is used in GPU deployment scenarios.
- Adds support to exporting the quantized model to frozen pb format or onnx format.
- Adds support to exporting the quantized model with power-of-two scales to frozen pb format with "FixNeuron" inside, to be compatible with some compilers with pb format input.
- Adds support for splitting Avgpool and Maxpool with large kernel sizes into smaller kernel sizes.

Bug Fixes:
1.	Fixes a gradient bug in the 'pof2s_tqt' quantize strategy.
2.	Fixes a bug of quantization position change introduced by the fast fine-tuning process after the PTQ.
3.	Fixes a graph transformation bug when a TFOpLambda op has multiple inputs.

TensorFlow 1 CNN Quantizer
--------------------------
- Adds support for fast fine-tuning that improves PTQ accuracy.
- Adds support for folding Reshape and ResizeNearestNeighbor operators.
- Adds support for splitting Avgpool and Maxpool with large kernel sizes into smaller kernel sizes.
- Adds support for quantizing Sum, StridedSlice, and Maximum operators.
- Adds support for setting the input shape of the model, which is useful in the deployment of models with undefined input shapes.
- Adds support for setting the opset version in exporting onnx format.

Bug Fixes:
1.	Fixes a bug where the AddV2 operation is misunderstood as a BiasAdd.

Compiler
--------
- Release notes to be announced ASAP

PyTorch Optimizer
-----------------
- Removed requirement for license purchase
- Migrated to Github open-source
- Supports PyTorch 1.11, 1.12 and 1.13
- Supports pruning of grouped convolution
- Supports setting the number of channels to be a multiple of the specified number after pruning

TensorFlow 2 Optimizer
----------------------
- Removed requirement for license purchase
- Migrated to Github open-source
- Supports TensorFlow 2.11 and 2.12
- Supports pruning of tf.keras.layers.SeparableConv2D
- Fixed tf.keras.layers.Conv2DTranspose pruning bug
- Supports setting the number of channels to be a multiple of the specified number after pruning

Runtime
-------
- Supports Versal AI Edge VEK280 evalustion kit
- Buffer optimization for multi-batches to improve performance 
- Add new tensor buffer interface to enhance zero copy

Vitis ONNX Runtime Execution Provider (VOE)
-------------------------------------------
- Supports ONNX Opset version 18, ONNX Runtime 1.16.0 and ONNX version 1.13
- Supports both C++ and Python APIs(Python version 3)
- Supports VitisAI EP and other EPs to work together to deploy the model
- Provide Onnx examples based on C++ and Python APIs
- VitisAI EP is open source and upstreamed to ONNX public repo on Github

Library
-------
- Added three new model libraries and support for five additional models

Model Inspector
---------------
- Release notes to be announced ASAP

Profiler
--------
- Added Profiler support for DPUCV2DX8G

DPU IP - Versal AIE-ML Targets DPUCV2DX8G (Versal AI Edge / Core)
-----------------------------------------------------------------------------
- First general access release
- Configurable from C20B1 to C20B14
- Support most 2D operators required to deploy models found in the Model Zoo
- General support for the VE2802/VC2802 and V70
- Early access support for the VE2302 via `this lounge <https://www.xilinx.com/member/vitis-ai-vek280.html>`__

DPU IP - Zynq Ultrascale+ DPUCZDX8G
-----------------------------------
- IP has reached maturity
- No updates for this release
- No updated reference design (DPU TRD) will be published for minor (ie x.5) releases
- No updated pre-built board image will be published for minor (ie x.5) releases

DPU IP - Versal AIE Targets DPUCVDX8H
-------------------------------------
- IP has reached maturity
- No updates for this release
- No updated reference design (DPU TRD) will be published for minor (ie x.5) releases
- No updated pre-built board image will be published for minor (ie x.5) releases																					 

DPU IP - CNN - Alveo Data Center DPUCVDX8G 
------------------------------------------
- IP has reached maturity
- No updates for this release
- No updated reference design (DPU TRD) will be published for minor (ie x.5) releases
- No updated pre-built board image will be published for minor (ie x.5) releases																					 

WeGO
------------------------------------------
- Enhanced WeGO to support V70 DPU GA release. 
- Upgraded WeGO to provide support for PyTorch 1.13.1 and TensorFlow r2.12. 
- Enhanced WeGO-Torch to support PyTorch 2.0 as a preview feature.
- Introduced new C++ API support for WeGO-Torch in addition to Python APIs. 
- Implemented WeGO-TF1 and WeGO-TF2 as out-of-tree plugins.

Known Issues
------------
- Engineering to add comments

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
   
   
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc.
