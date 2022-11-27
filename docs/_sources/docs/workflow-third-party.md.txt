
## Third-party Inference Stack Integration

Vitis-AI provides integration support for TVM, ONNXRuntime and TensorFlow Lite workflows. The subfolders here are the launch point for developers who wish to leverage these workflows.  A brief description of these workflows is presented below.

### TVM

[TVM.ai](https://tvm.apache.org/) is Apache Software Foundation project and inference stack that can parse machine learning models from almost any training framework.  The model is converted to an intermediate representation (TVM relay), and the stack can then compile the model for a variety of targets, including embedded SoCs, CPUs, GPUs as well as x86 and x64 platforms.  TVM incorporates an open-source programmable-logic accelerator known as the VTA, which was created using the Xilinx HLS compiler.  TVM supports the capability of partitioning a graph into a number of sub-graphs, and these sub-graphs may be targeted to specific accelerators within the target platform (CPU, GPU, VTA....) with the goal of enabling heterogeneous acceleration.

For the published Vitis AI - TVM workflow, the VTA is not used, instead opting to integrate the DPU for offloading of compiled subgraphs.  Subgraphs that can be partitioned for execution on the DPU are quantized and compiled by the Vitis AI compiler for a specific DPU target, while the remaining subgraphs and operations are compiled by the TVM compiler for execution on LLVM.

Additional details of Vitis AI - TVM integration can be found [here](https://tvm.apache.org/docs/how_to/deploy/vitis_ai.html)

<div align="center">
  <img width="100%" height="100%" src="../docs/reference/images/VAI_3rd_party_TVM.PNG">
</div>


</br></br>
### ONNXRuntime

[ONNXRuntime](https://onnxruntime.ai/) was devised as a cross-platform inference deployment runtime for ONNX models.  ONNXRuntime provides the benefit of runtime-interpretation of models represented as an ONNX intermediate representation (IR).  

The [ONNXRuntime Execution Provider](https://onnxruntime.ai/docs/execution-providers/) framework enables the integration of proprietary or customized tensor accelerator cores from any "execution provider".  Such "execution providers" are typically tensor acceleration IP blocks, integrated into an SoC by the semiconductor vendor.  The ability of a given accelerator to offload operations is presented as a listing of capabilities to the ONNXRuntime.  Specific subgraphs or operations within the ONNX graph may then be offloaded to that core based on the advertised capabilities of that execution provider.

Vitis AI Execution Provider support has been integrated as an [experimental flow](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/third_party/onnxruntime) in recent releases.

Additional details of the Vitis AI Execution Provider can be found [here](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html).

<div align="center">
  <img width="100%" height="100%" src="../docs/reference/images/VAI_3rd_party_ONNXRuntime.PNG">
</div>


</br></br>
### TensorFlow Lite

TensorFlow Lite has been used as the preferred inference solution for TensorFlow users in the embedded space for many years.  TensorFlow Lite provides the benefit of runtime-interpretation of models trained in TensorFlow Lite, with the implication that no compilation is required in order to execute the model on-target.  TensorFlow Lite provides support for embedded ARM processors, as well as NEON tensor acceleration.  This has made TensorFlow Lite a convenient solution for embedded and mobile MCU targets which did not incorporate purpose-built tensor acceleration cores.

With the addition of [TensorFlow Delegates](https://www.tensorflow.org/lite/performance/delegates), it became possible for semiconductor vendors with purpose-built tensor accelerators to integrate support into the TensorFlow Lite framework.  Certain operations can be offloaded (delegated) to these specialized accelerators, repositioning TensorFlow Lite runtime interpretation as a useful workflow in the high-performance space.

Vitis AI Delegate support has been integrated as an [experimental flow](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/third_party/tflite) in recent releases.


<div align="center">
  <img width="100%" height="100%" src="../docs/reference/images/VAI_3rd_party_TFLite.PNG">
</div>
