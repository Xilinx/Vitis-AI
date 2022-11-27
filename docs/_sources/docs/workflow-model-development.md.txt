
##  Model Inspector

-----------------------
📌**Important Note** - The first release of Model Inspector included with Vitis AI 2.5, should be considered to be of "beta" quality.  Please submit Github issues if you encounter problems with this initial release.

-----------------------

The Vitis AI quantizer and compiler are designed to parse and compile operators within a frozen, FP32 graph for acceleration in hardware.  However, novel neural networks architectures, operators and activation types are constantly being developed and optimized for prediction accuracy and performance.  In this context, it is important to understand that while we strive to provide support for a wide variety of neural network architectures and provide these graphs for user reference, not every operator is supported for acceleration on the DPU.  Furthermore, there are specific layer ordering requirements that enable Vitis AI model deployment.

In the early phases of development, it is highly recommended that the developer leverage the Vitis AI Model Inspector as an initial sanity check to confirm that the operators, and sequence of operators in the graph is compatible with Vitis AI.

<div align="center">
  <img width="100%" height="100%" src="reference/images/Model_Inspector_Flow.PNG">
</div>
</br>

-----------------------
#### *Model Inspector related resources:*
- When you are ready to get started with the Vitis AI Model Inspector, refer to the examples provided for both [PyTorch](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/inspector_tutorial.ipynb) and [TensorFlow](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Quantizer/vai_q_tensorflow2.x/#inspecting-vai_q_tensorflow2)
- If you determine that your graph uses operators that are not natively supported by your specific DPU target, there are several paths that you can explore, discussed in [Operator Support](#operator-support)

-----------------------

## Operator Support

In some cases, the user will want to leverage an operator that is not supported for acceleration on the DPU.  There are several obvious paths available, including C/C++ code, or even custom HLS or RTL kernels.  However, these DIY paths pose specific challenges related to the partitioning of a trained model.  For most developers, a workflow that supports automated partitioning is preferred.

-----------------------
📌**Important Note!**  
A high-level list of the supported operators is provided in the DPU IP product [guides](reference/release_documentation).  Both the Vitis AI quantizer and compiler implement layer fusion, using a pattern-match algorithm.  The net result of this is that the ordering of layers in the graph can be as important as the operators used.  For instance, if you implement a layer ordering scheme such as CONV -> ReLU -> Batchnorm, the outcome is quite different than [CONV -> Batchnorm -> ReLU](https://support.xilinx.com/s/question/0D52E00006hpW23SAE/resolving-debugging-shiftcut0-tensorflow?language=en_US).  In this context, it is always a very good idea to review the structure of similar Xilinx [Model Zoo](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/model_zoo) models to understand how to design your graph for optimum results.

-----------------------

For MPSoC and Versal embedded applications, Xilinx supports an official flow by which the user can add support for these custom operators.  More details can be found [here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/examples/custom_operator)

For Alveo cards, the [Whole Graph Optimizer](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/examples/wego) (WeGO) automatically performs subgraph partitioning for models quantized by Vitis AI quantizer, and applies optimizations and acceleration for the DPU compatible subgraphs. The remaining partitions of the graph are dispatched to the native framework for CPU execution.

In addition, the TVM compiler, TF Lite Delegate, and ONNXruntime Execution Provider (Alveo only) [solutions](workflow-third-party) may also be used to enable support for operations that cannot be accelerated by the DPU.  It is important to note that these third party solutions should be considered to be of "beta" quality and offer more limited support than the standard Vitis AI workflow.


## Model Optimization

The Vitis AI Optimizer exploits the notion of sparsity in order to reduce the overall computational complexity for inference.  Many deep neural networks topologies employ significant levels of redundancy.  This is particularly true when the network backbone was optimized for prediction accuracy with training datasets supporting a large number of classes.  In many cases, this redundancy can be reduced by "pruning" some of the operations out of the graph.  In general, there are two major forms of pruning - channel (kernel) pruning and sparse pruning.

-----------------------
📌**Important Note!**

The Vitis AI Optimizer is an optional tool which can provide considerable uplift in performance in many applications.  However, if your application is not hitting the wall on performance or logic density, or if your model is already well optimized for your dataset and application, you will likely not require the AI Optimizer.

-----------------------

The use of the Vitis AI Optimizer requires that the developer purchase a license for the tool.  You can request a quotation for either the node-locked (part# EF-AI-OPTIMIZER-NL) or floating license (part# EF-AI-OPTIMIZER-FL) by contacting your local [Xilinx Distributor or Sales Office](https://www.xilinx.com/about/contact.html).  **This is a perpetual license with no annual maintenance or renewal costs.**  

Should you wish to evaluate the AI Optimizer prior to considering purchase, you can request access by emailing xilinx_ai_optimizer@amd.com, or request access to the [AI Optimizer Lounge](https://www.xilinx.com/member/ai_optimizer.html)

The Vitis AI Optimizer leverages the native framework in which the model was trained, and both the input and output of the pruning process are a frozen FP32 graph.  At a high-level, the workflow of the AI Optimizer consists of several steps.  The optimizer first performs a sensitivity analysis designed to determine the degree to which each of the convolution kernels (channels) at each layer have an impact on the predictions of the network.  Following this, the kernel weights for channels that are to be pruned are zeroed, permitting accuracy evaluation of the "proposed" pruned model.  The remaining weights are then optimized (fine-tuned) for several training epochs in order to recover accuracy.  Multiple iterations of pruning are typically employed, and after each iteration the state can be captured, permitting the developer to backtrack by one or more pruning iterations.  This ability enables the developer to prune for multiple iterations and then select the iteration with the preferred result.  As necessary, pruning can be restarted from a previous iteration with different hyperparameters in order to address accuracy "cliffs" that may present at a specific iteration.

The final phase of pruning, the transform step, removes the channels that were selected for pruning (previously zeroed weights), resulting in a reduction of the number of channels at each pruned layer in the final computational graph.  For instance, a layer that previously required the computation of 128 channels (128 convolution kernels) may now only require the computation of output activations for 87 channels, (ie, 41 channels were pruned).  Following the transform step, the model is now in a form that can be ingested by the Vitis AI Quantizer and deployed on the target.

The below diagram illustrates the high-level pruning workflow:

<div align="center">
  <img width="100%" height="100%" src="reference/images/optimizer_workflow.PNG">
</div>
</br>

###  Channel Pruning

Current Vitis AI DPUs can take advantage of channel pruning in order to greatly reduce the computational cost for inference, often with very little or no loss in prediction accuracy.  In constrast to sparse pruning which requires that the the computation of specific activations within a channel or layer be "skipped" at inference time, channel pruning requires no special hardware to address the problem of these "skipped" computations.

The Vitis AI Optimizer is an optional component of the Vitis AI flow.  In general it is possible to reduce the overall computational cost by a factor of more than 2x, and in some cases by a factor of 10x, with minimal losses in prediction accuracy.  In many cases, there is actually an improvement in prediction accuracy during the first few iterations of pruning.  While the fine-tuning step is in part responsible for this improvement, it is not the only explanation.  Such accuracy improvements will not come as a surprise to developers who are familiar with the concept of overfitting, a phenomena that can occur when a large, deep, network is trained on a dataset that has a limited number of classes.

Many of the pre-trained networks available in the Xilinx [Model Zoo](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/model_zoo) were pruned using this technique.

###  Neural Architecture Search

In addition to channel pruning, a technique coined "Once-for-All" training is supported in Vitis-AI.  The concept of Neural Architecture Search (NAS) is that for any given inference task and dataset, there exist in the potential design space a number of network architectures that are both efficient and which have high prediction scores.  Often, a developer starts with a standard backbone that is familiar to them, such as ResNet50, and trains that network for the best accuracy.  However, there are many cases when a network topology with a much lower computational cost may have offered similar or better performance.  For the developer, the effort to train multiple networks with the same dataset (sometimes going so far as to make this a training hyperparameter) is not an efficient method to select the best network topology.  "Once-for-All" addresses this challenge by employing a single training pass and novel selection techniques.

-----------------------
#### *NAS and AI Optimizer related resources:*
- Sample scripts for channel pruning can be found in [examples](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/examples/Vitis-AI-Optimizer) 
- For additional details on channel pruning leveraging the Vitis AI Optimizer, please refer to [Vitis AI Optimizer User Guide](reference/release_documentation).
- For information on Xilinx NAS / Once-for-All, refer to the Once-for-All (OFA) section in the [Vitis AI Optimizer User Guide](reference/release_documentation)
- Once-for-All examples can be found [here](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/examples/ofa)
- An excellent overview of the advantages of OFA is available on the [Xilinx Developer website](https://www.xilinx.com/developer/articles/advantages-of-using-ofa.html)
-----------------------

## Model Quantization

Deployment of neural networks on Xilinx DPUs is made more efficient through the use of integer quantization to reduce the energy cost, memory footprint, and data path bandwidth required for inference.

Xilinx general-purpose CNN-focused DPUs leverage INT8 (8-bit integer) quantization of a trained network.  In many real-world datasets, the distribution of both the weights and activations at a given layer in the network typically spans a much narrower range than can be represented by a 32-bit floating point number.  It is thus possible to accurately represent the distribution of weights and activations at a given layer as integer values by simply applying a scaling factor.  The impact on prediction accuracy of INT8 quantization is typically low, often less than 1%.  This has been found to be true in many applications in which the input data consists of images and video, point-cloud data, as well as input data from a variety of sampled-data systems, including certain audio and RF applications.

###  Quantization Process

The Vitis AI Quantizer, integrated as a component of either TensorFlow or PyTorch, performs a calibration step in which a subset of the original training data (typically 100-1000 samples, no labels required) are forward propagated through the network in order to analyze the distribution of the activations at each layer.  The weights and activations are then quantized as 8-bit integer values.  This process is referred to as Post-Training Quantization.  Following quantization, the prediction accuracy of the network is re-tested using data from the validation set.  If the accuracy is found to be acceptable, the quantization process is complete.

With certain network topologies, the developer may experience excessive accuracy loss.  In these cases a technique referred to as QAT (Quantization Aware Training) can be used with the source training data to execute several back propagation passes with the objective of optimizing (fine-tuning) the quantized weights.


<div align="center">
  <img width="100%" height="100%" src="reference/images/quant_workflow.PNG">
</div>
</br>

The Vitis AI Quantizer is a component of the Vitis AI toolchain, installed in the VAI Docker, and is also provided as [open-source](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src).

-----------------------
#### *Quantization related resources:*
- For additional details on the Vitis AI Quantizer, please refer to Chapter 3 "Quantizing the Model" in [the Vitis AI User Guide](reference/release_documentation).

- TensorFlow 2.x examples are available as follows:</br>
[TF2 Post-Training Quantization](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/g3doc/guide/quantization/post_training.md)</br>
[TF2 Quantization Aware Training](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/g3doc/guide/quantization/training.md)

- PyTorch examples are available as follows:</br>
[PT Post-Training Quantization](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py)</br>
[PT Quantization Aware Training](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_qat.py)
-----------------------

## Model Compilation

Once the model has been quantized, the Vitis AI Compiler is used to construct an internal computation graph as an intermediate representation (IR).  This internal graph consists of independent control and data flow representations. The compiler then performs multiple optimizations, for example, batch normalization operations are fused with convolution when the convolution operator preceeds the normalization operator.  As the DPU supports multiple dimensions of parallelism, efficient instruction scheduling is key to exploiting the inherent parallelism and potential for data reuse in the graph.  Such optimizations are addressed by the Vitis AI Compiler.

The intermediate representation leveraged by Vitis-AI is "XIR" (Xilinx Intermediate Representation).  The XIR-based compiler takes the quantized TensorFlow or PyTorch model as input. First, the compiler transforms the input model into the XIR format. Most of the variations between different frameworks are eliminated at this stage. The compiler then applies optimizations to the graph, and as necessary will partition it into several subgraphs on the basis of whether the subgraph operators can be executed on the DPU. Architecture-aware optimizations are applied for each subgraph. For the DPU subgraph, the compiler generates the instruction stream.  Finally, the optimized graph is serialized into a compiled .xmodel file.

The compilation process leverages an additional input in the form of a DPU arch.json file.  The purpose of this file is to communicate to the compiler the target architecture and hence, capabilities, of the specific DPU for which the graph will be compiled.  If the correct arch.json file is not used, the compiled model will not run on the target.  The implication of this is also that models that were compiled for a specific target DPU must be recompiled if they are to be deployed on a different DPU architecture.  Runtime errors will occur if the model was not compiled for the correct DPU architecture.

Once you have compiled the .xmodel file, you can leverage [Netron](https://github.com/lutzroeder/netron) to review the final graph structure.  Note that as part of the compilation  process the weights are formatted as INT8, concatenated and shuffled in for efficient execution.  Thus, it is not possible to review the weights post-compilation.

The diagram below illustrates a high-level overview of the Vitis AI Compiler workflow:  

<div align="center">
  <img width="100%" height="100%" src="reference/images/compiler_workflow.PNG">
</div>
</br>

The Vitis AI Compiler is a component of the Vitis AI toolchain, installed in the VAI Docker.  The source code for the compiler is not provided.

-----------------------
#### *Compiler related resources:*
- For more information on Vitis AI Compiler and XIR refer to Chapter 4 in the [Vitis AI User Guide](reference/release_documentation)
- PyXIR, which supports TVM and ONNXRuntime integration is available as [open source](https://github.com/Xilinx/pyxir)
- XIR source code is released as a [component of VART](https://gitenterprise.xilinx.com/quentonh/vitis-ai-staging/tree/master/src/Vitis-AI-Runtime/VART/xir)
-----------------------