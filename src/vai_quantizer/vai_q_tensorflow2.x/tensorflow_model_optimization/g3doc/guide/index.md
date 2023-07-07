# TensorFlow model optimization

The *TensorFlow Model Optimization Toolkit* minimizes the complexity of
optimizing machine learning inference.

Inference efficiency is a critical concern when deploying machine learning
models because of latency, memory utilization, and in many cases power
consumption. Particularly on edge devices, such as mobile and Internet of Things
(IoT), resources are further constrained, and model size and efficiency of
computation become a major concern.

Computational demand for *training* grows with the number of models trained on
different architectures, whereas the computational demand for *inference* grows
in proportion to the number of users.

## Use cases

Model optimization is useful, among other things, for:

*   Reducing latency and cost for inference for both cloud and edge devices
    (e.g. mobile, IoT).
*   Deploying models on edge devices with restrictions on processing, memory
    and/or power-consumption.
*   Reducing payload size for over-the-air model updates.
*   Enabling execution on hardware restricted-to or optimized-for fixed-point
    operations.
*   Optimizing models for special purpose hardware accelerators.

## Optimization techniques

The area of model optimization can involve various techniques:

*   Reduce parameter count with pruning and structured pruning.
*   Reduce representational precision with quantization.
*   Update the original model topology to a more efficient one with reduced
    parameters or faster execution. For example, tensor decomposition methods
    and distillation

Our toolkit supports
[post-training quantization](./quantization/post_training.md),
[quantization aware training](./quantization/training.md), and
[pruning](./pruning/index.md).

### Quantization

Quantized models are those where we represent the models with lower precision,
such as 8-bit integers as opposed to 32-bit float. Lower precision is a
requirement to leverage certain hardware.

### Sparsity and pruning

Sparse models are those where connections in between operators (i.e. neural
network layers) have been pruned, introducing zeros to the parameter tensors.
