# Table of Contents
- [Pruning](#pruning)
  - [Fine-grained pruning](#fine-grained-pruning)
  - [Coarse-grained Pruning](#coarse-grained-pruning)
    - [Comparing Iterative Pruning and One-Step Pruning](#comparing-iterative-pruning-and-one-step-pruning)
	- [Iterative Pruning](#iterative-pruning)
	  - [Guidelines for Better Pruning Results](#guidelines-for-better-pruning-results)
	- [One-step Pruning](#one-step-pruning)
  - [Neural Architecture Search](#neural-architecture-search)
    - [Once-for-All (OFA)](#once-for-all-ofa)
- [Vitis AI Optimizer](#vitis-ai-optimizer)
  - [Overview](#overview)
  - [Optimizer PyTorch](#optimizer-pytorch)
  - [Optimizer TensorFlow](#optimizer-tensorflow)
  - [Optimizer TensorFlow2](#optimizer-tensorflow2)

# Pruning

Neural networks are typically over-parameterized with significant redundancy. Pruning is the process of eliminating redundant weights while keeping the accuracy loss as low as possible. Industry research has led to several techniques that serve to reduce the computational cost of neural networks for inference. These techniques include:

- Fine-grained pruning
- Coarse-grained pruning
- Neural Architecture Search (NAS)

## Fine-grained pruning

With fine-grained pruning, weights which have minimal effect on the output can be set to zero such that the corresponding computations would be skipped or otherwise removed from the inference graph. This results in sparse matrices (i.e., matrices which have many zero elements). Fine-grained pruning can achieve very high compression rates with very low accuracy loss. However, a hardware accelerator capable of implementing fine-grained sparsity must either be a fully customized, pipelined implementation, or a more general purpose “Matrix of Processing Engines” type of accelerator with the addition of specialized hardware and techniques for weight skipping and compression.

The Vitis AI sparsity pruner implements a fine-grained sparse pruning algorithm for multiple N:M sparsity patterns in each contiguous block of M values. N values must be zero. On the dense network, weights or activations would be pruned to satisfy the N:M structured sparsity criterion. Pruning is performed along input channel dimensions. Out of every M elements, the smallest N elements would be set to zero. Typical values for M would be 4/8/16, while N would be set to half the value of M to achieve 50% fine-grained sparsity. This naturally leads to a sparsity of 50%, which is fine-grained. The Vitis AI sparsity pruner supports sparsity of weights and activations for convolution and fully connected layers. The sparsity of activations can be 0 or 0.5. When the sparsity of activations is 0, the sparsity of weights can be 0, 0.5, or 0.75. When the sparsity of activations is 0.5, the sparsity of weights (percentage of zero weights in one block (M)) can only be 0.75.

The sparsity pruning steps are as follows:
1. Generate the sparse model
2. Fine-tune the sparse model
3. Export the sparse model

## Coarse-grained Pruning

In coarse-grained pruning, also known as channel pruning, the objective is to prune channels instead of individual weights from the graph. The result is a computational graph in which one or more convolution kernels need not be computed for a given layer. For instance, a convolution layer with 128 channels prior to pruning can require the computation of only 57 channels post-pruning.

Channel pruning is very friendly to hardware acceleration and can be applied to virtually any inference architecture. However, coarse-grained pruning is limited in terms of the overall computational cost reduction (pruning ratio) that can be achieved.

Coarse-grained pruning always reduces the accuracy of the original model. Retraining (fine-tuning) adjusts the remaining weights to recover accuracy. The technique works well on large models with common convolutions, for example, ResNet and VGGNet. However, with depthwise convolution models such as MobileNet-v2, the accuracy of the pruned model drops dramatically even at a small pruning rate.

### Comparing Iterative Pruning and One-Step Pruning

<table>
  <tr>
	<th>Criteria</th>
	<th>Iterative Pruning</th>
    <th>One-step Pruning</th>
  </tr>
  <tr>
    <th scope="row">Prerequisites</th>
    <td>-</td>
    <td>BatchNormalization in network</td>
  </tr>
  <tr>
    <th scope="row">Time taken</th>
    <td>More than one-step pruning</td>
    <td>Less than iterative pruning</td>
  </tr>
  <tr>
    <th scope="row">Retraining requirement</th>
    <td>Required</td>
    <td>Required</td>
  </tr>
  <tr>
    <th scope="row">Code organization</th>
    <td>Evaluation function</td>
    <td>Evaluation function<br>Calibration function</td>
  </tr>
</table>

### Iterative Pruning

The pruner is designed to reduce the number of model parameters while minimizing the accuracy loss. This is done iteratively as shown in the following figure. Pruning results in accuracy loss while retraining recovers accuracy. Pruning, followed by retraining, forms one iteration. In the first iteration of pruning, the input model is the baseline model, and it is pruned and fine-tuned. In subsequent iterations, the fine-tuned model obtained from the previous iteration becomes the new baseline. This process is usually repeated several times until a desired sparse model is obtained. The iterative approach is required because a model cannot be pruned in a single pass while maintaining accuracy. When too many parameters are removed in one iteration, the accuracy loss may become too steep and recovery may not be possible.
Leveraging the process of iterative pruning, higher pruning rates can be achieved without any significant loss of model performance

#### Guidelines for Better Pruning Results
The following is a list of suggestions to optimize pruning results. Following these guidelines has been found to help developers achieve higher pruning ratios and reduce accuracy loss.

- Use as much data as possible to perform model analysis. Ideally, you should use all the data in the validation dataset, but this can be time consuming. You can also use partial validation set data, but you need to make sure at least half of the dataset is used.
- During the fine-tuning stage, experiment with a few hyperparameters, including the initial learning rate and the learning rate decay policy. Use the best result as the input for the next iteration.
- The data used in fine-tuning should be a subset of the original dataset used to train the baseline model.
- If the accuracy does not improve sufficiently after conducting several fine-tuning experiments, try reducing the pruning rate parameter and then re-run pruning and fine-tuning.

### One-step Pruning
One-step pruning implements the EagleEye algorithm. It introduces a strong positive correlation between different pruned models and their corresponding fine-tuned accuracy by a simple, yet efficient, evaluation component called adaptive batch normalization. It enables you to get the subnetwork with the highest potential accuracy without actually fine-tuning the models. In short, the one-step pruning method searches for a bunch of subnetworks (i.e., generated pruned models) that meet the required model size, and selects the most promising one. The selected subnetwork is then retrained to recover the accuracy.

The pruning steps are as follows:

1. Search for subnetworks that meet the required pruning ratio.
2. Select a potential network from a bunch of subnetworks with an evaluation component.
3. Fine-tune the pruned model.

## Neural Architecture Search
The concept of Neural Architecture Search (NAS) is that for any given inference task and dataset, there exist in the potential design space several network architectures that are both efficient and which have high prediction scores. Often, a developer starts with a standard backbone that is familiar to them, such as ResNet50, and trains that network for the best accuracy. However, there are many cases when a network topology with a much lower computational cost may have offered similar or better performance. For the developer, the effort to train multiple networks with the same dataset (sometimes going so far as to make this a training hyperparameter) is not an efficient method to select the best network topology.

NAS can be flexibly applied for each layer. The number of channels and amount of sparsity is learned by minimizing the loss of the pruned network. NAS achieves a good balance between speed and accuracy, but requires extended training times. This method requires a four-step process:

1. Train
2. Search
3. Prune
4. Fine-tune (optional)
Compared with coarse-grained pruning, one-shot NAS implementations assemble multiple candidate "subnetworks" into a single, over-parameterized graph known as a Supernet. The training optimization algorithm attempts to optimize all candidate networks simultaneously using supervised learning. Upon the completion of this training process, candidate subnetworks are ranked based on computational cost and accuracy. The developer selects the best candidate to meet their requirements. The one-shot NAS method is effective in compressing models that implement both depthwise convolutions and conventional convolutions but requires a long training time and a higher level of skill on the part of the developer.

### Once-for-All (OFA)
Once-For-All (OFA) is a compression scheme based on One-Shot NAS. You often perform tasks such as compressing a trained model and deploying it on one or more devices. Conventional schemes require you to repeat the network design process and retrain the designed network from scratch for each device, which is computationally prohibitive.

OFA introduces a new solution to tackle this challenge: designing a once-for-all network that can be directly deployed under diverse architectural configurations. Therefore, training cost can be amortized. Inference is performed by selecting only a part of the once-for-all network.

OFA reduces the model size across many more dimensions than pruning. It builds a family of models of varying depth, width, kernel size, and image resolution, jointly optimizing all the candidate models. Once trained, the subnetwork with the best balance between accuracy and throughput can be discovered by evolutionary search.

For each layer in the original model, Vitis OFA allows you to use an arbitrary pruning ratio for channels and arbitrary kernel sizes. The original model is split into many child networks with shared weights and becomes a super network. The child networks can do forward passes and update using a part of the convolution weights. All subnetworks should be jointly optimized during training. When all child networks are trained well, you can search for the subnetwork with the optimum balance between accuracy and throughput from the supernetwork.

To get the highest compression, the Vitis OFA pruner can optimize the search space based on the ratio of the number of depthwise convolutions to the number of regular convolutions. If the number of depthwise convolutions is more than the number of regular convolutions, the Vitis OFA pruner mainly compresses convolution layers where the kernel size is > 1. This would result in a supernetwork with narrow channel widths and reduced accuracy.

OFA uses the original model as a teacher model to guide the training of subnetworks. Knowledge distillation allows the output of teacher models to be used as softened labels to provide more information about intraclasses and interclasses. The Vitis OFA uses adaptive soft KDLoss and the sandwich rule to improve performance and efficiency. Compared with the original OFA, the Vitis OFA reduces the training time by half.

# Vitis AI Optimizer

## Overview

Inference in machine learning is computationally intensive and requires high memory bandwidth to meet the low-latency and high-throughput requirements of various applications. Vitis Optimizer provides the ability to prune neural network models. It prunes redundant kernels in neural networks thereby reducing the overall computational cost for inference. The pruned models produced by Vitis Optimizer are then quantized by Vitis Quantizer to be further optimized.

The following tables show the features that are supported by Vitis Optimizer for different frameworks:

<table>
  <colgroup span="3"></colgroup>
  <colgroup span="3"></colgroup>
  <tr>
	<th rowspan="2">Framework</th>
	<th rowspan="2">Versions</th>
    <th colspan="3" scope="colgroup">Features</th>
  </tr>
  <tr>
    <th scope="col">Iterative</th>
    <th scope="col">One-step</th>
	<th scope="col">OFA</th>
  </tr>
  <tr>
    <th scope="row">PyTorch</th>
    <td>Supports 1.4 - 1.13</td>
    <td>Yes</td>
    <td>Yes</td>
    <td>Yes</td>
  </tr>
  <tr>
    <th scope="row">TensorFlow</th>
    <td>Supports 2.4 - 2.12</td>
    <td>Yes</td>
    <td>No</td>
    <td>No</td>
  </tr>
</table>

## Optimizer PyTorch

Please see [Optimizer PyTorch](pytorch_binding/pytorch_nndct/pruning/README.md)

## Optimizer TensorFlow

Please see [Optimizer TensorFlow](tensorflow_v1/README.md)

## Optimizer TensorFlow2

Please see [Optimizer TensorFlow2](tensorflow/README.md)
