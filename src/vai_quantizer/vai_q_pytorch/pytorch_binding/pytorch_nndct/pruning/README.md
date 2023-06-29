<h1 align="center">Prune Model with Vitis Optimizer PyTorch</h1>

Vitis Optimizer PyTorch provides three methods of model pruning:

- Iterative pruning
- One-step pruning
- Once-for-all (OFA)

Iterative pruning and one-step pruning belong to the coarse-grained pruning category, while the once-for-all technique is an NAS-based approach.

## Table of Contents
- [Installation](#installation)
- [Iterative Pruning](#iterative-pruning)
- [One-step Pruning](#one-step-pruning)
- [Once-for-All(OFA)](#once-for-all-ofa)
- [Supported Operations](#supported-operations)
- [Examples](#examples)
- [Results](#results)

## Installation
We provide two ways to install: [From Source](#from-source) and [Docker Image](#docker-image), respectively.
### From Source
It is recommended to use an [Anaconda](https://www.anaconda.com/distribution/#download-section) environment.

#### Envrionment Setup
Export the environment variable `CUDA_HOME`:

```shell
$ export CUDA_HOME=/usr/local/cuda
```

Or, if you want to install with ROCM support: 

```shell
$ export ROCM_HOME=/opt/rocm
$ export CPPFLAGS=$(hipconfig --cpp_config)
```

#### Install PyTorch
Take the CUDA 11.3 version of torch 1.12 as an example, you can install PyTorch by:

```shell
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
For more details, please see https://pytorch.org/get-started/locally/

#### Install pytorch_nndct

```shell
$ cd nndct && pip install -r requirements.txt
$ cd pytorch_binding && python setup.py install
```

#### Verify the Installation:

If the following command line does not an report error, the installation is done.
```shell
$ python -c "import pytorch_nndct"
```

### Docker Image
Vitis AI provides a Docker environment for the Vitis AI Optimizer. The Docker image encapsulates the required tools and libraries necessary for pruning in these frameworks. To get and run the Docker image, please refer to https://xilinx.github.io/Vitis-AI/3.5/html/docs/install/install.html#leverage-vitis-ai-containers.

The prebuilt ROCm docker image can be get by:
```shell
$ docker pull xilinx/vitis-ai-opt-pytorch-rocm:latest
```

For CUDA docker image, there is no prebuilt one and you have to build it yourself.
You can read [this](https://xilinx.github.io/Vitis-AI/3.5/html/docs/install/install.html#option-2-build-the-docker-container-from-xilinx-recipes) for detailed instructions. 

## Iterative Pruning
The method includes two stages: model analysis and pruned model generation.

### Create a Baseline Model

For simplicity, ResNet18 from torchvision is used here. In real life applications, the process of creating a model can be complicated.

```python
from torchvision.models.resnet import resnet18
model = resnet18(pretrained=True)
```

### Create a Pruning Runner

Import modules and prepare dummy inputs:

```python
from pytorch_nndct import IterativePruningRunner

# Inputs should have the same shape and dtype as the model input.
inputs = torch.randn([1, 3, 224, 224], dtype=torch.float32)
```
Create an iterative pruning runner:

```python
runner = IterativePruningRunner(model, inputs)
```

### Running Model Analysis
Define an evaluation function. The function must take a model as its first argument and return a score.

```python
def eval_fn(model, dataloader):
  top1 = AverageMeter('Acc@1', ':6.2f')
  model.eval()
  with torch.no_grad():
  for i, (images, targets) in enumerate(dataloader):
    images = images.cuda()
    targets = targets.cuda()
    outputs = model(images)
    acc1, _ = accuracy(outputs, targets, topk=(1, 5))
    top1.update(acc1[0], images.size(0))
  return top1.
  
runner.ana(eval_fn, args=(val_loader,))
```

For the same model, the model analysis only needs to be done once. You can prune the model iteratively without re-running analysis because there is only one pruned model generated for a specific pruning ratio.

After the model analysis is completed, the analysis result is saved in the file named `.vai/your_model_name.sens`. You can prune a model iteratively using this file. In other words, prune the model to the target ratio gradually to avoid the failure to improve the model performance in the retraining stage that is caused by setting the pruning ratio too high.

### Getting the Pruned Model

```python
model = runner.prune(removal_ratio=0.2)
```

### Retraining the Pruned Model

Retraining the pruned model is the same as training a baseline model:

```python
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
best_acc1 = 0 

for epoch in range(args.epoches):
  train(train_loader, model, criterion, optimizer, epoch)
  acc1, acc5 = evaluate(val_loader, model, criterion)

  is_best = acc1 > best_acc1
  best_acc1 = max(acc1, best_acc1)
  if is_best:
    torch.save(model.state_dict(), 'model_pruned.pth')
    # Sparse model has one more special method in iterative pruning.
    torch.save(model.slim_state_dict(), 'model_slim.pth')
```

## One-step Pruning

The method also includes two stages: adaptive batch normalization (BN) based searching for pruning strategy, and pruned model generation.

### Prepare a Baseline Model and Pruning Runner
```python
from torchvision.models.resnet import resnet18
from pytorch_nndct import OneStepPruningRunner

model = resnet18(pretrained=True)
# Inputs should have the same shape and dtype as the model input.
inputs = torch.randn([1, 3, 224, 224], dtype=torch.float32)
runner = IterativePruningRunner(model, inputs)
```

### Running Sub-network Searching

`num_subnet` specifies the number of candidate subnetworks satisfying the sparsity requirement to be searched.
The best subnetwork can be selected from these candidates. The higher the value, the longer it takes to search, but the higher the probability of finding a better subnetwork.

```python
# Adaptive-BN-based searching for pruning strategy. 'calibration_fn' is a function for calibrating BN layer's statistics.
runner.search(
    gpus=['0'],
	calibration_fn=calibration_fn, calib_args=(val_loader,),
	eval_fn=eval_fn, eval_args=(val_loader,),
	num_subnet=1000, removal_ratio=0.7)
```

The `eval_fn` is the same with the iterative pruning method. A `calibration_fn` function that implements adaptive-BN is shown in the following example code. It is recommended to define your code in a similar way.

```python
def calibration_fn(model, dataloader, number_forward=100):
  model.train()
  with torch.no_grad():
    for index, (images, target) in enumerate(dataloader):
      images = images.cuda()
      model(images)
    if index > number_forward:
      break
```

After searching, a file named `.vai/your_model_name.search` is generated in which the search result (pruning strategies and corresponding evaluation scores) is stored. You can get the final pruned model in one-step.

### Getting the Pruned Model
```python
model = runner.prune(removal_ratio=0.7)
```

### Retraining the Pruned Model

Retraining the pruned model is the same as training a baseline model:

```python
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
best_acc1 = 0 

for epoch in range(args.epoches):
  train(train_loader, model, criterion, optimizer, epoch)
  acc1, acc5 = evaluate(val_loader, model, criterion)

  is_best = acc1 > best_acc1
  best_acc1 = max(acc1, best_acc1)
  if is_best:
    torch.save(model.state_dict(), 'model_pruned.pth')
```

The one-step pruning method has several advantages over the iterative approach:

- The generated pruned models are more accurate. All subnetworks that meet the requirements are evaluated.
- The workflow is simpler because you can obtain the final pruned model in one step without iterations.
- Retraining a slim model is faster than a sparse model.

There are two disadvantages to one-step pruning: one is that the random generation of pruning strategy is unstable. The other is that the subnetwork searching must be performed once for every pruning ratio.


## Once-for-All (OFA)

Steps for the once-for-all method are as follows:

### Creating a Model

For simplicity, mobilenet_v2 from torchvision is used here.

```python
from torchvision.models.mobilenet import mobilenet_v2
model = mobilenet_v2(pretrained=True)
```

### Creating an OFA Pruner

The pruner requires two arguments:

- The model to be pruned
- The inputs needed by the model for inference

```python
import torch
from pytorch_nndct import OFAPruner

inputs = torch.randn([1, 3, 224, 224], dtype=torch.float32)
pruner = OFAPruner(model, inputs)
```
**Note:** The input does not need to be real data. You can use randomly generated dummy data if it has the same shape and type as the real data.

### Generating an OFA Model

Call `ofa_model()` to get an OFA model. This method finds all the `nn.Conv2d` / `nn.ConvTranspose2d`and `nn.BatchNorm2d` modules,
then replaces those modules with `DynamicConv2d` / `DynamicConvTranspose2d` and `DynamicBatchNorm2d`.

A list of pruning ratios is required to specify what the OFA model will be.

For each convolution layer in the OFA model, an arbitrary pruning ratio can be used in the output channel. The maximum and minimum values in this list represent the maximum and minimum compression rates of the model. Other values in the list represent the subnetworks to be optimized. By default, the pruning ratio is set to [0.5, 0.75, 1].

For a subnetwork sampled from the OFA model, the out channels of a convolution layer are one of the numbers in the pruning ratio list multiplied by its original number. For example, for a pruning ratio list of [0.5, 0.75, 1] and a convolution layer nn.Conv2d(16, 32, 5), the out channels of this layer in a sampled subnetwork are [0.5*32, 0.75*32, 1*32].

Because the first and last layers have a significant impact on network performance, they are commonly excluded from pruning.
By default, this method automatically identifies the first convolution and the last convolution, then puts them into the list of excludes. Setting `auto_add_excludes=False` can cancel this feature.

```python
ofa_model = ofa_pruner.ofa_model([0.5, 0.75, 1], excludes = None, auto_add_excludes=True)
```
### Training an OFA Model
This method uses the [sandwich rule](https://arxiv.org/abs/2003.11142) to jointly optimize all the OFA subnetworks. The `sample_random_subnet()` function can be used to get a subnetwork. The dynamic subnetwork can do a forward/backward pass.

In each training step, given a mini-batch of data, the sandwich rule samples a ‘max’ subnetwork, a ‘min’ subnetwork, and two random subnetworks. Each subnetwork does a separate forward/backward pass with the given data and then all the subnetworks update their parameters together.

```python
# using sandwich rule and sampling subnet.
for i, (images, target) in enumerate(train_loader):

  images = images.cuda(non_blocking=True)
  target = target.cuda(non_blocking=True)

  # total subnets to be sampled
  optimizer.zero_grad()

  teacher_model.train()
  with torch.no_grad():
    soft_logits = teacher_model(images).detach()

  for arch_id in range(4):
    if arch_id == 0:
      model, _ = ofa_pruner.sample_subnet(ofa_model,'max')
    elif arch_id == 1:
      model, _ = ofa_pruner.sample_subnet(ofa_model,'min')
    else:
      model, _ = ofa_pruner.sample_subnet(ofa_model,'random') 

    output = model(images)

    loss = kd_loss(output, soft_logits) + cross_entropy_loss(output, target) 
    loss.backward()

  torch.nn.utils.clip_grad_value_(ofa_model.parameters(), 1.0)
  optimizer.step()
  lr_scheduler.step()
```
### Searching Subnetworks under Constraints

After the training is completed, you can conduct an [evolutionary search](https://arxiv.org/abs/1802.01548) based on the neural-network-twins to get a subnetwork with the best trade-offs between FLOPs and accuracy using a minimum and maximum FLOPs range.

```python
pareto_global = ofa_pruner.run_evolutionary_search(ofa_model, calibration_fn,
    (train_loader,) eval_fn, (val_loader,), 'acc1', 'max', min_flops=230, max_flops=250)
ofa_pruner.save_subnet_config(pareto_global, 'pareto_global.txt')
```
The searching result looks like the following:

```
{ 
"230": { 
    "net_id": "net_evo_0_crossover_0", 
    "mode": "evaluate",
    "acc1": 69.04999542236328,
    "flops": 228.356192,
    "params": 3.096728,
    "subnet_setting": [...]
}
"240": {
    "net_id": "net_evo_0_mutate_1",
    "mode": "evaluate",
    "acc1": 69.22000122070312,
    "flops": 243.804128,
    "params": 3.114,
    "subnet_setting": [...]
}}
```
### Getting a Subnetwork

Call `get_static_subnet()` to get a specific subnetwork. The `static_subnet` can be used for finetuning and doing quantization.

```python
pareto_global = ofa_pruner.load_subnet_config('pareto_global.txt')
static_subnet, static_subnet_config, flops, params = ofa_pruner.get_static_subnet(
    ofa_model, pareto_global['240']['subnet_setting'])
```

## Supported Operations

Operation             |Iterative|One-step|OFA
:--------------------:|:-------:|:------:|:--:
|torch.nn.Conv2d|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|torch.nn.Conv3d|:heavy_check_mark:|:heavy_check_mark:|:x:
|torch.nn.ConvTranspose2d|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|torch.nn.ConvTranspose3d|:heavy_check_mark:|:heavy_check_mark:|:x:
|torch.nn.BatchNorm2d|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:
|torch.nn.BatchNorm3d|:heavy_check_mark:|:heavy_check_mark:|:x:

## Examples
Please find examples [here](/example/pruning/torch)

## Results

The pruned models are named in `model_name-method-pruning_ratio`. For example, `ResNet50-Iterative-0.75` means a pruned ResNet50 by iterative pruning and the pruning ratio is 0.75 (the FLOPs is 0.75x of the baseline)

<pre>
===============================================================================
Model                            FLOPs                  Accuray      
===============================================================================             
ResNet50                         4G                     76.01/92.93(top1/top5)
-------------------------------------------------------------------------------
ResNet50-Iterative-0.75          3.06G                  76.06/92.67
ResNet50-OneStep-0.75            3.09G                  76.40/92.84
ResNet50-OFA-0.77                3.15G                  77.13/93.47
ResNet50-Iterative-0.5           2.06G                  74.42/92.67
ResNet50-OneStep-0.5             2.07G                  76.01/92.84
ResNet50-OFA-0.54                2.23G                  76.73/93.29
_______________________________________________________________________________
InceptionV3                      5.71G                  77.44/93.62(top1/top5)
-------------------------------------------------------------------------------
InceptionV3-Iterative-0.65       3.15G                  76.74/92.89
InceptionV3-OneStep-0.65         3.12G                  77.06/93.21
_______________________________________________________________________________
MobileNetV2                      300.77M                71.85/90.33(top1/top5)
-------------------------------------------------------------------------------
MobileNetV2-Iterative-0.75       226.42M                70.20/89.48
MobileNetV2-OneStep-0.76         228.47M                69.84/89.21
MobileNetV2-OFA-0.74             222.46M                70.82/89.77
_______________________________________________________________________________
YoloV5-m                         24.44G                 43.4%(mAP)
-------------------------------------------------------------------------------
YoloV5-m-Iterative-0.8           17.09G                 39.0%
yolov5-m-OFA-0.8                 17.10G                 42.0%
_______________________________________________________________________________
</pre>

