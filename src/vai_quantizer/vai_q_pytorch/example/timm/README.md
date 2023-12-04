# TIMM

An example of INT-8 post-training quantization on the timm models using vai_q_pytorch.
    
## Preparation

1. Vai_q_pytorch preparation.

Install vai_q_pytorch by referring to the Chapter ['Quick Start in Docker environment'](./../../README.md#quick-start-in-docker-environment) or the Chapter ['Install from source code'](./../../README.md#install-from-source-code) in README.md of vai_q_pytorch for detailed instructions.

2. Environment preparation.

Install the extra package with the following command:

```
pip install timm
```
3. Dataset preparation.

Follow the instructions in the Chapter ['Requirements'](https://github.com/pytorch/examples/tree/main/imagenet#requirements) of Pytorch example to download ImageNet Dataset into [imagenet-folder with val folder]. Here we use `./dataset/imagenet` as example.

## Quantization

The following steps show how to apply post-training static quantization on a timm model. Here we take `resnet50` as example. For other timm models, `timm.list_models()` returns a complete list of available models in timm and `timm.list_models(pretrained=True)` returns a complete list of pretrained models.

1. Evaluate the FP32 model to verify the accuracy. (Optional)

```
sh script/run_eval.sh resnet50 dataset/imagenet
```

2. Inspect the FP32 model with target hardware before quantization. (Optional)

```
sh script/run_inspect.sh resnet50 dataset/imagenet
```

3. Post-training quantization.

```
sh script/run_ptq.sh resnet50 dataset/imagenet
```

4.  Post-training quantization with fast-finetune to improve the accuracy. (Optional)

```
sh script/run_fast_finetune.sh resnet50 dataset/imagenet
```
## Reference

[PyTorch Image Models](https://github.com/huggingface/pytorch-image-models/blob/v0.9.10/validate.py)