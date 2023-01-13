# Setup Conda Environment for WeGO-Torch

Suppose you have entered the Vitis-AI CPU docker container, then using following command to activate the conda env for WeGO-Torch.

```bash
$ conda activate vitis-ai-wego-torch
```

# Preparation

## Install the python dependencies

```
$ pip install -r requirements.txt
```

## Prepare dataset for calibration

If you need to quantize the model with Post Training Quantization(PTQ), a small unlabeled dataset will be needed(100-1000 images). You can use the ImageNet dataset or your own dataset.

To use ImageNet, first download it from http://www.image-net.org/. For calibration purpose, only the validation set is needed.

After downloading prepare the dataset with [the following shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).

The directory structure should look like this:

```bash
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
```

We'll only be using the validation set here(imagenet/val/).

## Get pretained inception v3 model

Download it from pytorch website:

```bash
wget https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
```

# Quantize, compile, serialize and run

use quantize_compile_serialize_run.sh to quantize, compile, optionally serialize, and run the model.

Usage:

```bash
$ bash quantize_compile_serialize_run.sh path_to_pretrained_float_model path_to_imagenet_val_folder
```

For example:

```bash
$ bash quantize_compile_serialize_run.sh ./inception_v3_google-0cc3c7bd.pth imagenet/val/
```

The serialized WeGO module will be saved in current working directory with name inception_v3_wego_compiled.wg

# Deserialize a previously compiled WeGO module and run it

use deserialize_run.sh to deserialize a previously compiled WeGO module and run it.

```bash
$ bash deserialize_run.sh ./inception_v3_wego_compiled.wg
```

# License

Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
