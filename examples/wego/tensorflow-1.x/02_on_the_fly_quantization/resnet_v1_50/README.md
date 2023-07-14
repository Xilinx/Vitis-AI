# Setup Conda Environment for WeGO-TensorFlow1.x

Suppose you have entered the Vitis-AI CPU docker container, then using following command to activate the conda env for WeGO TensorFlow-1.x.

```bash
$ conda activate vitis-ai-tensorflow
```

# Preparation

## Prepare float model and dataset

This example will use Resnet_v1_50 model and please first download the float model for it as follows.
```
$ cd /workspace/examples/wego/tensorflow-1.x/02_on_the_fly_quantization/resnet_v1_50
$ wget https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_50_imagenet_224_224_0.38_4.3G_3.0.zip -O tf_resnetv1_50_imagenet_224_224_0.38_4.3G_3.0.zip
$ unzip tf_resnetv1_50_imagenet_224_224_0.38_4.3G_3.0.zip
```

Download the ImageNet dataset and make sure there are valid images in the following path. Otherwise, you can also modify the `CALIB_IMAGE_DIR` variable in the `input_fn.py` file to point to your own path.
```
/scratch/data/Imagenet/val_dataset
```

# Run Resnet_v1_50 quantize and compile flow

Execute the following command, quantize_eval_model.pb is a temporary quantized model.

```bash
$ bash quantize_compile_run.sh
```

# Run Resnet_v1_50 serialization and deserialization flow

Execute the following command, resnet_v1_50_wego.pb is the serialized WeGO TF1 model. For this flow, the above `Preparation` steps are not necessary.

```bash
$ bash compile_serialize_run.sh
```

# License

Copyright 2022 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
