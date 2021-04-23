# Compiling a Model


## Overview

 The TVM with Vitis AI flow contains two stages: Compilation and Execution. During the compilation a user can choose a model to compile for the cloud or edge target devices that are currently supported. Once a model is compiled, the generated files can be used to run the model on a the specified target device during the Execution stage. Currently, the TVM with Vitis AI flow supported a selected number of Xilinx data center and edge devices.

This document provides instructions to compile deep learning models using the TVM with Vitis AI support. We further walk through one of the tutorial scripts provided in the examples directory.

Before continuing with this document, please ensure you have properly installed and setup your environment the TVM with Vitis AI, as described [here](../README.md).

### Resources

If you are not familiar with Apache TVM, the following materials are provided as a guideline to understand the TVM framework. You can find more information on the Apache TVM website.

* [Apache TVM Tutorials]
* [Compiling Deep Learning Models] using TVM


### Compilation Examples

The examples directory incorporates example python scripts for compiling models using the TVM with Vitis flow. Copy the examples directory to the docker container and run any of the compile script after setting the conda environment to the "vitis-ai-tensorflow".

```sh
# In docker
$ conda activate vitis-ai-tensorflow
# !! For DPUCADX8G source the XRT setup script: 
#   $ source /opt/xilinx/xrt/setup.sh

# Execute examples in /workspace/examples. (Note that this assumes that the Vitis-AI/external/tvm directory has been mounted on /workspace inside the docker container, this will be the case if you executed Vitis-AI/docker_run.sh from inside Vitis-AI/external/tvm: ../../docker_run.sh tvm.ci_vai_1x)
$ cd /workspace/examples
# DPU_TARGET options: 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102', 'DPUCZDX8G-som', 'DPUCZDX8G-ultra96'
$ python3 compile_mxnet_resnet_18.py "DPU_TARGET"
```

The compilation output is saved on disk (as a .so) to run the model on the target device during the Execution stage. For edge devices, the compilation output needs to be transfered over to the board.

### Compiling MXNet ResNet 18

In this section we walk through the mxnet_resent_18.py tutorial script to further demonstrate the Compilation stage of the TVM with Vitis AI support. The script demonstrates how to import, quantize and compile models using this flow.

#### Declare the target

```python
tvm_target = 'llvm'
dpu_target = 'DPUCADX8G' # options: 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'
```

The TVM with Vitis AI flow currently supports 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'. AS mentioned previously, the "DPUCADX8G" computation engine targets cloud devices and "DPUCZDX8G-*" targets edge devices. Once the approrpiate targets are defined, we invoke the TVM compiler to build the graph for the specified target:

#### Import the Model

The TVM with Vitis AI support provides ease of use by mimicking the flow of that by the TVM. As such, we leverage the front end capabilities of the TVM framework for importing models. The TVM tutorial [Compiling MXNet Models] document provides an example to import MXNet models and compile them using only the TVM compiler. The TVM documentation also provides tutorials to import models from different supported framework [here].

```python
mod, params = relay.frontend.from_mxnet(block, input_shape)
```
To be able to target the Vitis-AI cloud DPUCADX8G target we first have to import the target in PyXIR. This PyXIR package is the interface being used by TVM to integrate with the Vitis-AI stack. Additionaly, import the typical TVM and Relay modules and the Vitis-AI contrib module inside TVM.


```python
import pyxir
# Only needed for DPUCADX8G
import pyxir.contrib.target.DPUCADX8G

import tvm
import tvm.relay as relay
from tvm.relay import transform
from tvm.contrib.target import vitis_ai
from tvm.contrib import util, graph_executor
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai
```

#### Partition the Model

After importing the model, we utilize the Relay API to annotate the Relay expression for the provided DPU target and partition the graph.

```python
# Make sure parameters become constants in the model and remove unused functions
mod["main"] = bind_params_by_name(mod["main"], params)
mod = transform.RemoveUnusedFunctions()(mod)

# For edge DPU we recommend converting the convolutions' data layout
#    to NHWC for best performance. Therefore, we first convert the layouts
#    of all convolutions to NHWC before partitioning. Afterwards, we can
#    convert any remaining convolutions (to be executed on CPU) back to NCHW.
if dpu_target.startswith('DPUCZDX8G'):
    desired_layouts = {'nn.conv2d': ['NHWC', 'OIHW']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts),
                                    relay.transform.FoldConstant()])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

mod = partition_for_vitis_ai(mod, params, dpu=dpu_target)

# For edge DPU, we recommend transforming the remaining convolutions after
#    partitioning (that will be executed on CPU, if any) back to NCHW data layout
#    for best CPU performance
if dpu_target.startswith('DPUCZDX8G'):
    desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts),
                                    relay.transform.FoldConstant()])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
````


#### Build the Model

The partitioned model is passed to the TVM compiler to generate the runtime libraries for the TVM Runtime.

```python
export_rt_mod_file = os.path.join(os.getcwd(), 'vitis_ai.rtmod')
build_options = {
    'dpu': dpu_target,
    'export_runtime_module': export_rt_mod_file
}
with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):
    lib = relay.build(mod, tvm_target, params=params)
```


#### Quantize the Model

As part of its compilation process, TVM with Vitis AI support automatically performs quantization for the target hardware. To do so, we make use of On-The-Fly (OTF) Quantization. Using this method one doesn’t need to quantize their model upfront; They can make use of the typical inference execution calls (module.run) to calibrate the model on-the-fly using the first N inputs that are provided. After the first N iterations, computations will be accelerated on the DPU. So, now we will feed N inputs to the TVM runtime module. Note that these first N inputs are executed on CPU and used for quantization and therefore will take a substantial amount of time.

We need a set of images for quantization and a callback function that can perform the model preprocessing on the quantization images and returns a dictionary mapping from input name to array containing dataset inputs. In this example we currently use the imagenet dataset images for quantization, but the user can plug in a different dataset of their choice.

```python
TVM_VAI_HOME = os.getenv('TVM_VAI_HOME')
quant_dir = os.path.join(TVM_VAI_HOME, 'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min')

# callback function
def inputs_func(img_files: List[str]):
    inputs = []
    for img_path in img_files:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((224,224))
       
        inputs.append(transform_image(img))
    return inputs

print("Create InferenceSession for OTF Quantization")
InferenceSession = graph_executor.GraphModule(lib["default"](tvm.cpu()))

px_quant_size = int(os.environ['PX_QUANT_SIZE']) \
    if 'PX_QUANT_SIZE' in os.environ else 128
...
for i in range(px_quant_size):
    InferenceSession.set_input(input_name, quant_images[i]) 
    InferenceSession.run()

```
By default, the number of images used for quantization is set to 128. you could change the OTF Quantization behavior using the environment variables below:

| Variable  | Default  | Description | 
|---|---|---|
| PX_QUANT_SIZE   | 128  | The number of inputs that will be used for quantization (necessary for Vitis-AI acceleration)  |

For example, execute the following line in the terminal before calling the compilation script to reduce the quantization calibration dataset to eight images. This can be used for quick testing.

```
$ export PX_QUANT_SIZE=8
```

Lastly, we store the compiled output from the TVM compiler on disk for running the model on the target device. This happens as follows for cloud DPU's (for example DPUCADX8G):

```python
lib_path = "deploy_lib.so"
lib.export_library(lib_path)
```

For DPUZDX8G targets we have to rebuild for aarch64. To do this we first have to normally export the module to also serialize the Vitis AI runtime module (`vitis_ai.rtmod`). We will load this runtime module again afterwards to rebuild and export for aarch64.

```python
temp = utils.tempdir()
lib.export_library(temp.relpath("tvm_lib.so"))

# Build and export lib for aarch64 target
tvm_target = tvm.target.arm_cpu('ultra96')
lib_kwargs = {
   'fcompile': contrib.cc.create_shared,
   'cc': "/usr/aarch64-linux-gnu/bin/ld"
}

build_options = {
    'load_runtime_module': export_rt_mod_file
}
with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):
     lib_dpuczdx8g = relay.build(mod, tvm_target, params=params)

lib_dpuczdx8g.export_library('tvm_dpu_cpu.so', **lib_kwargs)
```

## Next steps

This concludes the tutorial to compilation a model using TVM with Vitis AI. For instructions to run a compiled model please refer to the [running_on_zynq](./running_on_zynq.md) and [running_on_alveo](./running_on_alveo.md) documents

Checkout how other model types can be compiled through TVM and adjust the "compile_mxnet_resnet_18.py" script accordingly:
* [Compile Tensorflow models](https://tvm.apache.org/docs/tutorials/frontend/from_tensorflow.html)
* [Compile PyTorch models](https://tvm.apache.org/docs/tutorials/frontend/from_pytorch.html)
* [Compile ONNX models](https://tvm.apache.org/docs/tutorials/frontend/from_onnx.html)
* [Compile Keras models](https://tvm.apache.org/docs/tutorials/frontend/from_keras.html)
* [Compile TFLite models](https://tvm.apache.org/docs/tutorials/frontend/from_tflite.html)
* [Compile CoreML models](https://tvm.apache.org/docs/tutorials/frontend/from_coreml.html)
* [Compile DarkNet models](https://tvm.apache.org/docs/tutorials/frontend/from_darknet.html)
* [Compile Caffe2 models](https://tvm.apache.org/docs/tutorials/frontend/from_caffe2.html)


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

   [Compiling MXNet Models]: https://tvm.apache.org/docs/tutorials/frontend/from_mxnet.html#sphx-glr-tutorials-frontend-from-mxnet-py
   [here]: https://tvm.apache.org/docs/tutorials/index.html#compile-deep-learning-models
   [Apache TVM Tutorials]: https://tvm.apache.org/docs/tutorials/index.html
   [Compiling Deep Learning Models]: https://tvm.apache.org/docs/tutorials/index.html#compile-deep-learning-models
  
