# Compiling a Model


## Overview

 The TVM with Vitis AI flow contains two stages: Compilation and Execution. During the compilation a user can choose a model to compile for the cloud or edge target devices that are currently supported. Once a model is compiled, the generated files can be used to run the model on a the specified target device during the Execution stage. Currently, the TVM with Vitis AI flow supported a selected number of Xilinx data center and edge devices.

This document provides instructions to compile deep learning models using the TVM with Vitis AI support. We further walk through one of the tutorial scripts provided in the examples directory.

Before continuing with this document, please ensure you have properly installed and setup your environment the TVM with Vitis AI, as described in README.md document.

### Resources

If you are not familiar with Apache TVM, the following materials are provided as a guideline to understand the TVM framework. You can find more information on the Apache TVM website.

* [Apache TVM Tutorials]
* [Compiling Deep Learning Models] using TVM


### Compilation Examples

The examples directory incorporates example python scripts for compiling models using the TVM with Vitis flow. Copy the examples directory to the docker container and run any of the compile script after setting the conda environment to the "vitis-ai-tensorflow".

```sh
# In docker
$ conda activate vitis-ai-tensorflow
# copy example directory from /workspace/examples to the current directory
$ cp /workspace/examples .
$ cd examples
$ python3 compile_mxnet_resnet_18.py
```

The compilation output is saved on disk to run the model on a target device during the Execution stage. For edge devices, the compilation output needs to be transfered over to the target device.

### Compiling MXNet Resenet_18

In this section we walk through the mxnet_resent_18.py tutorial script to further demonstrate the Compilation stage of the TVM with Vitis AI support. The script demonstrates how to import, quantize and compile models using this flow.

#### Import the Model

The TVM with Vitis AI support provides ease of use by mimicking the flow of that by the TVM. As such, we leverage the front end capabilities of the TVM framework for importing models. The TVM tutorial [Compiling MXNet Models] document provides an example to import MXNet models and compile them using only the TVM compiler. The TVM documentation also provides tutorials to import models from different supported framework [here].

```python
mod, params = relay.frontend.from_mxnet(block, input_shape)
```
To be able to target the Vitis-AI cloud DPUCADX8G target we first have to import the target in PyXIR. This PyXIR package is the interface being used by TVM to integrate with the Vitis-AI stack. Additionaly, import the typical TVM and Relay modules and the Vitis-AI contrib module inside TVM.


```python
import pyxir
import pyxir.contrib.target.DPUCADX8G

import tvm
import tvm.relay as relay
from tvm.contrib.target import vitis_ai
from tvm.contrib import util, graph_runtime
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.vitis_ai import annotation
```
Similarly, DPUCZDX8G target needs to be imported to PyXIR when targetting edge devices.


#### Partition the Model

After importing the model, we utilize the Relay API to annotate the Relay expression for the provided targer and partition the graph.

```python
mod["main"] = bind_params_by_name(mod["main"], params)
mod = annotation(mod, params, target)
mod = relay.transform.MergeCompilerRegions()(mod)
mod = relay.transform.PartitionGraph()(mod)
````


#### Build the Model

The partitioned model is passed to the TVM compiler. The TVM compiler generates the runtime libraries for the TVM Runtime, for the sepecifed target. For instance, when targetting Cloud devices, the TVM target and hardware accelerator target name is set as follows:

```python
tvm_target = 'llvm'
target     = 'DPUCADX8G' # options: 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'
```

The TVM with Vitis AI flow currently supports 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'. AS mentioned previously, the "DPUCADX8G" computation engine targets cloud devices and "DPUCZDX8G-*" targets edge devices. Once the approrpiate targets are defined, we invoke the TVM compiler to build the graph for the specified target:

```python
with tvm.transform.PassContext(opt_level=3, config= {'relay.ext.vitis_ai.options.target': target}):
   lib = relay.build(mod, tvm_target, params=params)
```


#### Quantize the Model

As part of its compilation process, The TVM with Vitis AI support automatically performs quantization for the target hardware. To do so, we make use of our added On-The-Fly (OTF) Quantization feature of the TVM with Vitis AI support. Using this method one doesn’t need to quantize their model upfront; They can make use of the typical inference execution calls (module.run) to calibrate the model on-the-fly using the first N inputs that are provided. After the first N iterations, computations will be accelerated on the DPU. So now we will feed N inputs to the TVM runtime module. Note that these first N inputs will take a substantial amount of time.

We need a set of images for quantization and a callback function that needs to perform the model preprocessing on the quantization images and to return a dictionary mapping from input name to array containing dataset inputs. In this example we currently use the imagenet dataset images for quantization, but the user can choose a different dataset of their choice.

```python
quant_dir = os.path.join(HOME_DIR,'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min')

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
InferenceSession = graph_runtime.GraphModule(lib["default"](tvm.cpu()))

px_quant_size = int(os.environ['PX_QUANT_SIZE']) \
    if 'PX_QUANT_SIZE' in os.environ else 128
...
for i in range(px_quant_size):
    InferenceSession.set_input(input_name, quant_images[i]) 
    # print("running") 
    InferenceSession.run()

```
By default, the number of images used for quantization is set to 128. you could change the OTF Quantization behavior using the environment variables below:

| Varibale  | Default  | Description | 
|:-:|:-:|:-:|
| PX_QUANT_SIZE   | 128  | The number of inputs that will be used for quantization (necessary for Vitis-AI acceleration)  |
| PX_BUILD_DIR  | Use the on-the-fly quantization flow  | Loads the quantization and compilation information from the provided build directory and immediately starts Vitis-AI hardware acceleration. This configuration can be used if the model has been executed before using on-the-fly quantization during which the quantization and comilation information was cached in a build directory.  |

Lastly, we store the compiled output from the TVM compiler on disk for running the model on the target device, as follows:

```python
lib_path = "deploy_lib.so"
lib.export_library(lib_path)
```


This concludes the tutorial to compilation a model using the TVM with Vitis support. For instruction to run a compiled model please refer to the "running_on_zynq.md" and "running_on_alveo" documents


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

   [Compiling MXNet Models]: https://tvm.apache.org/docs/tutorials/frontend/from_mxnet.html#sphx-glr-tutorials-frontend-from-mxnet-py
   [here]: https://tvm.apache.org/docs/tutorials/index.html#compile-deep-learning-models
   [Apache TVM Tutorials]: https://tvm.apache.org/docs/tutorials/index.html
   [Compiling Deep Learning Models]: https://tvm.apache.org/docs/tutorials/index.html#compile-deep-learning-models
  
