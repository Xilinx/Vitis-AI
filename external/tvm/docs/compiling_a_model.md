# Compiling a Model


## Overview

 The TVM with Vitis AI flow contains two stages: Compilation and Execution. During the compilation a user can choose to compile a model for the target devices that are currently supported. Once a model is compiled, the generated files can be used to run the model on a target device during the Execution stage. Currently, the TVM with Vitis AI flow supported a selected number of Xilinx data center and edge devices.

This document provides instructions to compile deep learning models using the TVM with Vitis AI support. We further walk through one of the example tutorials provided in 'tutorials/accelerators/" directory.

Before continuing with this document, please ensure you have properly installed and setup your environment the TVM with Vitis AI, as described in README.md document.

### Resources

If you are not familiar with Apache TVM, the following materials are provided as a guideline to understand the TVM framework. You can find more information on the Apache TVM website.

* [Apache TVM Tutorials]
* [Compiling Deep Learning Models] using TVM


### Compilation Examples

While inside the docker, the "/opt/tvm-vai/tvm/tutorials/accelerators/compile/" directory incorporates example python  scripts for compiling MXNet_resnet_18 and Darknet_yolov2 models. These tutorials demonstrate the compilation step using the TVM with Vitis AI flow. While in the docker, run any of the provided tutorials after setting the conda environment to the "vitis-ai-tensorflow".

```sh
# In docker
$ conda activate vitis-ai-tensorflow
$ python3 mxnet_resnet_18.py
```

The compilation output is saved on disk in a directory that includes the compiled model as well as runtime libraries to run the model on a target device during the Execution stage. For edge devices, the output directory needs to be copied over to the target device.

### Compiling MXNet Resenet_18

In this section we walk through the mxnet_resent_18.py tutorial script to further demonstrate the Compilation stage of the TVM with Vitis AI support. The Compilation stage consists of importing, quantizing, partitioning, and compiling the model without much user intervention.

#### Import the Model

The TVM with Vitis AI support provides ease of use by mimicking the flow of that by the TVM. As such, we leverage the front end capabilities of the TVM framework for importing models. The TVM tutorial [Compiling MXNet Models] document provides an example to import MXNet models and compile them using only the TVM compiler. The TVM documentation also provides tutorials to import models from different supported framework [here].

```python
mod, params = relay.frontend.from_mxnet(block, shape_dict)
```

#### Quantize the Model

As part of its compilation process, The TVM with Vitis AI support automatically performs quantization for the target hardware. We need a set of images for quantization and an input_function() needs to perform the model preprocessing on the quantization images and to return a dictionary mapping from input name to array containing dataset inputs. In this example we currently use the imagenet dataset images for quantization, but the user can choose a different dataset of their choice.

```python
quant_dir = os.path.join(HOME_DIR,'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min')

def inputs_func(iter):
    import os

    # specify the number of images used for quantization. Currently set to 10 images
    img_files = [os.path.join(quant_dir, f) for f in os.listdir(quant_dir) if f.endswith(('JPEG', 'jpg', 'png'))][:10]
    size=shape_dict[list(shape_dict.keys())[0]][2:]
    
    # LOAD IMAGES
    imgs = []
    for path in img_files:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img.astype(np.float32))
        
    # RESNET_18 PREPROCESSING
    out = []
    for img in imgs:
        img = cv2.resize(img, tuple(size), interpolation=1)
        img = transform_image(img)
        img = img.reshape(img.shape[1:])
        out.append(img)

    # PREPARE OUTPUT
    res = np.array(out).astype(np.float32)
    print (res.shape)
    input_name = list(shape_dict.keys())[0]
    return {input_name: res}
```
The number of images used for quantization is set to 10. you could change the number of images by modifying the "img_files" variable in the above code snippet. .

#### Specify Target Hardware

The "target" parameter in the script changes the target hardware for compiling the model. By default, the models are compiled for the "DPUCADX8G" computation engine, targeting Alveo Board. The TVM with Vitis AI flow currently supports 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'.

```python
target  = 'DPUCADX8G' # options: 'DPUCADX8G', 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'
```

#### Partition the Model

After importing the model and setting up the quantization input_func() and directory, we partition the model by calling the PartitionPass() function. We pass the target hardware, the model and the parameters imported by the TVM, the quantization input_function and postprocessing function as inputs to the partition function.

```python

mod = PartitioningPass(target=target, params=params,
      inputs_func=inputs_func, postprocessing= postprocessing)(mod)
```

#### Build the Model

The output of the Partitioning is a partitioned model that is passed to the TVM compiler. The TVM compiler generates the runtime libraries for the TVM Runtime. 

```python
graph, lib, params = relay.build(mod, tvm_target, params=params)
```

Lastly, we store the graph, lib and params output from the TVM compiler on disk for running the model on the target device.

This concludes the tutorial to compilation a model using the TVM with Vitis support. For instruction to run a compiled model please refer to the "running_on_zynq.md" and "running_on_alveo" documents


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

   [Compiling MXNet Models]: https://tvm.apache.org/docs/tutorials/frontend/from_mxnet.html#sphx-glr-tutorials-frontend-from-mxnet-py
   [here]: https://tvm.apache.org/docs/tutorials/index.html#compile-deep-learning-models
   [Apache TVM Tutorials]: https://tvm.apache.org/docs/tutorials/index.html
   [Compiling Deep Learning Models]: https://tvm.apache.org/docs/tutorials/index.html#compile-deep-learning-models
  
