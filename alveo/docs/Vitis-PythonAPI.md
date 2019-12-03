# Vitis-AI Python APIs to run inference on a single image

The python APIs provide users an easy way to deploy Deep CNNs on FPGA. The APIs have been unified for the edge and the cloud platforms.

## Prerequisites

**A. vai_c compiler output**

The compiler output directory which contains below files.

  - `compiler.json` : File containing low level hardware instructions.
  - `weights.h5` : File containing preprocessed floating point data (weights/biases).
  - `quantizer.json` : File containing scaling factors for each layer in the corresponding network.
  - `meta.json` : File containing library path, xclbin paths.

>**:pushpin: NOTE:** These directory and files are minimum requirement.

Below are the list of API.

## Import Vitis Runner and I/O APIs

Vitis Runner Class provides the abstraction for FPGA initialization and provides methods to achieve the tasks on FPGA. 
```Python
from vai.dpuv1.rt.vitis.python.dpu.runner import Runner    
```

## Create Instance of Vitis Runner  

Programs a hardware acceleration engine to the FPGA and initializes communication. Loads the kernel from xclbin and creates kernel object.

**Syntax**
```python
  runner = Runner(
                  vitis_rundir
                 )
```
**Parameters**

***Inputs***
 - `vitis_rundir` : (String) Path to the Vitis run directory. The directory must contain compiler output files.
  
***Outputs***
- `runner` : On success, returns Vitis Runner object.

## Get Input and Output Tensors

Get the input and output tensors for FPGA run.

**Syntax**
```Python
# Input Tensor
inTensors = runner.get_input_tensors()
# Output Tensor
outTensors = runner.get_output_tensors()
```
On success, these APIs return a Python list of input and output layers for FPGA processing.

## Import I/O APIs

I/O python APIs can be imported from `vai/dpuv1/rt/xdnn_io.py`. Make sure to import this module before invoking python API. 

```Python
from vai.dpuv1.rt import xdnn_io
```

## Prepare Input

Below are the sequence of APIs needed for loading of input data from image.

#### A. Get Image Paths

Defined in `vai/dpuv1/rt/xdnn_io.py`. Generates list of image paths for the given image directory or image file.

**Syntax**
```python
img_path_list = xdnn_io.getFilePaths(
                                    image_path
                                    )
```
**Parameters**

***Inputs***
 - `image_path` : (String) Path to image file or directory.

 ***Outputs***
 - `img_path_list` : (List) Returns list of image paths.


#### B. Load Image Data

Loads image and performs resize and mean subtraction.

**Syntax**
```python
imgdata, _ = xdnn_io.loadImageBlobFromFile(
                                          img_path,
                                          raw_scale,
                                          mean,
                                          input_scale,
                                          img_h,
                                          img_w
                                          )
```
**Parameters**

***Inputs***
 - `img_path` : (String) Path to the input image.
 - `raw_scale` : (Float) Raw-scale is a scale of input data. The model expects pixels to be normalized, so if the input data has values between 0 and 255, then raw_scale should be set to 255. If the data has values between 0 and 1, then raw_scale should be set to 1.
 - `mean` : (List) Image mean values.
 - `input_scale` : (Float) Input scale is a scaling factor. It should be as same the value used in training.
 - `img_h` : (Integer) Image height.
 - `img_w` : (Integer) Image width.

***Outputs***
 - `imgdata` : (Numpy Array) Returns mean subtracted array.
 - `_` : Reserved (Ignore)

## Execute Inference (Asynchronus)

`Runner` class provides `execute_aync()` & `wait()` methods which supports asynchronous mode of execution on FPGA.

**Syntax**
```python
jobId = runner.execute_async(
                 inFpgaBlobs,
                 outFpgaBlobs
                 )
```

**Parameters**

***Inputs***
 - `inFpgaBlobs` : (List) List of input nodes (numpy arrays). Array holding the input data for which to run inference.
 - `outFpgaBlobs` :  (List) List of output nodes (numpy arrays). Array for holding the result of inference ran on the hardware accelerator.

Below snippet shows how to create `inFpgaBlobs` and `outFpgaBlobs` with numpy.

```Python
# Create memory for input tensors
inFpgaBlobs = []
for t in inTensors:
  shape = (batch_sz,) + tuple([t.dims[i] for i in range(t.ndims)][1:])
  blobs.append(np.empty((shape), dtype=np.float32, order='C'))
  inFpgaBlobs.append(blobs)
# Create memory for output tensors
outFpgaBlobs = []
for t in inTensors:
  shape = (batch_sz,) + tuple([t.dims[i] for i in range(t.ndims)][1:])
  blobs.append(np.empty((shape), dtype=np.float32, order='C'))
  outFpgaBlobs.append(blobs)
```

>**:pushpin: NOTE:** Order of numpy arrats in inFpgaBlobs/outFpgaBlobs must match the order in `get_input_tensors()` and `get_output_tensors()`. See `examples/deployment_modes/test_classify.py` for example.

***Outputs***
 - `jobId` :  (Integer) Job ID.

#### Get Results in Asynchronous Execution Mode

Get result of execution for a given job.

**Syntax**
```python
runner.wait(
            jobId
           )
```
**Parameters**

***Inputs***
 - `jobId` :  (Integer) Index of the network in case of multiple networks.


>**:pushpin: NOTE:** For running multiple networks, VAI uses asynchronous mode of execution. Refer to example <a href="../examples/deployment_modes/test_classify_async_multinet.py">test_classify_async_multinet.py</a> for more details.

## References

 - Refer to example <a href="../examples/deployment_modes/test_classify.py">test_classify.py</a> for use case the python APIs.
 - Refer to example <a href="../examples/deployment_modes/test_classify_async_multinet.py">test_classify_async_multinet.py</a> for multi-net deployments.
