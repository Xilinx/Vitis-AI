# Vitis-AI C++ APIs to run inference on a single image

The C++ APIs provide users an easy way to deploy Deep CNNs on FPGA. The APIs have been unified for the edge and the cloud platforms.
## Prerequisites

**vai_c compiler outputs**
  - `compiler.json` : File containing low level hardware instructions.
  - `weights.h5` : File containing preprocessed floating point data (weights/biases).
  - `quantizer.json` : File containing scaling factors for each layer in the corresponding network.
  - `meta.json` : File Containing library path, xclbin paths.

## Header Files

Any of the Vitis-AI C++ APIs can be called by including following header file and namespaces.

```c++
#include <dpu/dpu_runner.hpp>

using namespace vitis;
using namespace ai;
```

## Create Vitis Runner

Programs the hardware acceleration engine and initializes communication with the FPGA. Loads the kernel from xclbin and creates a kernel object.

**Syntax**
```c++
// Create FPGA handle
auto runners = vitis::ai::DpuRunner::create_dpu_runner (
                                            vitis_rundir
                                          );
auto runner = runners[0].get();
```
**Parameters**

***Inputs***
 - `vitis_rundir`	: (String) Path to the Vitis run directory. The directory must contain compiler output files.

***Outputs***
 - Returns runner object.

## Get Input and Output Tensors & Dimensions

Get the input and output tensors for FPGA run.

**Syntax**
```c++
// Input and Output Tensors
auto inputTensors  = runner->get_input_tensors();
auto outputTensors = runner->get_output_tensors();

// Input and Output Dimensions
auto in_dims  = inputTensors[0]->get_dims();
auto out_dims = outputTensors[0]->get_dims();
```

**Parameters**

***Outputs***
 - `inputTensors` : Onput tensor object
 - `outputTensors` : Output tensor object
 - `in_dims` : Input Dimensions
 - `out_dims` : Output Dimensions 

## Create Input and Output Tensor Buffers

Creates input and output tensor buffers.

**Syntax**
```c++
// Declare input buffer
std::vector<vitis::ai::CpuFlatTensorBuffer> inputs;
// Declare output buffer
std::vector<vitis::ai::CpuFlatTensorBuffer> outputs;
```
Follow the below steps to create memory for input and outputs.

```c++
// Get shape info
int outSize = outputTensors[0]->get_element_num() / outputTensors[0]->get_dim_size(0);
int inSize = inputTensors[0]->get_element_num() / inputTensors[0]->get_dim_size(0);

int inHeight = 0;
int inWidth = 0;
if (runner->get_tensor_format() == DpuRunner::TensorFormat::NCHW) {
    inHeight = inputTensors[0]->get_dim_size(2);
    inWidth = inputTensors[0]->get_dim_size(3);
} else {
    inHeight = inputTensors[0]->get_dim_size(1);
    inWidth = inputTensors[0]->get_dim_size(2);
}

// Batch Size
batchSize = 2

// Create Input buffer for image data
float *imageInputs = new float [inSize * batchSize];

// Create Output buffer for FPGA output
float *FCResult = new float [outSize * batchSize];

// Create Pointers to Inputs and Outputs 
std::vector<vitis::ai::TensorBuffer*> inputsPtr, outputsPtr;
// Create Batch Tensor
std::vector<std::shared_ptr<ai::Tensor>> batchTensors;

//
// Fill preprocessed image data to imageInput buffer, then proceed to next step.
//

// In/Out tensor refactory for batch input/output
batchTensors.push_back (
                std::shared_ptr<ai::Tensor> ( 
                  new ai::Tensor (
                    inputTensors[0]->get_name(),
                    in_dims, 
                    inputTensors[0]->get_data_type()
                  )
                )
              );
                
inputs.push_back (
          ai::CpuFlatTensorBuffer ( 
            imageInputs,
            batchTensors.back().get()
          )
        );
        
batchTensors.push_back (
                std::shared_ptr<ai::Tensor> (
                  new ai::Tensor (
                    outputTensors[0]->get_name(),
                    out_dims,
                    outputTensors[0]->get_data_type()
                  )
                )
              );

outputs.push_back (
          ai::CpuFlatTensorBuffer (
            FCResult, 
            batchTensors.back().get()
          )
        );

// Tensor buffer define
inputsPtr.clear();
outputsPtr.clear();

// Push the input and output tensors to the vector
inputsPtr.push_back(&inputs[0]);
outputsPtr.push_back(&outputs[0]);
```

## Execute Inference

Vitis-AI provides `execute_aync()` & `wait()` methods which supports asynchronous mode of execution on FPGA.

#### Execute of FPGA.

**Syntax**
```c++
auto job_id = runner->execute_async (
                          inputsPtr, 
                          outputsPtr
                        );
```

**Parameters**

***Inputs***
 - `inputsPtr` : Vector of Input Tensor Buffers containing the input data for inference.
 - `outputsPtr` : Vector of Output Tensor Buffers which has to be filled with output data.

***Outputs***
- `job_id` : (Pair) Return Job ID.


#### Wait for results.

**Syntax**
```c++
runner->wait (
        job_id.first, 
        ign
      );
```

***Inputs*** 
 - `job_id.first` : 
 - `ign` : Reserved (Ignore)  

## Clear Tensors

```c++
// Clear Input tensor buffer
inputs.clear();
// Clear Output tensor buffer
outputs.clear();
```

## Reference

- Refer to <a href="../examples/vitis_ai_alveo_samples/resnet50/src/main.cc">ResNet50 Classification Example</a> for use case of the C++ APIs. 
