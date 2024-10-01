
"""!
    vart.Runner Example
   
    This example assumes creating a DPU runner from a DPU subgraph (called dpu_subgraph).

    @code
     # create DPU runner
     dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
      
     # get a list of runner inputs
     inputTensors = dpu_runner.get_input_tensors()
      
     # optional – print names and shapes of each input tensor
     for inputTensor in inputTensors:
         print('Input tensor :',inputTensor.name, inputTensors.dims)
      
     # create input buffer
     # Important: Order of values passed to DPU thru’ input data buffer must match the order of tensor objects returned by get_input_tensor() 
     inputData = []
     for inputTensor in inputTensors:
         inputData.append(some_input_data.reshape(inputTensor.dims))
      
     # pass input buffer to DPU runner, launch and wait for completion
     job_id = dpu_runner.execute_async(inputData,outputData)
     dpu_runner.wait(job_id)   
    @endcode
   
"""

