
"""!
   @brief Get all input tensors of runner.
   @return List[xir.Tensor].  A list of DPU runner inputs, each of which have type xir.Tensor.

   Sample code:

     @code
      # Each element of the list returned by get_input_tensors() corresponds to a DPU runner input. 
      # Each list element has a number of class attributes which can be displayed like this:
      inputTensors = dpu_runner.get_input_tensors()
      print(dir(inputTensors[0])
      # The most useful of these attributes are name, dims and dtype:
      for inputTensor in inputTensors:
        print(inputTensor.name)
        print(inputTensor.dims)
        print(inputTensor.dtype)
     @endcode

    Note that the dimensions (.dim) of an input tensor are in the form NHWC (batchsize, height,width,channels).
"""
def get_input_tensors():
    pass

