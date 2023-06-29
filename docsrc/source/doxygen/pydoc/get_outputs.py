
"""!
   @brief Get all output tensor buffers of runner.
   @return  List[vart.TensorBuffer].  All output tensors. A vector of raw pointer to the output tensor.
   
   Sample code:

     @code
      output_tensor_buffers = runner.get_outputs()
      output_element_num = tuple(output_tensor_buffers[0].get_tensor().get_element_num())
     @endcode
"""
def get_outputs():
    pass

