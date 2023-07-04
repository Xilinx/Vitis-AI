
"""!
    vart.RunnerExt Example
   
    The vart.RunnerExt Class example is show below.

    @code
      # create runner
      runner = vart.RunnerExt.create_runner(subgraph, “run”)
      # get input and output tensor buffers
      input_tensor_buffers = runner.get_inputs()
      output_tensor_buffers = runner.get_outputs()
      # run graph runner
      v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
      runner.wait(v)
      output_data = np.asarray(output_tensor_buffers[0])
    @endcode
   
"""

