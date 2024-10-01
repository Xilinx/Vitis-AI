
"""!
    @brief Factory function to create an instance of runner by graph and attributes
    @param graph  xir.Graph,  XIR Graph runners on the same graph.
    @return vart.RunnerExt.  An instance of runner.
   
    Sample code:

   @code
     # create graph runner
     graph = xir.Graph.deserialize(xmodel_file)
     runner = vitis_ai_library.GraphRunner.create_graph_runner(graph)
     # get input and output tensor buffers
     input_tensor_buffers = runner.get_inputs()
     output_tensor_buffers = runner.get_outputs()
     # run graph runner
     v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
     runner.wait(v)
     output_data = np.asarray(output_tensor_buffers[0])
   @endcode
"""
def create_graph_runner( graph):
    pass
