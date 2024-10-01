
"""!  @brief Factory function to create an instance of DPU runner by
    subgraph.
    @param subgraph : xir.Subgraph,  XIR Subgraph
    @param mode 1 mode supported: 'run' - DPU runner.
    @return vart.Runner, an instance of DPU runner.

    Sample code:

     @code
     // This API can be used like:
     runner = vart.Runner.create_runner(subgraph, "run")
     @endcode
"""
def create_runner(subgraph, mode):
    pass
