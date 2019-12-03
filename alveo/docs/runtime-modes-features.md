# Runtime deployment modes and features

## Latency and Throughput modes

The DPU-v1 processor can run in two modes, Latency and Throughput. In Latency mode, the compiler generates a schedule that runs on the Main Block of the processor. Since all operations are done by one block, this minimizes data movement overheads and delivers the lowest latency.

In Throughput mode, the compiler generates a schedule that utilizes a Dedicated Block in addition to the Main Block. The Dedicated Block is a smaller block that will only run the first convolution operation in the network. This is to exploit the property that the first convolution makes up most of the computation ops in most networks.

Both the Dedicated and Main Blocks are configured to work in a pipelined manner, trading latency for throughput. Specifically:
- The host application sends input activation to the Dedicated Block to execute the first Convolution
- The result from the Dedicated Block is streamed to DDR, and the the Main Block is signalled to start
- The Main Block streams the activation data from DDR and executes the remaining operations of the network
- The host reads the output of the Main Block

In Throughput mode, the throughput of the DPU-v1 processor is determined by the maximum latency of the Main Block or Dedicated Block.

Example:
```
examples/deployment_modes/test_classify.py
./run.sh -t test_classify -k v3 -c latency 
./run.sh -t test_classify -k v3 -c throughput
```


## Asynchronous execution

The runtime engine allows the user application to submit concurrent inference tasks to the DPU-v1 processor asynchronously. This is so that:
- the user application can ensure that there is always work queued up to keep the FPGA 100% busy
- pre-processing and post-processing tasks before and after FPGA execution can run in parallel
- the user application is free to do other things and decide when to block and collect results from the FPGA

We use the following scheme for enqueuing execution asynchronously and collecting results:
```
  fpgaRT.exec_async()
  fpgaRT.get_result()
```

Example:
```
examples/deployment_modes/test_classify_async_multinet.py
./run.sh -t multinet -k v3 
```


## Streaming mode

For maximum throughput, it is recommended to take advantage of the DPU-v1 processor's ability to run in a streaming manner.  A typical inference flow may include the following steps: 
- A. Decode image
- B. Pre-process image
- C. Transfer image to FPGA
- D. Execute network on FPGA
- E. Read result from FPGA
- F. Post-process image

It is not necessary to wait for one image to finish all steps before proceeding to the next image. As soon as image 1 is done with step A, image 2 can be submitted to step A while image 1 moves through the next steps in the pipeline.

The following example shows how to achieve maximum end-to-end inference throughput using the API: 
```
examples/deployment_modes/mp_classify.py
./run.sh -t streaming_classify -k v3 
```

## Multiple PE and same model / Multiple PE and different models

DPU-v1 supports implementing multiple Processing Elements (PE) on one FPGA. Each PE can run independently -- PEs have their own DDR address space, on-chip memory address space and instruction memory. Because of this, the application has to flexibility mato run the same model on all PEs on an FPGA for maximum throughput for one model. 

Or, the application may load different models to different PEs for maximum throughput for multiple models. The throughput will be determined by the latency of the slowest model.

Examples:
```
# Multiple PEs running the same model (default)
examples/deployment_modes/test_classify.py
./run.sh -t test_classify -k v3 

# Multiple PEs, each running a different model
examples/deployment_modes/test_classify_async_multinet.py
./run.sh -t multinet -k v3 
```


## Single PE and multiple models

Our runtime engine supports loading multiple CNN models onto one PE. The runtime loads one set of DPU-v1 instructions for each model. Each set of instructions is assigned a start and stop index after they are pushed into the DPU-v1 processor's instruction memory. The runtime remembers the start and stop instruction index for each model. When the application requests to run a model, the runtime recalls the instruction start/stop index of that model and executes instructions in its range.


## Single model multiple PE

To further optimize for throughput, the application may partition a model so that subgraphs of one model are distributed over multiple DPU-v1 PEs. 

To optimize for lower latency, the application may apply tensorization techniques to partition and distribute one layer/operation's computation across multiple DPU-v1 PEs.


## Multiple FPGA execution

The runtime engine allows the application to select which FPGA to run a compiled model on. To take advantage of multiple FPGAs on a single host machine, the application may open one process per FPGA with an optional master process orchestrating the overall execution.

See apps/perpetual_demo/README.md
