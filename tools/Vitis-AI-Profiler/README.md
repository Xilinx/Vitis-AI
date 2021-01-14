<p align="center">
    <img src="img/Vitis-AI.png"/>
</p>

# Xilinx Vitis AI Profiler

## Overview

Vitis-AI Profiler is an application level tool that could help to optimize the whole AI application. The main purpose of Vitis-AI profiler is to help detect bottleneck of the whole AI application.With Vitis-AI Profiler, you can profile the pre-processing functions and the post-processing functions together with DPU kernels' running status. If the profiling result shows that one pre-processing function takes very long time, it leads to a high CPU utilization and DPU cores wait a long time for CPUs to finish processing. In this situation, we find that the pre-processing and CPU is the bottleneck. if user want to improve performance, try to rewrite this pre-processing function with HLS or OpenCL to reduce CPU's workload.
- It’s easy to use, this tool requires neither any change in user’s code nor re-compilation of the program
- Figuring out hot spots or bottlenecks of preference at a glance
- vaitrace: running on devices, take the responsibility for data collection

## What's New
### v1.3 
- Use Vitis Analyzer 2020.2 as default GUI
- Support profiling for Vitis-AI python applications
- Fix various vaitrace bugs

## Vitis AI Profiler Architecture
<div align="center">
<img width="800px" src="img/arch.png"/>
</div>  

## Why Vitis AI Profiler
### What's the benefit of this tool
- An all-in-one profiling soution for Vitis-AI
- Vitis-AI is a heterogeneous system, it's complicated, so that we need a more powerful and customized tool for profiling. The Vitis AI Profiler could be used for a application level profiling. For a AI application, there will be some parts running on hardware, for example, neural network computation usually runs on DPU, and also, there are some parts of the AI application running on CPU as a function that was implemented by c/c++ code like image pre-processing. This tool could help put the running status of all parts together. So that, we get an all-in-one profiling tool for Vitis-AI applications. 

### What Information Can Be Obtained from This Tool

<p align="center"> Vitis AI Profiler GUI Overview<img src="img/vitis_analyzer_timeline1.PNG"></p>

From Vitis-AI v1.3, [Vitis Analyzer](https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/jfn1567005096685.html) is the default GUI for vaitrace

- DPU Summary  
  A table of the number of runs and min/avg/max times for each kernel 
  <p align="center"><img src="img/dpu_dma_profiling.png"></p>

- DPU Throughput and DDR Transfer Rates  
  Line graphs of achieved FPS and read/write transfer rates (in MB/s) as sampled during the application
  <p align="center"><img src="img/dpu_summary.png"></p>
- Timeline Trace   
  This will include timed events from VART, HAL APIs, and the DPUs
      <p align="center"><img src="img/timeline_trace.png"></p>

- Notes  
  - From Vitis-AI v1.3, Vitis Analyzer is the default GUI for vaitrace
  - The legacy Vitis-AI Profiler web GUI still works for edge devices(Zynq MPSOC) in Vitis-AI v1.3, for more information, please see [Legacy Vitis-AI Profiler Web GUI README](./README_LEGACY.md)

## Get Started with Vitis AI Profiler
-	System Requirements  
    - Hardware  
        - Support Zynq MPSoC (DPUCZD series)
        - Support Alveo (DPUCAH series)
    - Software  
        - Support VART v1.2+
        - Support Vitis AI Library v1.2+

- Installing  
    - Preparing debug environment for vaitrace in MPSoC platform   
      These steps are __not__ required for Vitis AI prebuilt images for ZCU102 & ZCU104 https://github.com/Xilinx/Vitis-AI/tree/master/VART#setting-up-the-target   
        1. Configure and Build Petalinux  
        Run _petalinux-config -c kernel_ and Enable these for Linux kernel  
        ```
        General architecture-dependent options ---> [*] Kprobes
        Kernel hacking  ---> [*] Tracers
        Kernel hacking  ---> [*] Tracers  --->
        			[*]   Kernel Function Tracer
        			[*]   Enable kprobes-based dynamic events
        			[*]   Enable uprobes-based dynamic events
        ```    
        2. Run _petelinux-config -c rootfs_ and enable this for root-fs  
        ```
        user-packages  --->  modules   --->
        		[*]   packagegroup-petalinux-self-hosted
        ```
        3. Run _petalinux-build_ and update kernel and rootfs

    - Preparing debug environment for docker
      If you are using Vitis AI with docker, please add this patch to docker_run.sh to get root permission for vaitrace  
      ```diff
      @@ -89,6 +71,7 @@ docker_run_params=$(cat <<-END
           -e USER=$user -e UID=$uid -e GID=$gid \
           -e VERSION=$VERSION \
           -v $DOCKER_RUN_DIR:/vitis_ai_home \
      +    -v /sys/kernel/debug:/sys/kernel/debug  --privileged=true \
           -v $HERE:/workspace \
           -w /workspace \
           --rm \

      ```
      - This step is only required for Alveo devices working in docker environment   
        - For Zynq MPSoC devices, vaitrace does not interact with docker, therefore modification for the docker_run.sh is __not required__    
        - For Alveo devices running in docker environment, there are some limitations for an in-depth profiling. Because some tools require superuser permission that cannot work well with docker in default setting. So we need this modification to get more permissions

      - Due to an issue of overlay-fs, to support all the features of Vitis-AI Profiler in docker environment, it's recommended to use Linux kernel 4.8 or above on your host machine, see [here](https://lore.kernel.org/patchwork/patch/890633/)


### Starting A Simple Trace with vaitrace  
We use VART resnet50 sample  
  - Download and setup Vitis AI
  - Start testing and tracing
    - vaitrace requires root permission
    ```bash
      # sudo bash
    ```
    - For C++ programs, add vaitrace in front of the test command, the test command is:
    ```bash
      # cd ~/Vitis_AI/examples/VART/samples/resnet50
      # vaitrace ./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
    ```
    - For Python programs, add -m vaitrace_py to the python interpreter command
    ``` bash
      # cd ~/Vitis_AI/examples/VART/samples/resnet50_mt_py
      # python3 -m vaitrace_py ./resnet50.py 2 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
    ```
  -	vaitrace and XRT generates some files in the working directory  
Copy all .csv files and xclbin.ex.run_summary to you PC, the xclbin.ex.run_summary can be opened by vitis_analyzer 2020.2 and above
    -	Command Line:
    ```bash
      # vitis_analyzer xclbin.ex.run_summary
    ```
    -	GUI:  
    File->Open Summary… : select the xclbin.ex.run_summary
  - About Vitis Analyzer, please see [Using the Vitis Analyzer](https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/jfn1567005096685.html)


## Vaitrace Usage
### Command Line Usage
```bash
# vaitrace --help
usage: Xilinx Vitis AI Trace [-h] [-c [CONFIG]] [-d] [-o [TRACESAVETO]] [-t [TIMEOUT]] [-v]

  cmd   Command to be traced
  -b    Bypass mode, just run command and by pass vaitrace, for debug use
  -c [CONFIG]       Specify the configuration file
  -o [TRACESAVETO]  Save trace file to
  -t [TIMEOUT]      Tracing time limitation, default value is 30(s) for vitis analyzer format, abd 5(s) for .xat format
  --va              Generate trace data for Vitis Analyzer(Default format)
  --xat             Generate trace data in .xat(Only available for Zynq devices)

```
### Important and frequently used arguments
- cmd: cmd is your executable program of vitis-ai that want to be traced  

-	-t  control the tracing time(in seconds) starting from the [cmd] being launched, default value is 30 in --va mode and 5 in --xat mode
  
-	-o  where to save trace file (.xat), by default, the .xat file will be saved to current working directory, and be named as the same as executable program plus .xat as suffix. In --va mode, -o is invalid
-	-c  users can start a trace with more custom options by writing these a json  format configuration file and specify the configuration by -c, details of configuration file will be explained in the next section
- --va generate trace data for Vitis Analyzer, this option is enabled by default
- --xat generate trace data in .xat, the .xat can be opened by the leagcy web-based Vitis-AI Profiler, this option is disabled by default

### Configuration
Another way to launch a trace is to save all necessary information for vaitrace into a configuration file then use __vaitrace -c [config_name.json]__

- Configuration priority: Configuration File > Command Line > Default

- Here is an example of vaitrace configuration file
    ```json
    {
      "options": {
        "runmode": "normal",
        "cmd": "/home/root/Vitis_AI/examples/VART/samples/resnet50/resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel",
        "timeout": 10
      },
      "trace": {
        "enable_trace_list": ["vitis-ai-library", "opencv", "vart", "custom"]
      },
      "trace_custom": []
    }
    ```

    |  Key | Sub-level Key | Type | Description |
    |  :-  | :- | :-: | :- |
    | options  |  | object |  |
    |   | cmd | string | the same with command line argument cmd |
    |   | output | string | the same with command line argument -o |
    |   | timeout | integer | the same with command line argument -t |
    |   | runmode | string | Xmodel run mode control, can be “debug” or “normal”, if runmode == “debug” VART will control xmodel run in a debug mode by using this, user can achieve **fine-grained profiling** for xmodel. For .elf type mode, only "normal" is valid |
    | trace  |  | object | |
    |  | enable_tracelist | list_of_string |enable_trace_list	list	Built-in trace function list to be enabled, available value **"vitis-ai-library", "vart", “opencv”, "custom"**, custom for function in trace_custom list|
    |  trace_custom | | list_of_string | trace_custom	list	The list of functions to be traced that are implemented by user. For the name of function, naming space are supported. You can see an example of using custom trace function later in this document|
		

### Vitis-AI Profiler DPU Profiling Examples  
[Examples](./examples.md)
