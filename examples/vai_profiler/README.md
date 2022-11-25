<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


# Xilinx Vitis AI Profiler

## Overview

The Vitis-AI Profiler is a tool that assists the developer in optimizing the end-to-end AI application. The main purpose of the Vitis-AI profiler is to provide developers with accurate, visual analytics of bottlenecks in the pipeline.  With the Vitis-AI Profiler, you can profile the timeline of pre and post processing functions, custom operators, as well as DPU kernel workloads. 

If the profiler results show that a specific function takes very long time to complete, there is a potential opportunity to optimize that specific function.  There are various techniques available to optimize the pipeline, including:

- Enabling DPU Hardware Softmax
- Leveraging [Whole-Application-Acceleration](../Whole-App-Acceleration) or [VVAS](https://xilinx.github.io/VVAS/)
- Modification of layer operators, activation types and layer ordering to ensure more efficient mapping to DPU-supported topologies
- Custom NEON or programmable logic accelerators (HLS, RTL)

Heterogeneous system are complex, so much so that we need a powerful and customized tool for profiling.  In the majority of AI applications, there will be functions accelerated in the programmable logic fabric, for example, neural network computation targets the DPU.  However, there will also be some portions of the pipeline executed on CPU, perhaps as a function that was implemented by with C/C++ code, such as OpenCV-based image pre-processing.  Furthermore, customized user accelerators, Xilinx IP blocks such as the Multiscaler, or the Vitis Libraries (Vision, Math, etc) may be employed to provide hardware acceleration for some pipeline functions.

The highest levels of efficiency (and hence highest inference rates) are typically achieved when the DPU is used to compute all or the majority of the graph.  To achieve optimum performance, the DPU will be processing a very high percentation of the time.  However, if other processes within the pipeline are a bottleneck (often because they are computed on the CPU using C/C++), the DPU will sit idle waiting.  This results in lower overall efficiency and inference rates.

The Vitis AI Profiler provides an end-to-end solution for profiling AI pipelines.

The Vitis AI Profiler:

- Is easy to use.  Use of this tool requires no change to the user application (no recompliation necessary)
- Enables the developer to locate hot spots or bottlenecks quickly
- Leverages vaitrace: which runs directly on the target, and which has the responsibility for data collection

****************
**What's New?** Please see the release notes [here](../docs/reference/release_notes.md)
****************

## Vitis AI Profiler Architecture

The below diagram offers a glimpse into the workflow leveraged by the VAI Profiler.  Running on the target, vaitrace collects temporal trace data in the form of a .csv file that is then copied to the host for further analysis:

<div align="center">
<img width="800px" src="img/arch_v1.4.png"/>
</div>  

### Vitis Analyzer

The Vitis Analyzer GUI is used to parse the .csv from the target, providing the developer wtih visual insights.  An example timeline of an Alveo application with two DPU cores is shown below:

<p align="center"><img src="img/vitis_analyzer_timeline1.PNG"></p>

📌Note: For Vitis-AI releases >= v1.3, the [Vitis Analyzer](https://www.xilinx.com/html_docs/xilinx2020_2/vitis_doc/jfn1567005096685.html) is the default GUI for vaitrace.

Within the Vitis Analyzer, the developer can take advantage of multiple views.  Examples of the key views are shown below:

#### DPU Summary View
  A table illustrating the number of runs as well as the min/avg/max times for each kernel deployed on the DPU 
  <p align="center"><img src="img/dpu_summary.png"></p>
  
#### DPU Throughput and DDR Transfer View
  This view provides a timeline of inference performance (FPS) as well as read/write transfer rates (in MB/s) as sampled during execution of the application
  <p align="center"><img src="img/dpu_dma_profiling.png"></p>
  
#### Timeline Trace   
  This view provides insights into timed events related to VART, HAL APIs, and the DPUs as sampled during execution of the application
      <p align="center"><img src="img/timeline_trace.png"></p>


## Get Started with Vitis AI Profiler
-	Target System Requirements  
    - Hardware  
        - All Zynq MPSoC and Kria (DPUCZD series)
        - All Alveo (DPUCAH series) platforms
        - All Versal (DPUCVDX8G/DPUCVDX8H)platforms
    - Software  
        - Supports all VART releases > v1.2
        - Supports all Vitis AI Library releases > v1.2

- Target preparation  
    - The following steps are required to enable the debug environment for vaitrace on MPSoC platforms
    
      📌**Note:** Installation is __not__ required for Vitis AI prebuilt images for the ZCU102 & ZCU104 https://github.com/Xilinx/Vitis-AI/tree/master/setup/mpsoc/VART       
        1. Configure and Build Petalinux  
        Run _petalinux-config -c kernel_ and enable the below settings for the Linux kernel  
        ```
        General architecture-dependent options ---> [*] Kprobes
        Kernel hacking  ---> [*] Tracers
        Kernel hacking  ---> [*] Tracers  --->
        			[*]   Kernel Function Tracer
        			[*]   Enable kprobes-based dynamic events
        			[*]   Enable uprobes-based dynamic events
        ```    
        2. Run _petalinux-config -c rootfs_ and enable the below settings for the root-fs  
        ```
        Petalinux package Groups  --->
			packaggroup-petalinux-self-hosted --->
				[*] packagegroup-petalinux-self-hosted
        ```
        3. Run _petalinux-build_ and update the kernel and rootfs on the target
  
- Host preparation


	- 📌**Important:**  The following step is only required for Alveo targets leveraging a docker environment.  Also, please note that due to an issue with overlay-fs it is recommended to use Linux kernel 4.8 or above on your host machine in order to support all the features of Vitis-AI Profiler in docker environment.  See [here](https://lore.kernel.org/patchwork/patch/890633/) 


     - With Alveo docker targets we need to modify the docker_run.sh script in order to enable root permissions for vaitrace.
        
     - For Zynq MPSoC devices, vaitrace does not interact with docker, therefore modification of the docker_run.sh script is __not required__   

     - If you are using Vitis AI with docker, please add this patch to docker_run.sh to enable root permissions for vaitrace   

      
    - If you are using Vitis AI with docker, please add this patch to docker_run.sh to enable root permissions for vaitrace    
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
      

### Starting a Simple Trace with vaitrace  
In this case, we will use a VART ResNet50 sample.  The steps are as follows:

  - Download and setup Vitis AI
  - Start testing and tracing
 
 
  - The process to trace on the target is as follows:
    - Ensure that vaitrace has root permissions
    ```bash
      # sudo bash
    ```
    - For C++ programs, add vaitrace as a precursor to the executable command, for example:
    ```bash
      # cd ~/Vitis_AI/demo/VART/resnet50
      # vaitrace ./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
    ```
    - For Python programs, add -m vaitrace_py as a switch to the python interpreter command:
    ``` bash
      # cd ~/Vitis_AI/demo/VART/samples/resnet50_mt_py
      # python3 -m vaitrace_py ./resnet50.py 2 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
    ```
  -	Upon execution, vaitrace and XRT will save several files files in the working directory.  You should copy these files (.csv files and xrt.run_summary) to your host machine.  The xrt.run_summary can be opened with Vitis Analyzer versions > 2020.2.  To open the xrt.run_summary execute the following command at the command line, and then review the summary in the GUI:
 
    -	Command Line:
    ```bash
      # vitis_analyzer xrt.run_summary
    ```
    -	GUI:  
    File->Open Summary… : select the xrt.run_summary
  - For additional documentation on using the Vitis Analyzer, please see [Using the Vitis Analyzer](https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Using-the-Vitis-Analyzer)


## Vaitrace Usage
### Command Line Usage
```bash
# vaitrace --help
usage: Xilinx Vitis AI Trace [-h] [-c [CONFIG]] [-d] [-o [TRACESAVETO]] [-t [TIMEOUT]] [-v]

  cmd   Application executable to be traced
  -b    Bypass mode, just run application and by pass vaitrace.  This is used for debug purposes.
  -c [CONFIG]       Specify the configuration file
  -o [TRACESAVETO]  Location/file to save, applicable to --txt_summary mode only
  -t [TIMEOUT]      Trace timeout, default value is 30(s)
  --txt_summary
  --txt             Display txt summary
  --fine_grained    Fine grained mode

```
### Important and frequently used arguments
- cmd: cmd is the executable application, based on Vitis AI, that is to be traced.  This should include both the executable name and standard arguments for that executable.

-	-t  set the trace timeout (in seconds).  This timeout starts when the [cmd] is launched, the default value is 30 seconds
  
-	-o  where to save report.  This option is only available for text summary mode.  By default, the test summary will be output to STDOUT.

-	-c  users can start a trace with more custom options by writing these a json format configuration file and passing this to the analyzer with the -c switch.  Details of the configuration file will be explained in the next section

- --txt  Output text summary.  Note that vaitrace will not generate the .csv and reports for Vitis Analyzer in this mode

- --fine_grained  Start trace with fine grained mode, this mode will generate a high-resolution of trace data.  In this mode, the trace time is limited to 10 seconds.

### Configuration
Another way to launch a trace is to save all necessary information for vaitrace into a configuration file then use __vaitrace -c [config_name.json]__

- Configuration priority: Configuration File > Command Line > Default

- Here is an example of vaitrace configuration file
    ```json
    {
      "trace": {
        "enable_trace_list": ["vitis-ai-library", "opencv", "vart", "custom"]
      },
      "trace_custom": []
    }
    ```

    |  Key | Sub-level Key | Type | Description |
    |  :-  | :- | :-: | :- |
    | trace  |  | object | |
    |  | enable_tracelist | list_of_string |enable_trace_list	list	Built-in trace function list to be enabled, available value **"vitis-ai-library", "vart", “opencv”, "custom"**, custom for function in trace_custom list|
    |  trace_custom | | list_of_string | trace_custom	list	The list of functions to be traced that are implemented by user. For function names, spaces are supported. You can see an example of using the custom trace function later in this document|
		

### Text Summary
When the --txt_summary or --txt option is used, vaitrace prints an ASCII table as shown in the following figure:

<div align="center">
<img width="1000px" src="img/txt_summary.png"/>
</div>  

</br>

The fields are defined in the following list:

- DPU Id: Name of the DPU instance  
- Bat: Batch size of the DPU instance  
- SubGraph: Name of subgraph in the xmodel  
- WL(Work Load): Computation workload (MAC indicates two operations), unit is GOP  
- RT(Run time) : The execution time in milliseconds, unit is ms.
- Perf: The DPU performance in unit of GOP per second, unit is GOP/s.
- LdFM (Load Size of Feature Map): External memory load size of feature map, unit is MB.
- LdWB (Load Size of Weight and Bias): External memory load size of bias and weight, unit is MB.
- StFM (Store Size of Feature Map): External memory store size of feature map, unit is MB.
- AvgBw(Average bandwidth): Average DDR memory access bandwidth.  
AvgBw = (total load size of the subgraph (including feature map and weight/bias, from
DDR/HBM to DPU bank mem) + total store size of the subgraph (from DPU bank mem to
DDR/HBM)) / subgraph runtime

### Vitis-AI Profiler Examples  
[Examples](./examples.md)
