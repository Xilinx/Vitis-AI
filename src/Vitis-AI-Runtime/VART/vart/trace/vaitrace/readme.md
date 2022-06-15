# Xilinx Vitis AI Trace

vaitrace is a tool that could help tracing every detail in Xilinx Vitis AI Labrary and your AI interence applications based on Xilinx Vitis AI Labrary.

# Features
- This tool can helps profile and visualize AI inference applications and figure out hot-spots and bottlenecks of performance.

- The usage of this tool is quite easy, no any change in your code and no need to re-complie anything, just write a configuration file in json format to tell our tool that which function is needed to be traced. Also we will offer a basic template of configuration file with this you could easily get insight of Xilinx Vitis AI Labrary.

- Very low tracing overhead comparing to any traditional logging tools.

- This tool is based on Ftrace and Uprobe utilities to get the raw trace records, and there is an python script to analyze the records then output some human-readable infomation.

- This tool cad analyze multithreading programs precisely, it could figure out between the time that program is really running on a CPU core and the time of program that is scheduled out putting into a sleep state, in the other word, the sleep time of a thread is measurable, this feature is very useful while analyzing multithreading programs.

# Prerequisites

## Ftrace
Ftrace must be configured/enabled in the kernel. This requires CONFIG_FTRACE and other Ftrace options.

## Python
A python-3.7 or above version is requied

## Application and librarys
The tool need symbol tables to locate the address of functions that to be traced in memory, **The executable application and librarys to be traced could not be stripped**, GNU strip command will discard all symbols and debug infomation from object files. and it's better to compile your application with [-g] and [-pg] options.

# Quick Start
Let's begin with a very simple case, which is based on DNNDK libs

First of all entry the resnet50 sample directory
```shell
# cd /PathVitisAILabrary/samples/dnndk_sample/resnet50
```

Then build the sample
```shell
# sh build.sh
```

Try to start our first trace, threre is a simple default configuration inside the tool, but if you can specified a configuration file, it will get a better result
```shell
# vaitrace ./test_dnndk_resnet50
```

```

By default the result will be sorted by thread, so now we can see the detail of how a AI inference test program based on DNNDK runs, and the report shows the time costs for every function

# Command line options

```shell
# vaitrace -h

 usage: vaitrace [-h] [-a [TRACESAVETO]] [-b [BUFFERSIZE]] [-c [CONFIG]]
                     [-d] [-j] [-o [OUTPUT]] [-p] [-r] [-s] [-t [TIMEOUT]] [-v]
                     cmd [cmd ...]
 
 positional arguments:
 	cmd
 
 optional arguments:
 	-h, --help        show this help message and exit
 	-a [TRACESAVETO]  Save trace file to
 	-b [BUFFERSIZE]   Buffer size in kb for ftrace
 	-c [CONFIG]       Specify the config file
 	-d                Enable debug
 	-j                Just run trace
 	-o [OUTPUT]       Dump the result into a file, instead of stdout
 	-p                Parse trace file only
 	-r                Dryrun, just try parsing and checking config file
 	-s                Print result to screen
 	-t [TIMEOUT]      Profile time limitation
 	-v                Show version
```
 
- cmd: the command to be run, if the cmd have options, the command line parser of vaitrace might be confused, in such case, try put the whole command within double quotations, it helps the command parser work better:

- a: Save the raw trace recoeds to file

- p: Parse a raw trace record file got by **-a**

- s: The result will be printed to screen instead of saving to a file

- t [TIMEOUT]: Only enable tracing for **TIMEOUT** seconds

- r: With this option, the command will not be run actully, only try parse the configuration file and show the symbol table matching result


### Example
```shell
# vaitrace -a trace.txt -t 1 -c ./tracecfg_thv_test_performance_roadline.json -o report.txt "./test_performance_roadline ./test_performance_roadline.list -t4"
```
This command will:
1. Run and trace [./test_performance_roadline ./test_performance_roadline.list -t4], this is the sample of roadline detection in Xilinx Vitis AI Labrary
2. The out put report file will be named "report.txt" and saved in current working directory
3. Only the first second of program running will be traced, it's enough for this case, because every second a running program will produce thousands of records, too much records will slow down the speed of program running
4. The raw trace records will be saved to trace.txt, it's useful for debugging, and trace.txt can be parsed by **-p** option
```shell
# vaitrace ./trace.txt
```
5. **-c** specified a configuration file named *tracecfg_thv_test_performance_roadline.json*

# Trace configuration file:
## tracecfg.json:
The vaitrace will by default looking for *tracecfg.json* in the working directory **-c** option could be used to specify the configuration file
```json
{
	"trace_inference_run":[
		"run_resnet_50",
		"dpuRunTask",
		"dpuRunSoftmax",
		"dpuGetOutputTensorScale",
		"dpuSetInputImage2",
		"dpuCreateTask",
		"dpuDestroyTask",
		"TopK",
		"imread"
	],

	"options":{
		"trace_sched_switch": true,
		"thread_view": true
	}
}
```

### trace\_inference\_run
A list of function names which will be traced, namespace is acceptable something like "cv::resize" could be parsed well
### options
- trace\_sched\_switch
If true the thread switch event **sched_switch** from Linux kernel will be analysized, the report will show each function call's scheduled out time

- thread_view
If true the result will be sorted by thread, if false thr result will be sorted by images, our analyzer will try apportion events to each image, we call it **image\_view**, we hope that sorting events by images will show infomation about an 'end-to-end' latency of each image.

- inference\_top\_function
If **thread\_view** is false, in anouther word the "image_view" is enabled, we need to appoint a function that will be treated as the top level of each AI inference procedure, it's the container of all AI inference routines all DNNDK API calls included, in Xilinx Vitis AI Labrary based applications, we recommand it to be "run(img)" function of each class.  It's not so convent to ask user to appoint a "top" function, so in the next version of vaitrace we will try to interduce a new events analyzer that will apportion events all automatically.

- max\_imgs
The limitation of images in output report, only vaild if **thread_view** is false, we set this limitation because there might be hundreds of images in the whole records, we can just pick up few of them to output, so that the report could be concise.


# Known Issues

- The symbols table parser could not distinguish the C++ Overloading functions well 
- If a thread forks some child threads the child threads running time may not be counted in when option **trace_sched_switch** is true

# Further Considerations

It's a very initial version of this tool, there are more useful feathers under developing(the following items ranking by priority):

1. Outputing graph report instead of text report, make it a real visualize and easy to use tool

2. Dig out more useful info from the raw trace records. Enable a new analyzing algorithm for example     end-to-end latency from image loading to result display:

3. MPSoC system level profiling, because the bandwidth might become the bottleneck of AI inference application especially for a 'edge' SoC chip, AXI bus bandwidth usage and DDR memory bandwidth must be taken into account for a system level profiling

4. Open more details about DPU hardware working. This tool may get more abilities to show DPU hardware insights

5. Auto-Run mode, no need of any user configuration, auto run and figure out the top 5% time consumption functions

6. A tool to figure out the differences of two report, it's very useful for regression testing and automated testing
