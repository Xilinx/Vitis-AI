# Tests for kernels
Here are three test cases for three kernels respectively in this directory. Folder 'rtmforward' is for RTM forward kernel, 'rtmbackward' for RTM bacward kernel and 'rtm' for RTM kernel. 

## Platform requirements
These kernels can only be tested on  Alveo U280, the makefile does not support
other devices.

## How to run software emulation
For small dataset:
```
	make run TARGET=sw_emu
```
For large dataset (as large as the Pluto model):
```
	make run TARGET=sw_emu RTM_height=1280 RTM_width=7040 RTM_time=12860 RTM_verify=0
```
## How to run hardware emulation
For small dataset:
```
	make run TARGET=hw_emu
```
For large dataset (as large as the Pluto model):
```
	make run TARGET=hw_emu RTM_height=1280 RTM_width=7040 RTM_time=12860 RTM_verify=0
```
## How to run hardware test

Build bitstream
```
	make build TARGET=hw DEVICE=xilinx_u280_xdma_201920_1
```
Shell xilinx_u280_xdma_201920_1 is installed on Nimbix. 

Run on hardware (Alveo U280)
```
	make run TARGET=hw
or
	make run TARGET=hw RTM_height=1280 RTM_width=7040 RTM_time=12860 RTM_verify=0
```
