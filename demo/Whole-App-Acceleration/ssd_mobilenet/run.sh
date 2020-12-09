
## Run tensorflow ssd-mobilenet 

#set environment
export RTE_ACQUIRE_DEVICE_UNMANAGED=1
export XLNX_VART_FIRMWARE=/usr/lib/dpu.xclbin
export XLNX_ENABLE_FINGERPRINT_CHECK=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DPU_HW_POST=1

# Run on FPGA
./app.exe $1 $2 $3 1
