
## Run tensorflow ssd-mobilenet 

## set environment
export RTE_ACQUIRE_DEVICE_UNMANAGED=1
export XLNX_VART_FIRMWARE=/usr/lib/dpu.xclbin
export XLNX_ENABLE_FINGERPRINT_CHECK=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

## Disable for Software postprocess 
## set 1 for HW else 0
EN_HWPOSTPROC=$4
if [[ $EN_HWPOSTPROC -eq 1 ]];
then
    # Disable DPU output data transfer to host
	export DPU_HW_POST=1
else
    # Enable DPU output data transfer to host
	export DPU_HW_POST=0
fi

# Run on FPGA
./app.exe $1 $2 $3 1 $4
