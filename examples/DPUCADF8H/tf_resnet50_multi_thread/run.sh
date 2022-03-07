
# vart, xir & Unilog lib dir
PREFIX=${HOME}/.local/Ubuntu.18.04.x86_64

if [ -z $1 ];
then
	# setting default path
	XILINX_VART=$PREFIX.Release
else
    XILINX_VART=$1
fi

export LD_LIBRARY_PATH=build:${CONDA_PREFIX}/lib:/opt/xilinx/xrt/lib:${XILINX_VART}/lib
export XILINX_XRT=/opt/xilinx/xrt

# Set below env corresponding to xclbin
export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCADF8H

if [[ -z "${XLNX_VART_FIRMWARE}" ]]; then
    export XLNX_VART_FIRMWARE=/usr/lib
fi

#-r path_to_xmodel
#-d path_to_image_directory
#-n number_of_images_present_in_the_image_directory_provided
#-g golden labels files provided
#-e number of threads 
#-c Number of PEs

#1 PE, 1 Thread, 100 images
build/DPUCADF8H.exe -r resnet50.xmodel -d imageDir100 -n 100 -g true -e 1 -c 1 

#4 PE, 5 Threads, 100 images
build/DPUCADF8H.exe -r resnet50.xmodel -d imageDir100 -n 100 -g true -e 5 -c 4





