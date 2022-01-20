########################################################
#  impl_config.cl, used for global settings
#
puts "Start to source [info script]"

set BOARD "u200"
set VIVADO_VER "202002"
set DPU_NUM 2
set SLR 0
set FREQ 600
set SHELL_VER "202002"

regexp {build_flow/DPUCADF8H_u200} $loc script_dir
set SDA_PATH "[string range $loc -1 [expr [string first $script_dir $loc] -1 ] ]build_flow/DPUCADF8H_u200"

set BUILD_DIR $SDA_PATH/build_dir.hw.xilinx_u200_gen3x16_xdma_1_202110_1
