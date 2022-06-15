source /proj/xbuilds/2022.1_released/installs/lin64/Vitis/2022.1/settings64.sh 
source /proj/xbuilds/2022.1_released/xbb/xrt/packages/setenv.sh
export PLATFORM_REPO_PATHS=/proj/xbuilds/2022.1_released/internal_platforms/
export OPENCV_INCLUDE=/proj/xtools/dsv/projects/sdx_libs/cpp_libs/opencv/opencv3/include
export OPENCV_LIB=/proj/xtools/dsv/projects/sdx_libs/cpp_libs/opencv/opencv3/lib64
export LD_LIBRARY_PATH=/proj/xtools/dsv/projects/sdx_libs/cpp_libs/opencv/opencv3/lib64
export DEVICE=/proj/xbuilds/2022.1_released/internal_platforms/xilinx_zcu102_base_202210_1/xilinx_zcu102_base_202210_1.xpfm
export SDX_PLATFORM=/proj/xbuilds/2022.1_released/internal_platforms/xilinx_zcu102_base_202210_1/xilinx_zcu102_base_202210_1.xpfm
export EDGE_COMMON_SW=/proj/xbuilds/2022.1_released/internal_platforms/sw/zynqmp/xilinx-zynqmp-common-v2022.1
export SYSROOT=/proj/xbuilds/2022.1_released/internal_platforms/sw/zynqmp/xilinx-zynqmp/sysroots/aarch64-xilinx-linux/
./run.sh
