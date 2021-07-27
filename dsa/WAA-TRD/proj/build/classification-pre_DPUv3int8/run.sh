#source /proj/xbuilds/2020.2_released/installs/lin64/Vitis/2020.2/settings64.sh
#source /proj/xbuilds/2020.2_released/xbb/xrt/packages/setenv.sh
#export SDX_PLATFORM=/proj/xbuilds/2021.1_daily_latest/internal_platforms/xilinx_u200_gen3x16_xdma_1_202110_1/xilinx_u200_gen3x16_xdma_1_202110_1.xpfm
#/proj/xbuilds/2020.2_released/internal_platforms/xilinx_u200_xdma_201830_2/xilinx_u200_xdma_201830_2.xpfm

source /proj/xbuilds/2021.1_daily_latest/installs/lin64/Vitis/2021.1/settings64.sh
source /proj/xbuilds/2021.1_daily_latest/xbb/xrt/packages/setenv.sh
export SDX_PLATFORM=/proj/xbuilds/2021.1_daily_latest/internal_platforms/xilinx_u200_gen3x16_xdma_1_202110_1/xilinx_u200_gen3x16_xdma_1_202110_1.xpfm

./build_classification_pre_int8.sh
