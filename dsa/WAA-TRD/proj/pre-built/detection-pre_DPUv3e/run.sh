# setenv PLATFORM_REPO_PATHS /proj/xbuilds/2020.2_daily_latest/internal_platforms
# setenv DEVICE xilinx_u50_gen3x4_xdma_2_202010_1
export PLATFORM_REPO_PATHS=`readlink -f ../../../bin`
export SDX_PLATFORM=xilinx_u50_gen3x4_xdma_2_202010_1

source /proj/xbuilds/2020.2_daily_latest/xbb/xrt/packages/setenv.sh
source /proj/xbuilds/2020.2_daily_latest/installs/lin64/Vitis/2020.2/settings64.sh

make -f ../../../bin/common/waa_trd.mk \
ACCEL=detection-pre \
PREBUILT_DPU=DPUV3E_3ENGINE_2dpu_u50 \
DPUver=DPUV3E_3ENGINE \
OUTPUT_DIR=_x_output \
BOARD=u50 \
dpu.xclbin | tee make.log
