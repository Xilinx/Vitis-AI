#setenv PLATFORM_REPO_PATHS /proj/DAB/ravishan/abstract_shell/from_nkpavan/dpuv3_me/20201103_u50_from_bo_with_pp/rel_to_Pavan.20201103/WAA-TRD/bin
#setenv DEVICE xilinx_u50_gen3x4_xdma_2_202010_1
#setenv SDX_PLATFORM /proj/DAB/ravishan/abstract_shell/from_nkpavan/dpuv3_me/20201103_u50_from_bo_with_pp/rel_to_Pavan.20201103/WAA-TRD/bin/xilinx_u50_gen3x4_xdma_2_202010_1/xilinx_u50_gen3x4_xdma_2_202010_1.xpfm
#source /proj/xbuilds/2020.2_daily_latest/xbb/xrt/packages/setenv.sh
#source /proj/xbuilds/2020.2_daily_latest/installs/lin64/Vitis/2020.2/settings64.sh

make -f ../../../bin/common/waa_trd.mk \
ACCEL=classification-pre \
PREBUILT_DPU=DPUV3E_3ENGINE_2dpu_u50 \
DPUver=DPUV3E_3ENGINE \
OUTPUT_DIR=_x_output_noKernFreq \
dpu.xclbin | tee make_noKernFreq.log
