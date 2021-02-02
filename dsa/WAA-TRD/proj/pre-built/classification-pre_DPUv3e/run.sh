rm -f ../../../accel/classification-pre/*.xo
make -f ../../../bin/common/waa_trd.mk \
ACCEL=classification-pre \
PREBUILT_DPU=DPUV3E_3ENGINE_2dpu_u50 \
DPUver=DPUV3E_3ENGINE \
OUTPUT_DIR=_x_output \
BOARD=u50 \
dpu.xclbin | tee make.log
