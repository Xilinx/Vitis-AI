rm -f ../../../accel/classification-pre/*.xo
make -f ../../../bin/common/waa_trd.mk \
ACCEL=classification-pre \
PREBUILT_DPU=DPUv2_B4096_2dpu_zcu102 \
OUTPUT_DIR=binary_container_1 \
all
