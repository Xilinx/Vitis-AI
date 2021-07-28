make -f ../../../bin/common/waa_trd.mk \
ACCEL=classification-pre_jpeg \
PREBUILT_DPU=DPUv2_B4096_2dpu_zcu102_jpg \
OUTPUT_DIR=binary_container_1 \
ADDON=JPEG \
all | tee make.log
