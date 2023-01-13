make -f ../scripts/waa_trd.mk \
ACCEL=blobfromimage \
PREBUILT_DPU=DPUCAHX8H \
DPUver=DPUCAHX8H_3ENGINE \
OUTPUT_DIR=binary_container_1 \
all | tee make.log

