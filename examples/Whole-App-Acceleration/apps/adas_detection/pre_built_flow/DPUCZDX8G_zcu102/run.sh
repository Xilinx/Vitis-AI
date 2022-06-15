#Generate SD card files
export TRD_PATH=$CUR_DIR/DPU-TRD/DPUCZDX8G/
export ACCEL=blobfromimage
make -f ../scripts/waa_trd.mk clean

make -f ../scripts/waa_trd.mk \
PREBUILT_DPU=DPUCZDX8G \
DPUver=DPUCZDX8G \
OUTPUT_DIR=binary_container_1 \
all | tee make.log
