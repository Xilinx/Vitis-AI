#Generate Pre-processor pipeline xo and so file
export CUR_DIR=$PWD
cd $CUR_DIR/../../../accel/classification-pre
make cleanall
make xo TARGET=hw BOARD=Zynq ARCH=aarch64

#Generate SD card files
cd $CUR_DIR
make KERNEL=DPU_PP PP_ACCEL=../../../accel/classification-pre DPU_IP=../../../../DPU-TRD/dpu_ip DEVICE=zcu102
