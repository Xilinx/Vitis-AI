#Generate Pre-processor xo file
export CUR_DIR=$PWD
cd $CUR_DIR/../../../../plugins/blobfromimage/pl
make cleanall
make xo TARGET=hw BOARD=Zynq ARCH=aarch64 BLOB_CHANNEL_SWAP_EN=0 BLOB_CROP_EN=0 BLOB_LETTERBOX_EN=0 BLOB_JPEG_EN=0

#Generate SD card files
cd $CUR_DIR
make KERNEL=DPU_PP BLOB_ACCEL=../../../../plugins/blobfromimage/pl DPU_IP=../../../../../dsa/DPU-TRD/dpu_ip DEVICE=zcu102
