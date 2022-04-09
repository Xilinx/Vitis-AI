#set to your actual path
export SYSROOT=/opt/xilinx/images/xilinx-zynqmp-common-v2021.2/sdk/sysroots/cortexa72-cortexa53-xilinx-linux
export EDGE_COMMON_SW=/opt/xilinx/images/xilinx-zynqmp-common-v2021.2
export SDX_PLATFORM=/opt/xilinx/platforms/xilinx_zcu104_base_202120_1/xilinx_zcu104_base_202120_1.xpfm

#Generate Pre-processor xo file
export CUR_DIR=$PWD
cd $CUR_DIR/../../../../plugins/blobfromimage/pl
make cleanall
make xo TARGET=hw DEVICE=xilinx_zcu104_base_202120_1 BOARD=Zynq ARCH=aarch64 BLOB_CHANNEL_SWAP_EN=1 BLOB_CROP_EN=0 BLOB_LETTERBOX_EN=1 BLOB_JPEG_EN=0 BLOB_NPC=4

#Generate SD card files
cd $CUR_DIR
make KERNEL=DPU_PP BLOB_ACCEL=../../../../plugins/blobfromimage/pl DPU_IP=../../../../../dsa/DPU-TRD/dpu_ip DEVICE=zcu104
