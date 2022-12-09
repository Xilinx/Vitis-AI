#Generate Pre-processor xo file
export CUR_DIR=$PWD
cd $CUR_DIR/../../../../plugins/blobfromimage/pl
make cleanall
make xo TARGET=hw BLOB_CHANNEL_SWAP_EN=1 BLOB_CROP_EN=0 BLOB_LETTERBOX_EN=1 BLOB_JPEG_EN=0 BLOB_NPC=4

#Generate xclbin files
cd $CUR_DIR
make clean
make u50 BLOB_ACCEL=../../../../../plugins/blobfromimage/pl DPU_XO=./release_u50_xo/
