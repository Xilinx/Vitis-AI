#Generate Pre-processor xo file and SD card files
export CUR_DIR=$PWD

#Generate Pre-processor xo file
cd $CUR_DIR/../../../../plugins/blobfromimage/pl
make cleanall
make xo TARGET=hw BLOB_CHANNEL_SWAP_EN=0 BLOB_CROP_EN=0 BLOB_LETTERBOX_EN=0 BLOB_JPEG_EN=0 BLOB_NPC=1

#Generate SD card files
cd $CUR_DIR/vitis_prj
make all BLOB_ACCEL=../../../../../plugins/blobfromimage/pl
