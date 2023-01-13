#Generate Pre-processor xo file and SD card files
export CUR_DIR=$PWD

#Generate Pre-processor xo file
cd $CUR_DIR/../../../Whole-App-Acceleration/plugins/blobfromimage/pl/
make cleanall
make xo TARGET=hw BLOB_CHANNEL_SWAP_EN=1 BLOB_CROP_EN=0 BLOB_LETTERBOX_EN=1 BLOB_JPEG_EN=0 BLOB_NPC=4

#Generate SD card files
cd $CUR_DIR
make -f 8pe.mk xclbin BLOB_ACCEL=../../../../../plugins/blobfromimage/pl 
