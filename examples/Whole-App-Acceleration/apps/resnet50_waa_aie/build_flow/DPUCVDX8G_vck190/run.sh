#Generate Pre-processor xo file and SD card files
export CUR_DIR=$PWD

#Generate SD card files
cd $CUR_DIR/vitis_prj
make all
