export CUR_DIR=$PWD

#Generate SD card files
cd $CUR_DIR/vitis_prj
make all CV_ACCEL=../../../../../plugins/cost_volume/pl/ 
