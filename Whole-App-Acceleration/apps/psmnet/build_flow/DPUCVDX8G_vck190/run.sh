export CUR_DIR=$PWD
cd $CUR_DIR/../../../../../dsa/XVDPU-TRD/vck190_platform
make all

#Generate SD card files
cd $CUR_DIR/vitis_prj
make all CV_ACCEL=../../../../../plugins/cost_volume/pl/ XVDPU_PATH=../../../../../../dsa/DPUCVDX8G-XO 
