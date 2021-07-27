#Generate Pre-processor pipeline xo
export CUR_DIR=$PWD
cd $CUR_DIR/../../../accel/classification-pre_int8
make cleanall
make xo TARGET=hw

#Generate xclbin
cd $CUR_DIR
make clean
make all

