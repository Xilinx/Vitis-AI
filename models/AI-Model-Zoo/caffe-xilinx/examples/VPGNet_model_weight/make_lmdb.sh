# Declare $PATH_TO_DATASET_DIR and $PATH_TO_DATASET_LIST

./build/tools/convert_driving_data $PATH_TO_DATASET_DIR $PATH_TO_DATASET_LIST/train_caltech.txt LMDB_train
./build/tools/compute_driving_mean LMDB_train ./driving_mean_train.binaryproto lmdb
./build/tools/convert_driving_data $PATH_TO_DATASET_DIR $PATH_TO_DATASET_LIST/test_caltech.txt LMDB_test

