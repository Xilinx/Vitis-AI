# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#!/usr/bin/env bash

echo "Start testing..."
export W_QUANT=0
checkpoint_dir=path_to_facereid_model
data_dir=path_to_facereid_dataset

echo "[Float mode]test..."
python test.py \
--quant_mode 'float' \
--config_file='configs/facereid_small.yml' \
--dataset='facereid' \
--dataset_root=${data_dir}'/face_reid' \
--load_model=${checkpoint_dir}'/facereid_small.pth.tar' \
--device=gpu \
 | tee ./test_facereid_small.log

echo "Finish testing!"
