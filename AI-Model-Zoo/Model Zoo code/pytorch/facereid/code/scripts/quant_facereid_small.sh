#!/usr/bin/env bash
#echo "Creating environment..."
#conda env create -f code/configs/environment.yaml
#source activate test_facereid

echo "Start testing..."
checkpoint_dir=/group/modelzoo/internal-cooperation-models/pytorch/facereid

python3 ./quant_small.py --quant_mode 1 --config_file='configs/facereid_small.yml' \
--dataset='facereid' \
--dataset_root='/group/modelzoo/test_dataset/face_reid' \
--load_model=${checkpoint_dir}'/facereid_small.pth.tar' \
--device gpu \
 | tee ./test_facereid_small.log

python3 ./quant_small.py --quant_mode 2 --config_file='configs/facereid_small.yml' \
--dataset='facereid' \
--dataset_root='/group/modelzoo/test_dataset/face_reid' \
--load_model=${checkpoint_dir}'/facereid_small.pth.tar' \
--device gpu \
 | tee ./test_facereid_small.log


echo "Finish testing!"
