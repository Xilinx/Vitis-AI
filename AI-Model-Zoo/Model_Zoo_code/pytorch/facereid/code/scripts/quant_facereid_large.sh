#!/usr/bin/env bash
#echo "Creating environment..."
#conda env create -f code/configs/environment.yaml
#source activate test_facereid

echo "Start testing..."
checkpoint_dir=/group/modelzoo/internal-cooperation-models/pytorch/facereid

python3 ./quant_large.py --quant_mode 1 --config_file='configs/facereid_large.yml' \
--dataset='facereid' \
--dataset_root='/group/modelzoo/test_dataset/face_reid' \
--load_model=${checkpoint_dir}'/facereid_large.pth.tar' \
--device gpu --gpu=0 \
 | tee ./test_facereid_large.log

python3 ./quant_large.py --quant_mode 2 --config_file='configs/facereid_large.yml' \
--dataset='facereid' \
--dataset_root='/group/modelzoo/test_dataset/face_reid' \
--load_model=${checkpoint_dir}'/facereid_large.pth.tar' \
--device gpu --gpu=0 \
 | tee ./test_facereid_large.log


echo "Finish testing!"
