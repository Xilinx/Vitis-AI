#!/usr/bin/env bash
#echo "Creating environment..."
#conda env create -f code/configs/environment.yaml
#source activate test_personreid

echo "Start test..."
checkpoint_dir=/group/modelzoo/internal-cooperation-models/pytorch/personreid

python3 quant.py --quant_mode 1 --config_file='configs/personreid_market.yml' \
--dataset='market1501' \
--dataset_root='/group/modelzoo/test_dataset/market1501' \
--load_model=${checkpoint_dir}'/personreid_resnet50.pth' \
--device=gpu \
 | tee ./test_personreid.log

python3 quant.py --quant_mode 2 --config_file='configs/personreid_market.yml' \
--dataset='market1501' \
--dataset_root='/group/modelzoo/test_dataset/market1501' \
--load_model=${checkpoint_dir}'/personreid_resnet50.pth' \
--device=gpu \
 | tee ./test_personreid.log


#--load_model=${checkpoint_dir}'/personreid_large.pth.tar' \
