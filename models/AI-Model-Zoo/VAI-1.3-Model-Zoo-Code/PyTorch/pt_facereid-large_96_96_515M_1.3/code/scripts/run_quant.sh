#!/usr/bin/env bash
#echo "Creating environment..."
#conda env create -f code/configs/environment.yaml
#source activate test_facereid

echo "Start testing..."
checkpoint_dir=path_to_facereid_model
data_dir=path_to_facereid_dataset
export W_QUANT=1

echo "[Calib mode]testing "
python test.py \
--quant_mode 'calib' \
--config_file='configs/facereid_large.yml' \
--dataset='facereid' \
--dataset_root=${data_dir}'/face_reid' \
--load_model=${checkpoint_dir}'/facereid_large.pth.tar' \
--output_path='facereid_large_quant_result' \
--device=gpu \
 | tee ./test_facereid_large.log

echo "[Test mode]testing and dumping model"
python test.py --quant_mode 'test' --config_file='configs/facereid_large.yml' \
--dataset='facereid' \
--dataset_root=${data_dir}'/face_reid' \
--load_model=${checkpoint_dir}'/facereid_large.pth.tar' \
--output_path='facereid_large_quant_result' \
--device=cpu \
--dump_xmodel \
 | tee ./test_facereid_large.log

echo "Finish testing!"
