# (c) Copyright 2019 Xilinx, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of Xilinx, Inc. and is protected under U.S. and
# international copyright and other intellectual property
# laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.

#DATASET and WEIGHTS softlink will be created automatically

#echo "Creating environment..."
#conda env create -f ./code/conda_config/environment.yaml
#source activate Multi-task_test_env

echo "Preparing dataset..."
INTERNAL_DATA=/group/modelzoo/test_dataset/multi-task_v2/
CAL_INTERNAL_DATA=/group/modelzoo/test_dataset/multi-task_v2/BDD_Waymo_Cityscape_mixdata
#INTERNAL_MODEL=/group/modelzoo/internal-cooperation-models/pytorch/multi_task_v2/pytorch_multi-task_v2_resnet18_512*320.pth
INTERNAL_MODEL=/group/modelzoo/NNDCT_Test/Multi_task/pytorch_multi-task_v2_resnet18_512*320_mod_input.pth
CAL_DATASET=./cal_data
DATASET=../../data
WEIGHTS=../../float/pytorch_multi-task_v2_resnet18_512*320.pth


if [ -d ${DATASET} ]
then
    echo "Data links already exist!"
else
    ln -s ${INTERNAL_DATA} data
    ln -s ${INTERNAL_MODEL} ${WEIGHTS}
fi
if [ -d ${CAL_DATASET} ]
then
    echo "Data links already exist!"
else
    ln -s ${CAL_INTERNAL_DATA} cal_data
    ln -s ${INTERNAL_MODEL} ${WEIGHTS}
fi



echo "Conducting calibration test..."
export CUDA_VISIBLE_DEVICES=3
IMG_LIST=mix_data.txt

python quant.py  --quant_mode 1 --trained_model ${WEIGHTS} --cuda=True --image_root ${CAL_DATASET} --image_list ${IMG_LIST}

echo "Conducting quantized Detection test..."
export CUDA_VISIBLE_DEVICES=3
IMG_LIST=det_val.txt
#IMG_LIST=demo.txt
GT_FILE=${DATASET}/det_gt.txt
SAVE_FOLDER=./result
DT_FILE=${SAVE_FOLDER}/det_test_all.txt
TEST_LOG=${SAVE_FOLDER}/det_log.txt

python quant.py --quant_mode 2 --trained_model ${WEIGHTS}  --cuda=True --image_root ${DATASET} --image_list ${IMG_LIST}
cat ./result/det/* > ${DT_FILE}
python ./code/test/evaluation/evaluate_det.py -gt_file ${GT_FILE} -result_file ${DT_FILE} | tee -a ${TEST_LOG}
echo "Test report is saved to ${TEST_LOG}"


echo "Conducting quantized Segmentation test..."
export CUDA_VISIBLE_DEVICES=3
IMG_LIST=seg_val.txt
SAVE_FOLDER=./result
GT_FILE=${DATASET}/seg_label/
DT_FILE=${SAVE_FOLDER}/seg/
TEST_LOG=${SAVE_FOLDER}/seg_log.txt
python quant.py --quant_mode 2 --trained_model ${WEIGHTS}  --image_root ${DATASET} --image_list ${IMG_LIST}
python evaluation/evaluate_seg.py seg ${GT_FILE} ${DT_FILE} | tee -a ${TEST_LOG}
echo "Test report is saved to ${TEST_LOG}"


#echo "Conducting Demo visual..."
#IMG_LIST=demo.txt

#python test_code/demo.py --trained_model ${WEIGHTS} --image_root ${DATASET} --image_list ${IMG_LIST}
