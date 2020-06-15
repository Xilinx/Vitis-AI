#!/bin/bash

DATA_TRAIN_DIR="../../data"
PRETRAIN_MODEL_PATH="../../float/points_quality_80x60_addqneg_nodrop_gray3_1.94_12.2.pth"


ORI_PRETRAIN_PATH="/group/modelzoo/internal-cooperation-models/pytorch/face_quality/float/points_quality_80x60_addqneg_nodrop_gray3_1.94_12.2.pth"
ORI_IMG_DIR="/group/modelzoo/test_dataset/face_quality"

cd code/quantize/

ln -s $ORI_IMG_DIR $DATA_TRAIN_DIR

echo "Prepare pretrain model..."
if [ -f $PRETRAIN_MODEL_PATH ]
then echo "pretrain model exists"
else
    ln -s $ORI_PRETRAIN_PATH $PRETRAIN_MODEL_PATH
fi

echo "Begin to test"
CUDA_VISUABLE_DEVICES="0" python quant.py --quant_mode 1 --pretrained $PRETRAIN_MODEL_PATH -e
CUDA_VISUABLE_DEVICES="0" python quant.py --quant_mode 2 --pretrained $PRETRAIN_MODEL_PATH -e



echo "If quality-error is 12.33, test sucessfullly"

rm -r ../../data/face_quality
