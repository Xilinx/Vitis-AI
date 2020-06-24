#!bin/sh
checkpoint='../../float/checkpoint_resnet20_bn_relu_am_msra_celebrity_80_0.4_0811_26.tar'
testset_images='../../data/test/lfw/lfw_aligned_densebox_new'
testset_list='../../data/test/lfw/lfw.txt'
testset_pairs='../../data/test/lfw/pairs.txt'

TEST_MODEL='../../float/checkpoint_resnet20_bn_relu_am_msra_celebrity_80_0.4_0811_26.tar'
ORI_TESTMODEL_PATH="/group/modelzoo/internal-cooperation-models/pytorch/facerec_pretrain_res20/float/checkpoint_resnet20_bn_relu_am_msra_celebrity_80_0.4_0811_26.tar"

python quant.py \
    --quant_mode 1 \
    --checkpoint $checkpoint \
    --testset_images $testset_images \
    --testset_list $testset_list \
    --testset_pairs $testset_pairs \
    --batch-size 128 \

python quant.py \
    --quant_mode 2 \
    --checkpoint $checkpoint \
    --testset_images $testset_images \
    --testset_list $testset_list \
    --testset_pairs $testset_pairs \
    --batch-size 128 \
