wget -O cf_resnet50_imagenet_224_224_7.7G.zip https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet50_imagenet_224_224_1.1.zip
unzip cf_resnet50_imagenet_224_224_7.7G.zip

wget -O tf_resnetv1_50_imagenet_224_224_6.97G.zip https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_50_imagenet_224_224_1.1.zip
unzip tf_resnetv1_50_imagenet_224_224_6.97G.zip

export CF_NETWORK_PATH='cf_resnet50_imagenet_224_224_7.7G'
export TF_NETWORK_PATH='tf_resnetv1_50_imagenet_224_224_6.97G'

cp ${CF_NETWORK_PATH}/float/trainval.prototxt ${CF_NETWORK_PATH}/float/trainval.prototxt.bak
cp example_file/trainval.prototxt ${CF_NETWORK_PATH}/float/trainval.prototxt

