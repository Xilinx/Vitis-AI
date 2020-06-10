# Tensorflow classificaiton models on Imagenet Dataset

## Enviroment requirement

* Tensorflow-gpu 1.9+(<1.14)
* Python-Opencv
* Numpy

## Dataset

* ImageNet dataset
  Download dataset at http://www.image-net.org 

## Model details

Provide several checkpoints and the frozeen Pbs file and corresponding test code. 
* Note that all models can be found at xcdl:/group/modelzoo/internal-cooperation-models/tensorflow/classification.
 
|Modelname|Ckpt Recall_1(%)|Ckpt Recall_5(%)|Pb Recall_1(%)|Pb Recall_5(%)|
|----|----|---|---|---|
|Inception_v1|69.762|89.626|69.764|89.626|
|Inception_v2|73.994|91.756|73.994|91.756|
|Inception_v3|77.978|93.942|77.978|93.942|
|Inception_v4|80.184|95.194|80.184|95.194|
|inception_resnet_v2|80.366|95.25|80.366|95.25|
|mobilenet_v1_0.25_128|41.438|66.318|41.438|66.318|
|mobilenet_v1_0.50_160|59.03|81.878|59.03|81.878|
|mobilenet_v1_1.0_224|71.02|89.9886|71.02|89.988|
|mobilenet_v2_1.0_224|70.126|89.532|70.126|89.532|
|mobilenet_v2_1.4_224|74.108|91.974|74.11|91.974|
|resnet_v1_50|75.202|92.194|75.202|92.194|
|resnet_v1_101|76.404|92.892|76.404|92.892|
|resnet_v1_152|76.814|93.174|76.814|93.174|
|resnet_v2_50|75.588|92.828|75.588|92.828|
|resnet_v2_101|76.952|93.724|76.952|93.724|
|resnet_v2_152|77.7862|94.108|77.7862|94.108|
|nasnet-a_large_04_10_2017|82.706|96.168|82.706|96.168|
|nasnet-a_moblie_04_10_2017|73.97|91.584|73.972|91.584|
|pnasnet-5_large_2017_12_13|82.858|96.182|82.858|96.182|
|pnasnet-5_moblie_2017_12_13|74.148|96.866|74.148|96.866|
|vgg_16|70.892|89.848|70.892|89.848|
|vgg_19|71.002|89.848|71.002|89.848|

## model information and evalution
 More model information can be found in "modelname"/readme.md

 Run evaluation instruction can be found in "modelname"/test_code/readme.md 
 


