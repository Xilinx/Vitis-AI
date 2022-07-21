<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Model Zoo</h1>
    </td>
 </tr>
 </table>

# Introduction
This repository includes optimized deep learning models to speed up the deployment of deep learning inference on Xilinx&trade; platforms. These models cover different applications, including but not limited to ADAS/AD, medical, video surveillance, robotics, data center, etc. You can get started with these free pre-trained models to enjoy the benefits of deep learning acceleration.

<p align="left">
  <img width="1264" height="420" src="images/vitis_ai_model_zoo.png">
</p>

## Vitis-AI 2.5 Model Zoo New Features！
1.Add 14 new models and total 134 models with diverse deep learning frameworks (TensorFlow 1.x, TensorFlow 2.x and PyTorch).</br>

2.The diversity of AI Model Zoo is continuously improved and fully cover a wider range of application fields.</br>

2.1 Provide more reference models for cloud requirements, such as text detection, E2E OCR, etc.</br>

2.2 Add transformer models on VCK5000，such as BERT-base NLP model，Vision Transformer (ViT), etc.</br>

2.3 Enrich OFA optimized models, such as Super-Resolution OFA-RCAN and Object Detection OFA-YOLO.</br>

2.4 For Industrial Vision and SLAM, provide Interest Point Detection & Description model and Hierarchical Localization model.</br>

3.EoU enhancement: Improved model index by application category and provide more convenient use experience.</br>

4.Provide 38 models for AMD EPYC™ CPU including popular base & optimized models. </br>


## Model Details
The following tables include comprehensive information about all models, including application, framework, input size, computation Flops as well as float and quantized accuracy.

### Naming Rules
Model name: `F_M_(D)_H_W_(P)_C_V`
* `F` specifies training framework: `tf` is Tensorflow 1.x, `tf2` is Tensorflow 2.x, `pt` is PyTorch
* `M` specifies the model 
* `D` specifies the dataset. It is optional depending on whether the dataset is public or private
* `H` specifies the height of input data
* `W` specifies the width of input data
* `P` specifies the pruning ratio, it means how much computation is reduced. It is optional depending on whether the model is pruned or not
* `C` specifies the computation of the model: how many Gops per image
* `V` specifies the version of Vitis-AI

For example, `pt_fadnet_sceneflow_576_960_0.65_154G_2.5` is `FADNet` model trained with `Pytorch` using `SceneFlow` dataset, input size is `576*960`, `65%` pruned, the computation per image is `154 Gops` and Vitis-AI version is `2.5`.

### Model Index

- Computation OPS in the table are counted as FLOPs
- Float & Quantized Accuracy unless otherwise specified, the default refers to top1 or top1/top5
- Models that have AMD EPYC™ CPU version are marked with <html>&#11088;</html>
- For more details and downloading CPU models, please refer to [UIF](https://github.com/amd/UIF)
- Supported Tasks
    - [Classification](#Classification) 
    - [Detection](#Detection) 
    - [Segmentation](#Segmentation) 
    - [NLP](#NLP) 
    - [Text-OCR](#Text-OCR)
    - [Surveillance](#Surveillance) 
    - [Industrial-Vision-Robotics](#Industrial-Vision-Robotics) 
    - [Medical-Image-Enhancement](#Medical-Image-Enhancement)

<br>

## Classification
<font size=2> 
  
| No.  | Application                                             | Name                                                      |               Float Accuracy                    |               Quantized Accuracy                 |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | General                                                 | tf\_inceptionresnetv2\_imagenet\_299\_299\_26\.35G_2.5    |                           0\.8037                            |                           0\.7946                            |                           299\*299                           |                26\.35G                |
| 2    | General                                                 | tf\_inceptionv1\_imagenet\_224\_224\_3G_2.5               |                           0\.6976                            |                           0\.6794                            |                           224\*224                           |                  3G                   |
| 3    | General                                                 | tf_inceptionv2_imagenet_224_224_3.88G_2.5                 |                            0.7399                            |                            07331                             |                           224*224                            |                 3.88G                 |
| 4    | General                                                 | tf\_inceptionv3\_imagenet\_299\_299\_11\.45G_2.5<html>&#11088;</html>          |                           0\.7798                            |                           0\.7735                            |                           299\*299                           |                11\.45G                |
| 5    | General                                                 | tf_inceptionv3_imagenet_299_299_0.2_9.1G_2.5<html>&#11088;</html>              |                            0.7786                            |                            0.7668                            |                           299\*299                           |                 9.1G                  |
| 6    | General                                                 | tf_inceptionv3_imagenet_299_299_0.4_6.9G_2.5<html>&#11088;</html>              |                            0.7669                            |                            0.7561                            |                           299\*299                           |                 6.9G                  |
| 7    | General                                                 | tf\_inceptionv4\_imagenet\_299\_299\_24\.55G_2.5          |                           0\.8018                            |                           0\.7928                            |                           299\*299                           |                24\.55G                |
| 8    | General                                                 | tf\_mobilenetv1\_0\.25\_imagenet\_128\_128\_27M_2.5       |                           0\.4144                            |                           0\.3464                            |                           128\*128                           |                27\.15M                |
| 9    | General                                                 | tf\_mobilenetv1\_0\.5\_imagenet\_160\_160\_150M\_2.5      |                           0\.5903                            |                           0\.5195                            |                           160\*160                           |                 150M                  |
| 10   | General                                                 | tf\_mobilenetv1\_1\.0\_imagenet\_224\_224\_1\.14G_2.5<html>&#11088;</html>     |                           0\.7102                            |                           0\.6780                            |                           224\*224                           |                1\.14G                 |
| 11   | General                                                 | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G_2.5<html>&#11088;</html>        |                            0.7056                            |                            0.6822                            |                           224\*224                           |                 1.02G                 |
| 12   | General                                                 | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G_2.5<html>&#11088;</html>           |                            0.7060                            |                            0.6850                            |                           224\*224                           |                  1G                   |
| 13   | General                                                 | tf\_mobilenetv2\_1\.0\_imagenet\_224\_224\_602M_2.5       |                           0\.7013                            |                           0\.6767                            |                           224\*224                           |                 602M                  |
| 14   | General                                                 | tf\_mobilenetv2\_1\.4\_imagenet\_224\_224\_1\.16G_2.5     |                           0\.7411                            |                           0\.7194                            |                           224\*224                           |                1\.16G                 |
| 15   | General                                                 | tf\_resnetv1\_50\_imagenet\_224\_224\_6\.97G_2.5<html>&#11088;</html>          |                           0\.7520                            |                           0\.7436                            |                           224\*224                           |                6\.97G                 |
| 16   | General                                                 | tf_resnetv1_50_imagenet_224_224_0.38_4.3G_2.5<html>&#11088;</html>             |                            0.7442                            |                            0.7375                            |                           224\*224                           |                 4.3G                  |
| 17   | General                                                 | tf_resnetv1_50_imagenet_224_224_0.65_2.45G_2.5<html>&#11088;</html>            |                            0.7279                            |                            0.7167                            |                           224\*224                           |                 2.45G                 |
| 18   | General                                                 | tf\_resnetv1\_101\_imagenet\_224\_224\_14\.4G_2.5         |                           0\.7640                            |                           0\.7560                            |                           224\*224                           |                14\.4G                 |
| 19   | General                                                 | tf\_resnetv1\_152\_imagenet\_224\_224\_21\.83G_2.5        |                           0\.7681                            |                           0\.7463                            |                           224\*224                           |                21\.83G                |
| 20   | General                                                 | tf\_vgg16\_imagenet\_224\_224\_30\.96G_2.5<html>&#11088;</html>                |                           0\.7089                            |                           0\.7069                            |                           224\*224                           |                30\.96G                |
| 21   | General                                                 | tf_vgg16_imagenet_224_224_0.43_17.67G_2.5<html>&#11088;</html>                 |                            0.6929                            |                            0.6823                            |                           224\*224                           |                17.67G                 |
| 22   | General                                                 | tf_vgg16_imagenet_224_224_0.5_15.64G_2.5<html>&#11088;</html>                  |                            0.6857                            |                            0.6729                            |                           224\*224                           |                15.64G                 |
| 23   | General                                                 | tf\_vgg19\_imagenet\_224\_224\_39\.28G_2.5                |                           0\.7100                            |                           0\.7026                            |                           224\*224                           |                39\.28G                |
| 24   | General                                                 | tf_resnetv2_50_imagenet_299_299_13.1G_2.5                 |                            0.7559                            |                            0.7445                            |                           299*299                            |                 13.1G                 |
| 25   | General                                                 | tf_resnetv2_101_imagenet_299_299_26.78G_2.5               |                            0.7695                            |                            0.7506                            |                           299*299                            |                26.78G                 |
| 26   | General                                                 | tf_resnetv2_152_imagenet_299_299_40.47G_2.5               |                            0.7779                            |                            0.7432                            |                           299*299                            |                40.47G                 |
| 27   | General                                                 | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G_2.5      |                        0.7702/0.9377                         |                        0.7660/0.9337                         |                           224*224                            |                 4.72G                 |
| 28   | General                                                 | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G_2.5      |                        0.7862/0.9440                         |                        0.7798/0.9406                         |                           240*240                            |                 7.34G                 |
| 29   | General                                                 | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G_2.5     |                        0.8026/0.9514                         |                        0.7996/0.9491                         |                           300*300                            |                19.36G                 |
| 30   | General                                                 | tf_mlperf_resnet50_imagenet_224_224_8.19G_2.5             |                            0.7652                            |                            0.7606                            |                           224*224                            |                 8.19G                 |
| 31   | General                                                 | tf_mobilenetEdge1.0_imagenet_224_224_990M_2.5             |                            0.7227                            |                            0.6775                            |                           224*224                            |                 990M                  |
| 32   | General                                                 | tf_mobilenetEdge0.75_imagenet_224_224_624M_2.5            |                            0.7201                            |                            0.6489                            |                           224*224                            |                 624M                  |
| 33   | General                                                 | tf2_resnet50_imagenet_224_224_7.76G_2.5                   |                            0.7513                            |                            0.7423                            |                           224*224                            |                 7.76G                 |
| 34   | General                                                 | tf2_mobilenetv1_imagenet_224_224_1.15G_2.5                |                            0.7005                            |                            0.5603                            |                           224*224                            |                 1.15G                 |
| 35   | General                                                 | tf2_inceptionv3_imagenet_299_299_11.5G_2.5                |                            0.7753                            |                            0.7694                            |                           299*299                            |                 11.5G                 |
| 36   | General                                                 | tf2_efficientnet-b0_imagenet_224_224_0.78G_2.5            |                        0.7690/0.9320                         |                        0.7515/0.9273                         |                           224*224                            |                 0.36G                 |
| 37   | General                                                 | tf2_mobilenetv3_imagenet_224_224_132M_2.5                 |                        0.6756/0.8728                         |                        0.6536/0.8544                         |                           224*224                            |                 132M                  |
| 38   | General                                                 | pt_inceptionv3_imagenet_299_299_11.4G_2.5<html>&#11088;</html>                 |                         0.775/0.936                          |                         0.771/0.935                          |                           299*299                            |                 11.4G                 |
| 39   | General                                                 | pt_inceptionv3_imagenet_299_299_0.3_8G_2.5<html>&#11088;</html>                |                         0.775/0.936                          |                         0.772/0.935                          |                           299*299                            |                  8G                   |
| 40   | General                                                 | pt_inceptionv3_imagenet_299_299_0.4_6.8G_2.5<html>&#11088;</html>              |                         0.768/0.931                          |                         0.764/0.929                          |                           299*299                            |                 6.8G                  |
| 41   | General                                                 | pt_inceptionv3_imagenet_299_299_0.5_5.7G_2.5<html>&#11088;</html>              |                         0.757/0.921                          |                         0.752/0.918                          |                           299*299                            |                 5.7G                  |
| 42   | General                                                 | pt_inceptionv3_imagenet_299_299_0.6_4.5G_2.5<html>&#11088;</html>              |                         0.739/0.911                          |                         0.732/0.908                          |                           299*299                            |                 4.5G                  |
| 43   | General                                                 | pt_squeezenet_imagenet_224_224_703.5M_2.5                 |                         0.582/0.806                          |                         0.582/0.806                          |                           224*224                            |                703.5M                 |
| 44   | General                                                 | pt_resnet50_imagenet_224_224_8.2G_2.5<html>&#11088;</html>                     |                         0.761/0.929                          |                         0.760/0.928                          |                           224*224                            |                 8.2G                  |
| 45   | General                                                 | pt_resnet50_imagenet_224_224_0.3_5.8G_2.5<html>&#11088;</html>                 |                         0.760/0.929                          |                         0.757/0.928                          |                           224*224                            |                 5.8G                  |
| 46   | General                                                 | pt_resnet50_imagenet_224_224_0.4_4.9G_2.5<html>&#11088;</html>                 |                         0.755/0.926                          |                         0.752/0.925                          |                           224*224                            |                 4.9G                  |
| 47   | General                                                 | pt_resnet50_imagenet_224_224_0.5_4.1G_2.5<html>&#11088;</html>                 |                         0.748/0.921                          |                         0.745/0.920                          |                           224*224                            |                 4.1G                  |
| 48   | General                                                 | pt_resnet50_imagenet_224_224_0.6_3.3G_2.5<html>&#11088;</html>                 |                         0.742/0.917                          |                         0.738/0.915                          |                           224*224                            |                 3.3G                  |
| 49   | General                                                 | pt_resnet50_imagenet_224_224_0.7_2.5G_2.5<html>&#11088;</html>                 |                         0.726/0.908                          |                         0.720/0.906                          |                           224*224                            |                 2.5G                  |
| 50   | General                                                 | pt_OFA-resnet50_imagenet_224_224_15.0G_2.5<html>&#11088;</html>                |                         0.799/0.948                          |                         0.789/0.944                          |                           224*224                            |                 15.0G                 |
| 51   | General                                                 | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G_2.5<html>&#11088;</html>            |                         0.795/0.945                          |                         0.784/0.941                          |                           224*224                            |                 8.2G                  |
| 52   | General                                                 | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G_2.5<html>&#11088;</html>            |                         0.791/0.943                          |                         0.780/0.939                          |                           224*224                            |                 6.0G                  |
| 53   | General                                                 | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G_2.5<html>&#11088;</html>            |                         0.777/0.937                          |                         0.770/0.933                          |                           192*192                            |                 3.6G                  |
| 54   | General                                                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G_2.5<html>&#11088;</html>            |                         0.752/0.918                          |                         0.744/0.918                          |                           160*160                            |                 1.8G                  |
| 55   | General                                                 | pt_OFA-depthwise-res50_imagenet_176_176_2.49G_2.5         |                        0.7633/0.9292                         |                        0.7629/0.9306                         |                           176*176                            |                 2.49G                 |
| 56   | General                                                 | tf_ViT_imagenet_352_352_21.3G_2.5                         |                            0.8282                            |                            0.8254                            |                           352*352                            |                 21.3G                 |
| 57   | Car type classification                                 | pt_vehicle-type-classification_CompCars_224_224_3.63G_2.5 |                            0.9025                            |                            0.9011                            |                           224*224                            |                 3.63G                 |
| 58   | Car make classification                                 | pt_vehicle-make-classification_CompCars_224_224_3.63G_2.5 |                            0.8991                            |                            0.8939                            |                           224*224                            |                 3.63G                 |
| 59   | Car color classification                                | pt_vehicle-color-classification_color_224_224_3.63G_2.5   |                            0.9549                            |                            0.9549                            |                           224*224                            |                 3.63G                 |

</font>
Note: #30 & #44-#49 are resnet_v1.5,  #33 is resnet_v1,  #50-#54 are resnet-D.
<br>

## Detection
<font size=2> 

| No.  | Application                                             | Name                                                      |               Float Accuracy                    |               Quantized Accuracy                |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | General                                                 | tf\_ssdmobilenetv1\_coco\_300\_300\_2\.47G_2.5            |                           0\.2080                            |                           0\.2100                            |                           300\*300                           |                2\.47G                 |
| 2    | General                                                 | tf\_ssdmobilenetv2\_coco\_300\_300\_3\.75G_2.5            |                           0\.2150                            |                           0\.2110                            |                           300\*300                           |                3\.75G                 |
| 3    | General                                                 | tf\_ssdresnet50v1\_fpn\_coco\_640\_640\_178\.4G_2.5       |                           0\.3010                            |                           0\.2900                            |                           640\*640                           |                178\.4G                |
| 4    | General                                                 | tf\_yolov3\_voc\_416\_416\_65\.63G_2.5                    |                           0\.7846                            |                           0\.7744                            |                           416\*416                           |                65\.63G                |
| 5    | General                                                 | tf\_mlperf_resnet34\_coco\_1200\_1200\_433G_2.5           |                           0\.2250                            |                           0\.2150                            |                          1200\*1200                          |                 433G                  |
| 6    | General                                                 | tf_ssdlite_mobilenetv2_coco_300_300_1.5G_2.5              |                            0.2170                            |                            0.2090                            |                           300*300                            |                 1.5G                  |
| 7    | General                                                 | tf_ssdinceptionv2_coco_300_300_9.62G_2.5                  |                            0.2390                            |                            0.2360                            |                           300*300                            |                 9.62G                 |
| 8    | General                                                 | tf_refinedet_VOC_320_320_81.9G_2.5                        |                            0.8015                            |                            0.7999                            |                           320*320                            |                 81.9G                 |
| 9    | General                                                 | tf_efficientdet-d2_coco_768_768_11.06G_2.5                |                            0.4130                            |                            0.3270                            |                           768*768                            |                11.06G                 |
| 10   | General                                                 | tf2_yolov3_coco_416_416_65.9G_2.5                         |                            0.377                             |                            0.331                             |                           416*416                            |                 65.9G                 |
| 11   | General                                                 | tf_yolov4_coco_416_416_60.3G_2.5                          |                            0.477                             |                            0.393                             |                           416*416                            |                 60.3G                 |
| 12   | General                                                 | tf_yolov4_coco_512_512_91.2G_2.5                          |                            0.487                             |                            0.412                             |                           512*512                            |                 91.2G                 |
| 13   | General                                                 | pt_OFA-yolo_coco_640_640_48.88G_2.5                       |                            0.436                             |                            0.421                             |                           640*640                            |                48.88G                 |
| 14   | General                                                 | pt_OFA-yolo_coco_640_640_0.3_34.72G_2.5                   |                            0.420                             |                            0.401                             |                           640*640                            |                34.72G                 |
| 15   | General                                                 | pt_OFA-yolo_coco_640_640_0.5_24.62G_2.5                   |                            0.392                             |                            0.378                             |                           640*640                            |                24.62G                 |
| 16   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_81.28G_2.5<html>&#11088;</html>               |                            0.7866                            |                            0.7857                            |                           320*320                            |                81.28G                 |
| 17   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G_2.5<html>&#11088;</html>           |                            0.7798                            |                            0.7772                            |                           320*320                            |                41.42G                 |
| 18   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G_2.5<html>&#11088;</html>          |                            0.7885                            |                            0.7826                            |                           320*320                            |                20.54G                 |
| 19   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G_2.5<html>&#11088;</html>          |                            0.7898                            |                            0.7877                            |                           320*320                            |                12.32G                 |
| 20   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G_2.5<html>&#11088;</html>           |                            0.7839                            |                            0.8002                            |                           320*320                            |                 9.83G                 |
| 21   | ADAS Traffic sign Detection                             | pt_yolox_TT100K_640_640_73G_2.5                           |                            0.623                             |                            0.621                             |                           640*640                            |                  73G                  |
| 22   | ADAS Lane Detection                                     | pt_ultrafast_CULane_288_800_8.4G_2.5                      |                            0.6988                            |                            0.6922                            |                           288*800                            |                 8.4G                  |
| 23   | ADAS 3D Detection                                       | pt_pointpillars_kitti_12000_100_10.8G_2.5                 | Car 3D AP@0.5(easy, moderate, hard) <br/>90.79, 89.66, 88.78 | Car 3D AP@0.5(easy, moderate, hard)<br/> 90.75, 87.04, 83.44 |                        12000\*100\*4                         |                 10.8G                 |
| 24   | ADAS Surround-view 3D Detection                         | pt_pointpillars_nuscenes_40000_64_108G_2.5                |                    mAP: 42.2<br>NDS: 55.1                    |                    mAP: 40.5<br>NDS: 53.0                    |                         40000\*64\*5                         |                 108G                  |
| 25   | ADAS 4D radar based 3D Detection                        | pt_centerpoint_astyx_2560_40_54G_2.5                      |            BEV AP@0.5: 32.84<br/>3D AP@0.5: 28.27            |         BEV AP@0.5: 33.82<br/>3D AP@0.5: 18.54(QAT)          |                         2560\*40\*4                          |                  54G                  |
| 26   | ADAS Image-lidar fusion based 3D Detection              | pt_CLOCs_kitti_2.5                                        | 2d detection: Mod Car bbox AP@0.70: 89.40<br/>3d detection:   Mod Car bev@0.7 :85.50<br/>Mod Car 3d@0.7 :70.01<br/>Mod Car bev@0.5 :89.69<br/>Mod Car 3d@0.5 :89.48<br/>fusionnet: Mod Car bev@0.7 :87.58<br/>Mod Car 3d@0.7 :73.04<br/>Mod Car bev@0.5 :93.98<br/>Mod Car 3d@0.5 :93.56 | 2d detection: Mod Car bbox AP@0.70: 89.50<br/>3d detection:   Mod Car bev@0.7 :85.50<br/>Mod Car 3d@0.7 :70.01<br/>Mod Car bev@0.5 :89.69<br/>Mod Car 3d@0.5 :89.48<br/>fusionnet: Mod Car bev@0.7 :87.58<br/>Mod Car 3d@0.7 :73.04<br/>Mod Car bev@0.5 :93.98<br/>Mod Car 3d@0.5 :93.56<br/> | 2d detection (YOLOX): 384*1248<br/>3d detection (PointPillars): 12000\*100\*4<br/>fusionnet: 800\*1000\*4 |                  41G                  |
| 27   | ADAS Image-lidar sensor fusion Detection & Segmentation | pt_pointpainting_nuscenes_126G_2.5                        |             mIoU: 69.1<br>mAP: 51.8<br>NDS: 58.7             |            mIoU: 68.6<br/>mAP: 50.4<br/>NDS: 56.4            |      semanticfpn:320*576<br>pointpillars:40000\*64\*16       | semanticfpn:14G<br/>pointpillars:112G |
| 28   | ADAS Multi Task                                         | pt_MT-resnet18_mixed_320_512_13.65G_2.5                   |                   mAP:39.51<br/>mIOU:44.03                   |                   mAP:38.41<br/>mIOU:42.71                   |                           320*512                            |                13.65G                 |
| 29   | ADAS Multi Task                                         | pt_multitaskv3_mixed_320_512_25.44G_2.5                   | mAP:51.2<br>mIOU:58.14<br/>Drivable mIOU: 82.57<br/>Lane IOU:43.71<br/>Silog: 8.78 | mAP:50.9<br/>mIOU:57.52<br/>Drivable mIOU: 82.30<br/>Lane IOU:44.01<br/>Silog: 9.32 |                           320*512                            |                25.44G                 |

</font>  
<br>

## Segmentation
<font size=2> 
  
| No.  | Application                                             | Name                                                      |               Float Accuracy                    |               Quantized Accuracy               |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | ADAS 2D Segmentation                                    | pt_ENet_cityscapes_512_1024_8.6G_2.5                      |                            0.6442                            |                            0.6315                            |                           512*1024                           |                 8.6G                  |
| 2    | ADAS 2D Segmentation                                    | pt_SemanticFPN-resnet18_cityscapes_256_512_10G_2.5        |                            0.6290                            |                            0.6230                            |                           256*512                            |                  10G                  |
| 3    | ADAS 2D Segmentation                                    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G_2.5   |                            0.6870                            |                            0.6820                            |                           512*1024                           |                 5.4G                  |
| 4    | ADAS 3D Point Cloud Segmentation                        | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G_2.5         |              Acc avg 0.8860<br/>IoU avg 0.5100               |              Acc avg 0.8350<br/>IoU avg 0.4540               |                           64*2048                            |                 20.4G                 |
| 5    | ADAS 3D Point Cloud Segmentation                        | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G_2.5        |                         mIou: 54.2%                          |                         mIou: 54.2%                          |                         64\*2048\*5                          |                  32G                  |
| 6    | ADAS Instance Segmentation                              | pt_SOLO_coco_640_640_107G_2.5                             |                            0.242                             |                            0.212                             |                           640*640                            |                 107G                  |
| 7    | ADAS 2D Segmentation                                    | tf_mobilenetv2_cityscapes_1024_2048_132.74G_2.5           |                            0.6263                            |                            0.4578                            |                          1024*2048                           |                132.74G                |
| 8    | ADAS 2D Segmentation                                    | tf2_erfnet_cityscapes_512_1024_54G_2.5                    |                            0.5298                            |                            0.5167                            |                           512*1024                           |                  54G                  |
| 9    | Medical Cell Nuclear Segmentation                       | tf2_2d-unet_nuclei_128_128_5.31G_2.5                      |                            0.3968                            |                            0.3968                            |                           128*128                            |                 5.31G                 |
| 10   | Medical Covid-19 Segmentation                           | pt_FPN-resnet18_covid19-seg_352_352_22.7G_2.5             |       2-classes Dice:0.8588<br/>3-classes mIoU:0.5989        |       2-classes Dice:0.8547<br/>3-classes mIoU:0.5957        |                           352*352                            |                 22.7G                 |
| 11   | Medical CT lung Segmentation                            | pt_unet_chaos-CT_512_512_23.3G_2.5                        |                         Dice:0.9758                          |                         Dice:0.9747                          |                           512*512                            |                 23.3G                 |
| 12   | Medical Polyp Segmentation                              | pt_HardNet_mixed_352_352_22.78G_2.5                       |                         mDice=0.9142                         |                         mDice=0.9136                         |                           352*352                            |                22.78G                 |
| 13   | RGB-D Segmentation                                      | pt_sa-gate_NYUv2_360_360_178G_2.5                         |                         miou: 47.58%                         |                         miou: 46.80%                         |                           360*360                            |                 178G                  |

</font>
<br>

## NLP
<font size=2> 
  
| No.  | Application                                             | Name                                                      |               Float Accuracy                    |               Quantized Accuracy               |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | Question and Answering                                  | tf_bert-base_SQuAD_128_22.34G_2.5                         |                            0.8694                            |                            0.8656                            |                             128                              |                11.17G                 |
| 2    | Sentiment Detection                                     | tf2_sentiment-detection_IMDB_500_32_53.3M_2.5             |                            0.8708                            |                            0.8695                            |                            500*32                            |                 53.3M                 |
| 3    | Customer Satisfaction Assessment                        | tf2_customer-satisfaction_Cars4U_25_32_2.7M_2.5           |                            0.9565                            |                            0.9565                            |                            25*32                             |                 2.7M                  |
| 4    | Open Information Extraction                             | pt_open-information-extraction_qasrl_100_200_1.5G_2.5     |               Acc/F1-score:<br/>58.70%/77.12%                |               Acc/F1-score:<br/>58.70%/77.19%                |                           100*200                            |                 1.5G                  |
</font>
<br>

## Text-OCR
<font size=2> 
  
| No.  | Application                                             | Name                                                      |               Float Accuracy                   |               Quantized Accuracy                |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | Text Detection                                          | pt_textmountain_ICDAR_960_960_575.2G_2.5                  |                            0.8863                            |                            0.8851                            |                           960*960                            |                575.2G                 |
| 2    | E2E OCR                                                 | pt_OCR_ICDAR2015_960_960_875G_2.5                         |                            0.6758                            |                            0.6776                            |                           960*960                            |                 875G                  |
</font>
<br>

## Surveillance
<font size=2> 
  
| No.  | Application                                             | Name                                                      |               Float Accuracy                    |               Quantized Accuracy                 |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | Face Recognition                                        | pt_facerec-resnet20_mixed_112_96_3.5G_2.5                 |                            0.9955                            |                            0.9947                            |                            112*96                            |                 3.5G                  |
| 2    | Face Quality                                            | pt_face-quality_80_60_61.68M_2.5                          |                            0.1233                            |                            0.1258                            |                            80*60                             |                61.68M                 |
| 3    | Face ReID                                               | pt_facereid-large_96_96_515M_2.5                          |                    mAP:0.794  Rank1:0.955                    |                    mAP:0.790  Rank1:0.953                    |                            96*96                             |                 515M                  |
| 4    | Face ReID                                               | pt_facereid-small_80_80_90M_2.5                           |                    mAP:0.560  Rank1:0.865                    |                    mAP:0.559  Rank1:0.865                    |                            80*80                             |                 90M5                  |
| 5    | ReID                                                    | pt_personreid-res18_market1501_176_80_1.1G_2.5            |                    mAP:0.753  Rank1:0.898                    |                    mAP:0.746  Rank1:0.893                    |                            176*80                            |                 1.1G                  |
| 6    | ReID                                                    | pt_personreid-res50_market1501_256_128_5.3G_2.5<html>&#11088;</html>           |                    mAP:0.866  Rank1:0.951                    |                    mAP:0.869  Rank1:0.948                    |                           256*128                            |                 5.3G                  |
| 7    | ReID                                                    | pt_personreid-res50_market1501_256_128_0.4_3.3G_2.5<html>&#11088;</html>       |                    mAP:0.869  Rank1:0.948                    |                    mAP:0.869  Rank1:0.948                    |                           256*128                            |                 3.3G                  |
| 8    | ReID                                                    | pt_personreid-res50_market1501_256_128_0.5_2.7G_2.5<html>&#11088;</html>       |                    mAP:0.864  Rank1:0.944                    |                    mAP:0.864  Rank1:0.944                    |                           256*128                            |                 2.7G                  |
| 9    | ReID                                                    | pt_personreid-res50_market1501_256_128_0.6_2.1G_2.5<html>&#11088;</html>       |                    mAP:0.863  Rank1:0.946                    |                    mAP:0.859  Rank1:0.942                    |                           256*128                            |                 2.1G                  |
| 10   | ReID                                                    | pt_personreid-res50_market1501_256_128_0.7_1.6G_2.5<html>&#11088;</html>       |                    mAP:0.850  Rank1:0.940                    |                    mAP:0.848  Rank1:0.938                    |                           256*128                            |                 1.6G                  |
| 11   | Person orientation estimation                           | pt_person-orientation_224_112_558M_2.5                    |                            0.930                             |                            0.929                             |                           224*112                            |                 558M                  |
| 12   | Joint detection and Tracking                            | pt_FairMOT_mixed_640_480_0.5_36G_2.5                      |                  MOTA 59.1%<br/>IDF1 62.5%                   |                  MOTA 58.1%<br/>IDF1 60.5%                   |                           640*480                            |                  36G                  |
| 13   | Crowd Counting                                          | pt_BCC_shanghaitech_800_1000_268.9G_2.5                   |                  MAE: 65.83<br>MSE: 111.75                   |             MAE: 67.60<br>MSE: 117.36                        |                           800*1000                           |                268.9G                 |
| 14   | Face Mask Detection                                     | pt_face-mask-detection_512_512_0.59G_2.5                  |                            0.8860                            |                            0.8810                            |                           512*512                            |                 0.59G                 |
| 15   | Pose Estimation                                         | pt_movenet_coco_192_192_0.5G_2.5                          |                            0.7972                            |                            0.7984                            |                           192*192                            |                 0.5G                  |

</font>
<br>

## Industrial-Vision-Robotics
<font size=2> 
  
| No.  | Application                                             | Name                                                      |               Float Accuracy                   |               Quantized Accuracy               |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | Depth Estimation                                        | pt_fadnet_sceneflow_576_960_441G_2.5                      |                          EPE: 0.926                          |                          EPE: 1.169                          |                           576*960                            |                 359G                  |
| 2    | Binocular depth estimation                              | pt_fadnet_sceneflow_576_960_0.65_154G_2.5                 |                          EPE: 0.823                          |                          EPE: 1.158                          |                           576*960                            |                 154G                  |
| 3    | Binocular depth estimation                              | pt_psmnet_sceneflow_576_960_0.68_696G_2.5                 |                          EPE: 0.961                          |                          EPE: 1.022                          |                           576*960                            |                 696G                  |
| 4    | Production Recognition                                  | pt_pmg_rp2k_224_224_2.28G_2.5                             |                            0.9640                            |                            0.9618                            |                           224*224                            |                 2.28G                 |
| 5    | Interest Point Detection and Description                | tf_superpoint_mixed_480_640_52.4G_2.5                     |                         83.4 (thr=3)                         |                         84.3 (thr=3)                         |                           480*640                            |                 52.4G                 |
| 6    | Hierarchical Localization                               | tf_HFNet_mixed_960_960_20.09G_2.5                         |        Day: 76.2/83.6/90.0<br/>Night: 58.2/68.4/80.6         |        Day: 74.2/82.4/89.2<br/>Night: 54.1/66.3/73.5         |                           960*960                            |                20.09G                 |

</font>
<br>

## Medical-Image-Enhancement
<font size=2>

| No.  | Application                                             | Name                                                      |               Float Accuracy                    |               Quantized Accuracy                |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | Super Resolution                                        | tf_rcan_DIV2K_360_640_0.98_86.95G_2.5                     |            Set5 Y_PSNR : 37.640<br/>SSIM : 0.959             |           Set5 Y_PSNR : 37.2495<br/>SSIM : 0.9556            |                           360*640                            |                86.95G                 |
| 2    | Super Resolution                                        | pt_SESR-S_DIV2K_360_640_7.48G_2.5                         | (Set5) PSNR/SSIM= 37.309/0.958<br/>(Set14) PSNR/SSIM= 32.894/ 0.911<br/>(B100)  PSNR/SSIM= 31.663/ 0.893<br/>(Urban100)  PSNR/SSIM = 30.276/0.908<br/> | (Set5) PSNR/SSIM= 36.813/0.954<br/>(Set14) PSNR/SSIM= 32.607/ 0.906<br/>(B100)  PSNR/SSIM= 31.443/ 0.889<br/>(Urban100)  PSNR/SSIM = 29.901/0.899<br/> |                           360*640                            |                 7.48G                 |
| 3    | Super Resolution                                        | pt_OFA-rcan_DIV2K_360_640_45.7G_2.5                       | (Set5) PSNR/SSIM= 37.654/0.959<br/>(Set14) PSNR/SSIM= 33.169/ 0.914<br/>(B100)  PSNR/SSIM= 31.891/ 0.897<br/>(Urban100)  PSNR/SSIM = 30.978/0.917<br/> | (Set5) PSNR/SSIM= 37.384/0.956<br/>(Set14) PSNR/SSIM= 33.012/ 0.911<br/>(B100)  PSNR/SSIM= 31.785/ 0.894<br/>(Urban100)  PSNR/SSIM = 30.839/0.913<br/> |                           360*640                            |                 45.7G                 |
| 4    | Image Denoising                                         | pt_DRUNet_Kvasir_528_608_0.4G_2.5                         |                         PSNR = 34.57                         |                       PSNR = 34.06                           |                           528*608                            |                 0.4G                  |
| 5    | Spectral Remove                                         | pt_SSR_CVC_256_256_39.72G_2.5                             |                        Visualization                         |                        Visualization                         |                           256*256                            |                39.72G                 |
| 6    | Coverage Prediction                                     | pt_C2D2lite_CC20_512_512_6.86G_2.5                        |                          MAE=7.566%                          |                         MAE=10.399%                          |                           512*512                            |                 6.86G                 |

</font>
<br>

## Model Download
Please visit [model-list](/model_zoo/model-list) in this page. You will get download link and MD5 of all released models, including pre-compiled models that running on different platforms.                                 

</details>

### Automated Download Script
With downloader.py, you could quickly find the model you are interested in and specify a version to download it immediately. Please make sure that downloader.py and [model-list](/model_zoo/model-list) folder are at the same level directory. 

 ```
 python3  downloader.py  
 ```
Step1: You need input framework and model name keyword. Use space divide. If input `all` you will get list of all models.

tf: tensorflow1.x,  tf2: tensorflow2.x,  pt: pytorch,  cf: caffe,  dk: darknet, all: list all model

Step2: Select the specified model based on standard name.

Step3: Select the specified hardware platform for your slected model.

For example, after running downloader.py and input `tf resnet` then you will see the alternatives such as:

```
0:  all
1:  tf_resnetv1_50_imagenet_224_224_6.97G_2.5
2:  tf_resnetv1_101_imagenet_224_224_14.4G_2.5
3:  tf_resnetv1_152_imagenet_224_224_21.83G_2.5
......
```
After you input the num: 1, you will see the alternatives such as:

```
0:  all
1:  tf_resnetv1_50_imagenet_224_224_6.97G_2.5    GPU
2:  resnet_v1_50_tf    ZCU102 & ZCU104 & KV260
3:  resnet_v1_50_tf    VCK190
4:  resnet_v1_50_tf    vck50006pe-DPUCVDX8H
5:  resnet_v1_50_tf    vck50008pe-DPUCVDX8H-DWC
6:  resnet_v1_50_tf    u50lv-DPUCAHX8H
......
```
Then you could choose it and input the number, the specified version of model will be automatically downloaded to the current directory.

In addition, if you need download all models on all platforms at once, you just need enter number in the order indicated by the tips of Step 1/2/3 (select: all -> 0 -> 0).

### Model Directory Structure
Download and extract the model archive to your working area on the local hard disk. For details on the various models, download link and MD5 checksum for the zip file of each model, see [model-list](/model_zoo/model-list).

#### Tensorflow Model Directory Structure
For a Tensorflow model, you should see the following directory structure:


    ├── code                            # Contains test code which can run demo and evaluate model performance. 
    │                          
    │
    ├── readme.md                       # Contains the environment requirements, data preprocess and model information.
    │                                     Refer this to know that how to test the model with scripts.
    │
    ├── data                            # Contains the dataset that used for model test and training.
    │                                     When test or training scripts run successfully, dataset will be automatically placed in it.
    │
    ├── quantized                          
    │   └── quantize_eval_model.pb      # Quantized model for evaluation.
    │
    └── float                             
        └── frozen.pb                   # Float-point frozen model, the input to the `vai_q_tensorflow`.
                                          The pb name of different models may be different.


#### Pytorch Model Directory Structure
For a Pytorch model, you should see the following directory structure:

    ├── code                            # Contains test and training code.  
    │                                                        
    │                                   
    ├── readme.md                       # Contains the environment requirements, data preprocess and model information.
    │                                     Refer this to know that how to test and train the model with scripts.
    │                                        
    ├── data                            # Contains the dataset that used for model test and training.
    │                                     When test or training scripts run successfully, dataset will be automatically placed in it.
    │
    ├── qat                             # Contains the QAT(Quantization Aware Training) results. 
    │                                     The accuracy of QAT result is better than direct quantization called PTQ. 
    │                                     Some models but not all provided QAT reference results, and only these models have qat folder. 
    │                                         
    ├── quantized                          
    │   ├── _int.pth                    # Quantized model.
    │   ├── quant_info.json             # Quantization steps of tensors got. Please keep it for evaluation of quantized model.
    │   ├── _int.py                     # Converted vai_q_pytorch format model.
    │   └── _int.xmodel                 # Deployed model. The name of different models may be different.
    │                                     For some models that support QAT you could find better quantization results in 'qat' folder. 
    │
    │
    └── float                           
        └── _int.pth                    # Trained float-point model. The pth name of different models may be different.
                                          Path and model name in test scripts could be modified according to actual situation.
        
**Note:** For more information on Vitis-AI Quantizer such as `vai_q_tensorflow` and `vai_q_pytorch`, please see the [Vitis AI User Guide](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai).


## Model Performance
All the models in the Model Zoo have been deployed on Xilinx hardware with [Vitis AI](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library](https://github.com/Xilinx/Vitis-AI/tree/master/examples/Vitis-AI-Library). The performance number including end-to-end throughput and latency for each model on various boards with different DPU configurations are listed in the following sections.

For more information about DPU, see [DPU IP Product Guide](https://www.xilinx.com/cgi-bin/docs/ipdoc?c=dpu;v=latest;d=pg338-dpu.pdf).

For RNN models such as NLP, please refer to [DPU-for-RNN](https://github.com/Xilinx/Vitis-AI/blob/master/demo/DPU-for-RNN) for dpu specification information.

Besides, for Transformer demos such as ViT, Bert-base you could refer to [Transformer](https://github.com/Xilinx/Vitis-AI/tree/master/examples/Transformer).

**Note:** The model performance number listed in the following sections is generated with Vitis AI v2.5 and Vitis AI Lirary v2.5. For different platforms, the different DPU configurations are used. Vitis AI and Vitis AI Library can be downloaded for free from [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library Github](https://github.com/Xilinx/Vitis-AI/tree/master/examples/Vitis-AI-Library).
We will continue to improve the performance with Vitis AI. The performance number reported here is subject to change in the near future.


### Performance on ZCU102 (0432055-05)
Measured with Vitis AI 2.5 and Vitis AI Library 2.5  

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-05)` board with a `3 * B4096  @ 281MHz` DPU configuration:


| No\. | Model                                | GPU Model Standard Name                               | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | bcc_pt                               | pt_BCC_shanghaitech_800_1000_268.9G                   | 3.31                                     | 10.86                                   |
| 2    | c2d2_lite_pt                         | pt_C2D2lite_CC20_512_512_6.86G                        | 2.84                                     | 5.24                                    |
| 3    | centerpoint                          | pt_centerpoint_astyx_2560_40_54G                      | 16.02                                    | 47.9                                    |
| 4    | CLOCs                                | pt_CLOCs_kitti_2.0                                    | 2.85                                     | 10.18                                   |
| 5    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 60.77                                    | 189.53                                  |
| 6    | efficientdet_d2_tf                   | tf_efficientdet-d2_coco_768_768_11.06G                | 3.10                                     | 6.05                                    |
| 7    | efficientnet-b0_tf2                  | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 78.02                                    | 152.61                                  |
| 8    | efficientNet-edgetpu-L_tf            | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 34.99                                    | 90.14                                   |
| 9    | efficientNet-edgetpu-M_tf            | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 79.88                                    | 205.22                                  |
| 10   | efficientNet-edgetpu-S_tf            | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 115.05                                   | 308.15                                  |
| 11   | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 10.10                                    | 36.43                                   |
| 12   | face_mask_detection_pt               | pt_face-mask-detection_512_512_0.59G                  | 115.80                                   | 401.08                                  |
| 13   | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 2988.30                                  | 8791.90                                 |
| 14   | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 168.33                                   | 337.68                                  |
| 15   | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 941.81                                   | 2275.70                                 |
| 16   | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 2240.00                                  | 6336.00                                 |
| 17   | fadnet                               | pt_fadnet_sceneflow_576_960_441G                      | 1.17                                     | 1.59                                    |
| 18   | fadnet_pruned                        | pt_fadnet_sceneflow_576_960_0.65_154G                 | 1.76                                     | 2.67                                    |
| 19   | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 22.59                                    | 66.03                                   |
| 20   | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 37.08                                    | 107.18                                  |
| 21   | HardNet_MSeg_pt                      | pt_HardNet_mixed_352_352_22.78G                       | 24.71                                    | 56.77                                   |
| 22   | HFnet_tf                             | tf_HFNet_mixed_960_960_20.09G                         | 3.57                                     | 15.90                                   |
| 23   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 23.89                                    | 51.93                                   |
| 24   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 191.26                                   | 470.88                                  |
| 25   | inception_v2_tf                      | tf_inceptionv2_imagenet_224_224_3.88G                 | 93.04                                    | 229.67                                  |
| 26   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 59.75                                    | 135.30                                  |
| 27   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 68.43                                    | 158.60                                  |
| 28   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 85.04                                    | 203.97                                  |
| 29   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 59.18                                    | 136.24                                  |
| 30   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 59.83                                    | 136.02                                  |
| 31   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 75.76                                    | 175.51                                  |
| 32   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 85.52                                    | 203.83                                  |
| 33   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 96.39                                    | 234.92                                  |
| 34   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 114.10                                   | 283.68                                  |
| 35   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 28.83                                    | 68.57                                   |
| 36   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 154.32                                   | 393.59                                  |
| 37   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 1.87                                     | 7.12                                    |
| 38   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 78.85                                    | 173.51                                  |
| 39   | mobilenet_edge_0_75_tf               | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 262.66                                   | 720.24                                  |
| 40   | mobilenet_edge_1_0_tf                | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 214.78                                   | 554.55                                  |
| 41   | mobilenet_1_0_224_tf2                | tf2_mobilenetv1_imagenet_224_224_1.15G                | 321.47                                   | 939.61                                  |
| 42   | mobilenet_v1_0_25_128_tf             | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 1332.90                                  | 4754.20                                 |
| 43   | mobilenet_v1_0_5_160_tf              | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 914.84                                   | 3116.80                                 |
| 44   | mobilenet_v1_1_0_224_tf              | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 326.62                                   | 951.00                                  |
| 45   | mobilenet_v1_1_0_224_pruned_0_11_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G        | 329.61                                   | 879.01                                  |
| 46   | mobilenet_v1_1_0_224_pruned_0_12_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G           | 335.60                                   | 934.39                                  |
| 47   | mobilenet_v2_1_0_224_tf              | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 268.07                                   | 707.03                                  |
| 48   | mobilenet_v2_1_4_224_tf              | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 191.08                                   | 473.15                                  |
| 49   | mobilenet_v2_cityscapes_tf           | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 1.75                                     | 5.28                                    |
| 50   | mobilenet_v3_small_1_0_tf2           | tf2_mobilenetv3_imagenet_224_224_132M                 | 336.62                                   | 965.76                                  |
| 51   | movenet_ntd_pt                       | pt_movenet_coco_192_192_0.5G                          | 94.72                                    | 390.02                                  |
| 52   | MT-resnet18_mixed_pt                 | pt_MT-resnet18_mixed_320_512_13.65G                   | 32.79                                    | 102.96                                  |
| 53   | multi_task_v3_pt                     | pt_multitaskv3_mixed_320_512_25.44G                   | 17.10                                    | 61.35                                   |
| 54   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 1.10                                     | 3.41                                    |
| 55   | ofa_depthwise_res50_pt               | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 106.03                                   | 370.18                                  |
| 56   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 17.00                                    | 28.03                                   |
| 57   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 47.64                                    | 93.90                                   |
| 58   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 72.22                                    | 144.17                                  |
| 59   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 88.77                                    | 162.01                                  |
| 60   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 129.44                                   | 258.29                                  |
| 61   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 183.81                                   | 354.73                                  |
| 62   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 16.87                                    | 42.75                                   |
| 63   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 21.53                                    | 54.59                                   |
| 64   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 27.71                                    | 71.35                                   |
| 65   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 661.42                                   | 1428.2                                  |
| 66   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 107.54                                   | 237.66                                  |
| 67   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 118.42                                   | 300.87                                  |
| 68   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 126.01                                   | 320.82                                  |
| 69   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 136.24                                   | 362.63                                  |
| 70   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 144.01                                   | 376.18                                  |
| 71   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 370.36                                   | 690.85                                  |
| 72   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 151.69                                   | 366.62                                  |
| 73   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G_2.5                    | 1.28                                     | 4.25                                    |
| 74   | pointpillars_kitti                   | pt_pointpillars_kitti_12000_100_10.8G                 | 19.63                                    | 49.20                                   |
| 75   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 2.23                                     | 9.61                                    |
| 76   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 8.61                                     | 18.05                                   |
| 77   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 11.31                                    | 34.50                                   |
| 78   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 12.04                                    | 34.90                                   |
| 79   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 22.47                                    | 68.28                                   |
| 80   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 39.53                                    | 125.60                                  |
| 81   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 59.57                                    | 197.08                                  |
| 82   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 67.62                                    | 229.37                                  |
| 83   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 87.95                                    | 191.05                                  |
| 84   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 109.45                                   | 213.71                                  |
| 85   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 138.81                                   | 276.33                                  |
| 86   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 46.38                                    | 110.72                                  |
| 87   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 31.64                                    | 77.23                                   |
| 88   | resnet_v2_50_tf                      | tf_resnetv2_50_imagenet_299_299_13.1G                 | 45.05                                    | 99.64                                   |
| 89   | resnet_v2_101_tf                     | tf_resnetv2_101_imagenet_299_299_26.78G               | 23.60                                    | 55.60                                   |
| 90   | resnet_v2_152_tf                     | tf_resnetv2_152_imagenet_299_299_40.47G               | 16.07                                    | 38.18                                   |
| 91   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 87.33                                    | 192.94                                  |
| 92   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 77.58                                    | 173.16                                  |
| 93   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 90.97                                    | 179.67                                  |
| 94   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 96.45                                    | 190.87                                  |
| 95   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 105.13                                   | 203.14                                  |
| 96   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 118.47                                   | 228.31                                  |
| 97   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 129.01                                   | 251.31                                  |
| 98   | SA_gate_base_pt                      | pt_sa-gate_NYUv2_360_360_178G                         | 3.29                                     | 9.45                                    |
| 99   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 5.61                                     | 21.28                                   |
| 100  | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 4.15                                     | 10.99                                   |
| 101  | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 7.44                                     | 23.88                                   |
| 102  | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 35.49                                    | 162.79                                  |
| 102  | SemanticFPN_Mobilenetv2_pt           | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 10.52                                    | 52.46                                   |
| 104  | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 88.30                                    | 140.63                                  |
| 105  | solo_pt                              | pt_SOLO_coco_640_640_107G                             | 1.45                                     | 4.84                                    |
| 106  | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 575.41                                   | 1499.60                                 |
| 107  | ssd_inception_v2_coco_tf             | tf_ssdinceptionv2_coco_300_300_9.62G                  | 39.41                                    | 102.23                                  |
| 108  | ssd_mobilenet_v1_coco_tf             | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 111.06                                   | 332.23                                  |
| 109  | ssd_mobilenet_v2_coco_tf             | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 81.77                                    | 213.25                                  |
| 110  | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 2.92                                     | 5.17                                    |
| 111  | ssdlite_mobilenet_v2_coco_tf         | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 106.06                                   | 304.19                                  |
| 112  | ssr_pt                               | pt_SSR_CVC_256_256_39.72G                             | 6.03                                     | 14.42                                   |
| 113  | superpoint_tf                        | tf_superpoint_mixed_480_640_52.4G                     | 12.61                                    | 53.79                                   |
| 114  | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 1.67                                     | 4.68                                    |
| 115  | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 13.13                                    | 33.70                                   |
| 116  | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 35.08                                    | 95.82                                   |
| 117  | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 22.80                                    | 69.63                                   |
| 118  | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 204.70                                   | 499.15                                  |
| 119  | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 203.44                                   | 498.51                                  |
| 120  | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 205.03                                   | 499.64                                  |
| 121  | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 20.15                                    | 40.94                                   |
| 122  | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 43.51                                    | 105.48                                  |
| 123  | vgg_16_pruned_0_5_tf                 | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 47.23                                    | 116.24                                  |
| 124  | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 17.37                                    | 36.47                                   |
| 125  | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 13.55                                    | 35.12                                   |
| 126  | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 13.24                                    | 34.87                                   |
| 127  | yolov4_416_tf                        | tf_yolov4_coco_416_416_60.3G                          | 13.52                                    | 33.96                                   |
| 128  | yolov4_512_tf                        | tf_yolov4_coco_512_512_91.2G                          | 10.25                                    | 25.24                                   |


</details>


### Performance on ZCU104
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 300MHz` DPU configuration:


| No\. | Model                                | GPU Model Standard Name                               | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | bcc_pt                               | pt_BCC_shanghaitech_800_1000_268.9G                   | 3.51                                     | 7.98                                    |
| 2    | c2d2_lite_pt                         | pt_C2D2lite_CC20_512_512_6.86G                        | 3.13                                     | 3.55                                    |
| 3    | centerpoint                          | pt_centerpoint_astyx_2560_40_54G                      | 16.84                                    | 20.72                                   |
| 4    | CLOCs                                | pt_CLOCs_kitti_2.0                                    | 2.89                                     | 10.24                                   |
| 5    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 63.75                                    | 152.73                                  |
| 6    | efficientdet_d2_tf                   | tf_efficientdet-d2_coco_768_768_11.06G                | 63.75                                    | 152.73                                  |
| 7    | efficientnet-b0_tf2                  | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 83.77                                    | 146.26                                  |
| 8    | efficientNet-edgetpu-L_tf            | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 37.33                                    | 73.44                                   |
| 9    | efficientNet-edgetpu-M_tf            | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 85.42                                    | 168.85                                  |
| 10   | efficientNet-edgetpu-S_tf            | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 122.86                                   | 248.87                                  |
| 11   | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 10.44                                    | 39.45                                   |
| 12   | face_mask_detection_pt               | pt_face-mask-detection_512_512_0.59G                  | 120.72                                   | 326.94                                  |
| 13   | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 3050.46                                  | 8471.64                                 |
| 14   | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 179.71                                   | 314.42                                  |
| 15   | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 992.52                                   | 2122.75                                 |
| 16   | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 2304.48                                  | 5927.11                                 |
| 17   | fadnet                               | pt_fadnet_sceneflow_576_960_441G                      | 1.66                                     | 3.46                                    |
| 18   | fadnet_pruned                        | pt_fadnet_sceneflow_576_960_0.65_154G                 | 2.65                                     | 5.28                                    |
| 19   | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 23.83                                    | 52.39                                   |
| 20   | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 39.44                                    | 81.02                                   |
| 21   | HardNet_MSeg_pt                      | pt_HardNet_mixed_352_352_22.78G                       | 26.42                                    | 49.36                                   |
| 22   | HFnet_tf                             | tf_HFNet_mixed_960_960_20.09G                         | 3.21                                     | 14.39                                   |
| 23   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 25.54                                    | 47.05                                   |
| 24   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 202.49                                   | 412.79                                  |
| 25   | inception_v2_tf                      | tf_inceptionv2_imagenet_224_224_3.88G                 | 98.99                                    | 195.15                                  |
| 26   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 63.43                                    | 121.49                                  |
| 27   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 72.73                                    | 141.54                                  |
| 28   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 90.22                                    | 180.10                                  |
| 29   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 62.84                                    | 121.59                                  |
| 30   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 63.53                                    | 122.11                                  |
| 31   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 80.35                                    | 157.27                                  |
| 32   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 90.53                                    | 180.34                                  |
| 33   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 101.89                                   | 206.08                                  |
| 34   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 120.56                                   | 248.18                                  |
| 35   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 30.72                                    | 59.03                                   |
| 36   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 163.55                                   | 338.55                                  |
| 37   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 1.87                                     | 5.22                                    |
| 38   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 84.21                                    | 157.65                                  |
| 39   | mobilenet_edge_0_75_tf               | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 278.38                                   | 604.87                                  |
| 40   | mobilenet_edge_1_0_tf                | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 228.03                                   | 476.52                                  |
| 41   | mobilenet_1_0_224_tf2                | tf2_mobilenetv1_imagenet_224_224_1.15G                | 339.56                                   | 770.94                                  |
| 42   | mobilenet_v1_0_25_128_tf             | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 1375.03                                  | 4263.73                                 |
| 43   | mobilenet_v1_0_5_160_tf              | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 950.02                                   | 2614.83                                 |
| 44   | mobilenet_v1_1_0_224_tf              | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 344.85                                   | 783.96                                  |
| 45   | mobilenet_v1_1_0_224_pruned_0_11_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G        | 351.57                                   | 775.26                                  |
| 46   | mobilenet_v1_1_0_224_pruned_0_12_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G           | 356.01                                   | 801.88                                  |
| 47   | mobilenet_v2_1_0_224_tf              | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 284.18                                   | 609.18                                  |
| 48   | mobilenet_v2_1_4_224_tf              | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 203.20                                   | 413.59                                  |
| 49   | mobilenet_v2_cityscapes_tf           | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 1.81                                     | 5.53                                    |
| 50   | mobilenet_v3_small_1_0_tf2           | tf2_mobilenetv3_imagenet_224_224_132M                 | 363.63                                   | 819.63                                  |
| 51   | movenet_ntd_pt                       | pt_movenet_coco_192_192_0.5G                          | 95.38                                    | 362.92                                  |
| 52   | MT-resnet18_mixed_pt                 | pt_MT-resnet18_mixed_320_512_13.65G                   | 34.31                                    | 93.07                                   |
| 53   | multi_task_v3_pt                     | pt_multitaskv3_mixed_320_512_25.44G                   | 17.76                                    | 55.32                                   |
| 54   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 1.04                                     | 2.56                                    |
| 55   | ofa_depthwise_res50_pt               | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 107.8                                    | 343.72                                  |
| 56   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 17.93                                    | 28.87                                   |
| 57   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 50.90                                    | 90.48                                   |
| 58   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 77.23                                    | 138.28                                  |
| 59   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 95.29                                    | 163.25                                  |
| 60   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 138.44                                   | 248.57                                  |
| 61   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 198.24                                   | 354.63                                  |
| 62   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 17.89                                    | 37.46                                   |
| 63   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 22.75                                    | 48.69                                   |
| 64   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 29.17                                    | 64.39                                   |
| 65   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 700.79                                   | 1369.03                                 |
| 66   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 114.82                                   | 216.34                                  |
| 67   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 124.41                                   | 277.83                                  |
| 68   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 132.06                                   | 298.51                                  |
| 69   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 142.58                                   | 333.84                                  |
| 70   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 150.03                                   | 352.19                                  |
| 71   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 395.46                                   | 700.55                                  |
| 72   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 161.53                                   | 319.21                                  |
| 73   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G_2.5                    | 1.31                                     | 4.51                                    |
| 74   | pointpillars_kitti                   | pt_pointpillars_kitti_12000_100_10.8G                 | 20.13                                    | 49.83                                   |
| 75   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 2.28                                     | 8.91                                    |
| 76   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 9.17                                     | 17.15                                   |
| 77   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 10.8                                     | 25.87                                   |
| 78   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 12.82                                    | 26.09                                   |
| 79   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 23.90                                    | 50.30                                   |
| 80   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 41.89                                    | 92.30                                   |
| 81   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 62.86                                    | 145.99                                  |
| 82   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 71.3                                     | 169.42                                  |
| 83   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 93.84                                    | 175                                     |
| 84   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 117.80                                   | 209.94                                  |
| 85   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 149.77                                   | 270.73                                  |
| 86   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 49.54                                    | 94.71                                   |
| 87   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 33.81                                    | 64.91                                   |
| 88   | resnet_v2_50_tf                      | tf_resnetv2_50_imagenet_299_299_13.1G                 | 48.10                                    | 90.57                                   |
| 89   | resnet_v2_101_tf                     | tf_resnetv2_101_imagenet_299_299_26.78G               | 25.23                                    | 47.89                                   |
| 90   | resnet_v2_152_tf                     | tf_resnetv2_152_imagenet_299_299_40.47G               | 17.18                                    | 32.61                                   |
| 91   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 93.23                                    | 175.39                                  |
| 92   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 83.56                                    | 157.2                                   |
| 93   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 97.59                                    | 175.01                                  |
| 94   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 103.34                                   | 185.46                                  |
| 95   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 112.27                                   | 199.57                                  |
| 96   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 126.82                                   | 225.88                                  |
| 97   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 137.70                                   | 247.08                                  |
| 98   | SA_gate_base_pt                      | pt_sa-gate_NYUv2_360_360_178G                         | 3.46                                     | 8.52                                    |
| 99   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 5.68                                     | 21.45                                   |
| 100  | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 4.30                                     | 11.74                                   |
| 101  | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 7.68                                     | 25.54                                   |
| 102  | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 36.37                                    | 149.18                                  |
| 103  | SemanticFPN_Mobilenetv2_pt           | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 10.73                                    | 51.97                                   |
| 104  | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 94.66                                    | 146.02                                  |
| 105  | solo_pt                              | pt_SOLO_coco_640_640_107G                             | 1.47                                     | 4.82                                    |
| 106  | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 599.88                                   | 1315.93                                 |
| 107  | ssd_inception_v2_coco_tf             | tf_ssdinceptionv2_coco_300_300_9.62G                  | 41.64                                    | 87.86                                   |
| 108  | ssd_mobilenet_v1_coco_tf             | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 115.34                                   | 307.17                                  |
| 109  | ssd_mobilenet_v2_coco_tf             | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 85.56                                    | 196.25                                  |
| 110  | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 2.91                                     | 5.22                                    |
| 111  | ssdlite_mobilenet_v2_coco_tf         | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 110.49                                   | 280.15                                  |
| 112  | ssr_pt                               | pt_SSR_CVC_256_256_39.72G                             | 6.46                                     | 12.37                                   |
| 113  | superpoint_tf                        | tf_superpoint_mixed_480_640_52.4G                     | 11.11                                    | 40.32                                   |
| 114  | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 1.77                                     | 3.68                                    |
| 115  | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 13.99                                    | 27.78                                   |
| 116  | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 37.25                                    | 78.39                                   |
| 117  | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 23.63                                    | 59.35                                   |
| 118  | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 217.68                                   | 437.02                                  |
| 119  | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 216.22                                   | 435.90                                  |
| 120  | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 218.02                                   | 437.32                                  |
| 121  | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 21.49                                    | 37.10                                   |
| 122  | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 46.41                                    | 88.40                                   |
| 123  | vgg_16_pruned_0_5_tf                 | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 50.44                                    | 96.87                                   |
| 124  | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 18.54                                    | 32.69                                   |
| 125  | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 14.44                                    | 28.82                                   |
| 126  | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 14.09                                    | 28.64                                   |
| 127  | yolov4_416_tf                        | tf_yolov4_coco_416_416_60.3G                          | 14.36                                    | 28.89                                   |
| 128  | yolov4_512_tf                        | tf_yolov4_coco_512_512_91.2G                          | 10.93                                    | 21.76                                   |


</details>


### Performance on VCK190
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Versal` board with 192 AIEs running at 1250 MHz:
  

| No\. | Model                                | GPU Model Standard Name                               | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | bcc_pt                               | pt_BCC_shanghaitech_800_1000_268.9G                   | 37.01                                    | 72.51                                   |
| 2    | c2d2_lite_pt                         | pt_C2D2lite_CC20_512_512_6.86G                        | 20.57                                    | 26.49                                   |
| 3    | centerpoint                          | pt_centerpoint_astyx_2560_40_54G                      | 128.12                                   | 236.99                                  |
| 4    | CLOCs                                | pt_CLOCs_kitti_2.0                                    | 8.11                                     | 14.63                                   |
| 5    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 256.13                                   | 459.87                                  |
| 6    | efficientdet_d2_tf                   | tf_efficientdet-d2_coco_768_768_11.06G                | 18.1                                     | 40.38                                   |
| 7    | efficientnet-b0_tf2                  | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 1090.29                                  | 1811.8                                  |
| 8    | efficientNet-edgetpu-L_tf            | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 397.35                                   | 513.32                                  |
| 9    | efficientNet-edgetpu-M_tf            | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 889.46                                   | 1394.48                                 |
| 10   | efficientNet-edgetpu-S_tf            | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 1209.7                                   | 2192.61                                 |
| 11   | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 25.64                                    | 54.58                                   |
| 12   | face_mask_detection_pt               | pt_face-mask-detection_512_512_0.59G                  | 451.11                                   | 914.73                                  |
| 13   | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 13746.3                                  | 29425.4                                 |
| 14   | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 3849.25                                  | 5691.74                                 |
| 15   | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 7574.57                                  | 19419.9                                 |
| 16   | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 11369.1                                  | 26161.2                                 |
| 17   | fadnet                               | pt_fadnet_sceneflow_576_960_441G                      | 8.02                                     | 13.46                                   |
| 18   | fadnet_pruned                        | pt_fadnet_sceneflow_576_960_0.65_154G                 | 9.01                                     | 16.24                                   |
| 19   | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 194.87                                   | 374.48                                  |
| 20   | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 481.49                                   | 788.38                                  |
| 21   | HardNet_MSeg_pt                      | pt_HardNet_mixed_352_352_22.78G                       | 205.76                                   | 274.23                                  |
| 22   | HFnet_tf                             | tf_HFNet_mixed_960_960_20.09G                         | 9.37                                     | 22.34                                   |
| 23   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 387.99                                   | 499.77                                  |
| 24   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 1303.24                                  | 2458.23                                 |
| 25   | inception_v2_tf                      | tf_inceptionv2_imagenet_224_224_3.88G                 | 827.14                                   | 1190.76                                 |
| 26   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 595.41                                   | 903.47                                  |
| 27   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 629.90                                   | 985.47                                  |
| 28   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 687.25                                   | 1130.21                                 |
| 29   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 657.37                                   | 1061.04                                 |
| 30   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 592.62                                   | 896.70                                  |
| 31   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 653.31                                   | 1052.53                                 |
| 32   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 683.24                                   | 1125.47                                 |
| 33   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 726.47                                   | 1242.57                                 |
| 34   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 787.76                                   | 1430.62                                 |
| 35   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 341.84                                   | 424.03                                  |
| 36   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 1528.81                                  | 3034.36                                 |
| 37   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 11.94                                    | 20.92                                   |
| 38   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 1367.1                                   | 2744.24                                 |
| 39   | mobilenet_edge_0_75_tf               | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 1909.12                                  | 4995.3                                  |
| 40   | mobilenet_edge_1_0_tf                | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 1839.19                                  | 4879.14                                 |
| 41   | mobilenet_1_0_224_tf2                | tf2_mobilenetv1_imagenet_224_224_1.15G                | 2026.52                                  | 5049.85                                 |
| 42   | mobilenet_v1_0_25_128_tf             | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 5019.74                                  | 10348.4                                 |
| 43   | mobilenet_v1_0_5_160_tf              | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 3604.67                                  | 7997.22                                 |
| 44   | mobilenet_v1_1_0_224_tf              | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 2022.61                                  | 5015.41                                 |
| 45   | mobilenet_v1_1_0_224_pruned_0_11_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G        | 2034.76                                  | 5013.31                                 |
| 46   | mobilenet_v1_1_0_224_pruned_0_12_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G           | 2025.73                                  | 4957.45                                 |
| 47   | mobilenet_v2_1_0_224_tf              | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 1891.80                                  | 4927.03                                 |
| 48   | mobilenet_v2_1_4_224_tf              | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 1640.06                                  | 4189.87                                 |
| 49   | mobilenet_v2_cityscapes_tf           | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 5.33                                     | 12.03                                   |
| 50   | mobilenet_v3_small_1_0_tf2           | tf2_mobilenetv3_imagenet_224_224_132M                 | 2052.58                                  | 4977.33                                 |
| 51   | movenet_ntd_pt                       | pt_movenet_coco_192_192_0.5G                          | 245.96                                   | 445.52                                  |
| 52   | MT-resnet18_mixed_pt                 | pt_MT-resnet18_mixed_320_512_13.65G                   | 144.05                                   | 261.02                                  |
| 53   | multi_task_v3_pt                     | pt_multitaskv3_mixed_320_512_25.44G                   | 77.73                                    | 173.24                                  |
| 54   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 8.67                                     | 18.67                                   |
| 55   | ofa_depthwise_res50_pt               | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 306.23                                   | 454.3                                   |
| 56   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 60.15                                    | 81.08                                   |
| 57   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 845.17                                   | 1225.92                                 |
| 58   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 1007.54                                  | 1602.95                                 |
| 59   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 1146.16                                  | 1982.49                                 |
| 60   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 1548.71                                  | 2848.63                                 |
| 61   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 2082.78                                  | 4027.03                                 |
| 62   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 128.26                                   | 223.24                                  |
| 63   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 146.2                                    | 255.8                                   |
| 64   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 167.96                                   | 297.74                                  |
| 65   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 5617.05                                  | 12583.8                                 |
| 66   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 1822.71                                  | 3816.15                                 |
| 67   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 1157.63                                  | 2238.36                                 |
| 68   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 1178.59                                  | 2239.07                                 |
| 69   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 1211.16                                  | 2234.36                                 |
| 70   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 1240.68                                  | 2228.85                                 |
| 71   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 4173.59                                  | 8691.82                                 |
| 72   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 1779.41                                  | 3635.07                                 |
| 73   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G_2.5                    | 3.8                                      | 6.75                                    |
| 74   | pointpillars_kitti                   | pt_pointpillars_kitti_12000_100_10.8G                 | 24.6                                     | 35.29                                   |
| 75   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 7.65                                     | 15.97                                   |
| 76   | psmnet                               | pt_psmnet_sceneflow_576_960_0.68_696G                 | 0.36                                     | 0.71                                    |
| 77   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 45.29                                    | 56.99                                   |
| 78   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 101.95                                   | 224.59                                  |
| 79   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 232.24                                   | 315.20                                  |
| 80   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 342.68                                   | 563.85                                  |
| 81   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 444.29                                   | 898.26                                  |
| 82   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 532.85                                   | 1259.42                                 |
| 83   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 502.46                                   | 1282.02                                 |
| 84   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 1452.5                                   | 3054.88                                 |
| 85   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 1531.45                                  | 3239.42                                 |
| 86   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 1744.88                                  | 4601.93                                 |
| 87   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 1063.41                                  | 1756.03                                 |
| 88   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 838.21                                   | 1215.98                                 |
| 89   | resnet_v2_50_tf                      | tf_resnetv2_50_imagenet_299_299_13.1G                 | 549.19                                   | 794.83                                  |
| 90   | resnet_v2_101_tf                     | tf_resnetv2_101_imagenet_299_299_26.78G               | 413.38                                   | 538.87                                  |
| 91   | resnet_v2_152_tf                     | tf_resnetv2_152_imagenet_299_299_40.47G               | 331.27                                   | 407.21                                  |
| 92   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 1462.41                                  | 3095.02                                 |
| 93   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 1374.31                                  | 2773.11                                 |
| 94   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 1414.68                                  | 2942.34                                 |
| 95   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 1470.40                                  | 3173.66                                 |
| 96   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 1533.74                                  | 3343.16                                 |
| 97   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 1612.50                                  | 3883.51                                 |
| 98   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 1698.97                                  | 4525.27                                 |
| 99   | SA_gate_base_pt                      | pt_sa-gate_NYUv2_360_360_178G                         | 7.61                                     | 9.62                                    |
| 100  | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 12.27                                    | 24.06                                   |
| 101  | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 10.61                                    | 22.3                                    |
| 102  | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 20.36                                    | 47.78                                   |
| 103  | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 110.51                                   | 224.58                                  |
| 104  | SemanticFPN_Mobilenetv2_pt           | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 27.15                                    | 55.92                                   |
| 105  | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 343.09                                   | 639.38                                  |
| 106  | solo_pt                              | pt_SOLO_coco_640_640_107G                             | 4.36                                     | 7.47                                    |
| 107  | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 3143.04                                  | 5797.62                                 |
| 108  | ssd_inception_v2_coco_tf             | tf_ssdinceptionv2_coco_300_300_9.62G                  | 273.59                                   | 381.22                                  |
| 109  | ssd_mobilenet_v1_coco_tf             | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 433.55                                   | 517.66                                  |
| 110  | ssd_mobilenet_v2_coco_tf             | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 387.64                                   | 508.52                                  |
| 111  | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 11.2                                     | 12.19                                   |
| 112  | ssdlite_mobilenet_v2_coco_tf         | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 408.27                                   | 523.56                                  |
| 113  | ssr_pt                               | pt_SSR_CVC_256_256_39.72G                             | 74.48                                    | 78.02                                   |
| 114  | superpoint_tf                        | tf_superpoint_mixed_480_640_52.4G                     | 49.86                                    | 106.38                                  |
| 115  | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 20.92                                    | 29.6                                    |
| 116  | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 132.05                                   | 193.27                                  |
| 117  | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 392.04                                   | 962.59                                  |
| 118  | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 80.52                                    | 242.28                                  |
| 119  | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 2036.82                                  | 5191.25                                 |
| 120  | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 1986.11                                  | 5191.13                                 |
| 121  | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 2053.59                                  | 5251.65                                 |
| 122  | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 530.06                                   | 655.44                                  |
| 123  | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 863.92                                   | 1261.41                                 |
| 124  | vgg_16_pruned_0_5_tf                 | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 904.06                                   | 1355.58                                 |
| 125  | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 480.75                                   | 582.57                                  |
| 126  | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 192.68                                   | 292.35                                  |
| 127  | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 218.53                                   | 291.4                                   |
| 128  | yolov4_416_tf                        | tf_yolov4_coco_416_416_60.3G                          | 141.81                                   | 214.67                                  |
| 129  | yolov4_512_tf                        | tf_yolov4_coco_512_512_91.2G                          | 105.09                                   | 154.65                                  |


</details>


### Performance on Kria KV260 SOM
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Kria KV260` board with a `1 * B4096F  @ 300MHz` DPU configuration:
  

| No\. | Model                                | GPU Model Standard Name                               | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | bcc_pt                               | pt_BCC_shanghaitech_800_1000_268.9G                   | 3.55                                     | 4.01                                    |
| 2    | c2d2_lite_pt                         | pt_C2D2lite_CC20_512_512_6.86G                        | 3.24                                     | 3.46                                    |
| 3    | centerpoint                          | pt_centerpoint_astyx_2560_40_54G                      | 17.21                                    | 20.2                                    |
| 4    | CLOCs                                | pt_CLOCs_kitti_2.0                                    | 3.08                                     | 9.05                                    |
| 5    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 65.17                                    | 78.01                                   |
| 6    | efficientdet_d2_tf                   | tf_efficientdet-d2_coco_768_768_11.06G                | 3.29                                     | 4.24                                    |
| 7    | efficientnet-b0_tf2                  | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 84.26                                    | 87.73                                   |
| 8    | efficientNet-edgetpu-L_tf            | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 37.52                                    | 38.64                                   |
| 9    | efficientNet-edgetpu-M_tf            | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 86.03                                    | 90.05                                   |
| 10   | efficientNet-edgetpu-S_tf            | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 124.06                                   | 131.85                                  |
| 11   | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 11.18                                    | 30.00                                   |
| 12   | face_mask_detection_pt               | pt_face-mask-detection_512_512_0.59G                  | 124.23                                   | 170.38                                  |
| 13   | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 3316.38                                  | 5494.16                                 |
| 14   | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 179.9                                    | 184.56                                  |
| 15   | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 1021.38                                  | 1134.19                                 |
| 16   | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 2472.62                                  | 3560.3                                  |
| 17   | fadnet                               | pt_fadnet_sceneflow_576_960_441G                      | 1.67                                     | 2.26                                    |
| 18   | fadnet_pruned                        | pt_fadnet_sceneflow_576_960_0.65_154G                 | 2.78                                     | 4.29                                    |
| 19   | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 24.17                                    | 27.04                                   |
| 20   | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 39.64                                    | 41.25                                   |
| 21   | HardNet_MSeg_pt                      | pt_HardNet_mixed_352_352_22.78G                       | 26.66                                    | 27.63                                   |
| 22   | HFnet_tf                             | tf_HFNet_mixed_960_960_20.09G                         | 3.39                                     | 11.24                                   |
| 23   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 25.65                                    | 26.19                                   |
| 24   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 205.17                                   | 227.17                                  |
| 25   | inception_v2_tf                      | tf_inceptionv2_imagenet_224_224_3.88G                 | 99.05                                    | 104.45                                  |
| 26   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 63.91                                    | 67.38                                   |
| 27   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 73.34                                    | 77.93                                   |
| 28   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 91.19                                    | 98.33                                   |
| 29   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 63.34                                    | 66.68                                   |
| 30   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 64.03                                    | 67.46                                   |
| 31   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 80.72                                    | 86.79                                   |
| 32   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 91.55                                    | 98.71                                   |
| 33   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 103.26                                   | 112.6                                   |
| 34   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 121.92                                   | 135.54                                  |
| 35   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 30.83                                    | 31.6                                    |
| 36   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 165.01                                   | 179.4                                   |
| 37   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 1.91                                     | 2.64                                    |
| 38   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 84.83                                    | 88.69                                   |
| 39   | mobilenet_edge_0_75_tf               | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 282.1                                    | 328.78                                  |
| 40   | mobilenet_edge_1_0_tf                | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 232.17                                   | 260.7                                   |
| 41   | mobilenet_1_0_224_tf2                | tf2_mobilenetv1_imagenet_224_224_1.15G                | 347.97                                   | 416.6                                   |
| 42   | mobilenet_v1_0_25_128_tf             | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 1470.42                                  | 2428.88                                 |
| 43   | mobilenet_v1_0_5_160_tf              | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 1002.62                                  | 1492.12                                 |
| 44   | mobilenet_v1_1_0_224_tf              | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 354.19                                   | 425.57                                  |
| 45   | mobilenet_v1_1_0_224_pruned_0_11_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G        | 361.27                                   | 437.31                                  |
| 46   | mobilenet_v1_1_0_224_pruned_0_12_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G           | 364.45                                   | 441.98                                  |
| 47   | mobilenet_v2_1_0_224_tf              | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 290.62                                   | 336.82                                  |
| 48   | mobilenet_v2_1_4_224_tf              | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 206.04                                   | 228.2                                   |
| 49   | mobilenet_v2_cityscapes_tf           | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 1.90                                     | 3.32                                    |
| 50   | mobilenet_v3_small_1_0_tf2           | tf2_mobilenetv3_imagenet_224_224_132M                 | 374.47                                   | 453.97                                  |
| 51   | movenet_ntd_pt                       | pt_movenet_coco_192_192_0.5G                          | 102.8                                    | 351.28                                  |
| 52   | MT-resnet18_mixed_pt                 | pt_MT-resnet18_mixed_320_512_13.65G                   | 35.45                                    | 50.52                                   |
| 53   | multi_task_v3_pt                     | pt_multitaskv3_mixed_320_512_25.44G                   | 18.57                                    | 29.95                                   |
| 54   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 1.06                                     | 1.29                                    |
| 55   | ofa_depthwise_res50_pt               | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 116.42                                   | 264.3                                   |
| 56   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 18.10                                    | 19.27                                   |
| 57   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 51.2                                     | 52.49                                   |
| 58   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 77.6                                     | 80.94                                   |
| 59   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 95.68                                    | 100.35                                  |
| 60   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 139.85                                   | 148.19                                  |
| 61   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 199.09                                   | 214.92                                  |
| 62   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 17.95                                    | 18.89                                   |
| 63   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 22.98                                    | 26.44                                   |
| 64   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 29.19                                    | 35.48                                   |
| 65   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 712.72                                   | 804.15                                  |
| 66   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 115.82                                   | 122.43                                  |
| 67   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 127.16                                   | 159.8                                   |
| 68   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 135.88                                   | 173.97                                  |
| 69   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 145.46                                   | 190.93                                  |
| 70   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 154.51                                   | 206.66                                  |
| 71   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 399.80                                   | 436.59                                  |
| 72   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 162.79                                   | 173.08                                  |
| 73   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G_2.5                    | 1.34                                     | 2.78                                    |
| 74   | pointpillars_kitti                   | pt_pointpillars_kitti_12000_100_10.8G                 | 20.81                                    | 32.24                                   |
| 75   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 2.30                                     | 5.46                                    |
| 76   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 9.21                                     | 9.5                                     |
| 77   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 10.90                                    | 13.11                                   |
| 78   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 12.86                                    | 13.21                                   |
| 79   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 24.04                                    | 25.31                                   |
| 80   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 42.23                                    | 46.49                                   |
| 81   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 63.87                                    | 73.62                                   |
| 82   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 71.75                                    | 85.60                                   |
| 83   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 94.21                                    | 99.01                                   |
| 84   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 119.45                                   | 126.55                                  |
| 85   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 151.27                                   | 162.96                                  |
| 86   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 49.86                                    | 51.07                                   |
| 87   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 33.98                                    | 34.54                                   |
| 88   | resnet_v2_50_tf                      | tf_resnetv2_50_imagenet_299_299_13.1G                 | 48.42                                    | 50.46                                   |
| 89   | resnet_v2_101_tf                     | tf_resnetv2_101_imagenet_299_299_26.78G               | 25.09                                    | 25.87                                   |
| 90   | resnet_v2_152_tf                     | tf_resnetv2_152_imagenet_299_299_40.47G               | 17.25                                    | 17.48                                   |
| 91   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 93.80                                    | 98.21                                   |
| 92   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 84.44                                    | 87.96                                   |
| 93   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 98.92                                    | 103.79                                  |
| 94   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 104.15                                   | 109.58                                  |
| 95   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 113.71                                   | 120.24                                  |
| 96   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 127.95                                   | 136.11                                  |
| 97   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 139.55                                   | 149.48                                  |
| 98   | SA_gate_base_pt                      | pt_sa-gate_NYUv2_360_360_178G                         | 3.55                                     | 4.64                                    |
| 99   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 6.11                                     | 19.55                                   |
| 100  | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 4.55                                     | 11.46                                   |
| 101  | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 8.11                                     | 15.38                                   |
| 102  | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 37.36                                    | 86.73                                   |
| 103  | SemanticFPN_Mobilenetv2_pt           | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 11.52                                    | 33.01                                   |
| 104  | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 95.84                                    | 105.55                                  |
| 105  | solo_pt                              | pt_SOLO_coco_640_640_107G                             | 1.46                                     | 4.44                                    |
| 106  | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 619.72                                   | 762.62                                  |
| 107  | ssd_inception_v2_coco_tf             | tf_ssdinceptionv2_coco_300_300_9.62G                  | 42.06                                    | 47.23                                   |
| 108  | ssd_mobilenet_v1_coco_tf             | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 119.11                                   | 172.23                                  |
| 109  | ssd_mobilenet_v2_coco_tf             | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 87.71                                    | 112.72                                  |
| 110  | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 2.96                                     | 5.26                                    |
| 111  | ssdlite_mobilenet_v2_coco_tf         | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 111.50                                   | 161.41                                  |
| 112  | ssr_pt                               | pt_SSR_CVC_256_256_39.72G                             | 6.47                                     | 6.50                                    |
| 113  | superpoint_tf                        | tf_superpoint_mixed_480_640_52.4G                     | 11.46                                    | 20.57                                   |
| 114  | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 1.78                                     | 1.89                                    |
| 115  | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 13.99                                    | 14.62                                   |
| 116  | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 37.58                                    | 41.32                                   |
| 117  | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 23.88                                    | 30.82                                   |
| 118  | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 219.52                                   | 239.61                                  |
| 119  | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 217.70                                   | 239.21                                  |
| 120  | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 220.38                                   | 239.78                                  |
| 121  | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 21.50                                    | 21.72                                   |
| 122  | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 46.43                                    | 47.56                                   |
| 123  | vgg_16_pruned_0_5_tf                 | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 50.52                                    | 51.76                                   |
| 124  | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 18.55                                    | 18.71                                   |
| 125  | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 14.48                                    | 14.86                                   |
| 126  | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 14.09                                    | 14.76                                   |
| 127  | yolov4_416_tf                        | tf_yolov4_coco_416_416_60.3G                          | 14.43                                    | 15.35                                   |
| 128  | yolov4_512_tf                        | tf_yolov4_coco_512_512_91.2G                          | 10.98                                    | 11.65                                   |

  
</details>


### Performance on VCK5000
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H running at 4PE@350
MHz in Gen3x16:

| No\. | Model                                | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | bcc_pt                               | pt_BCC_shanghaitech_800_1000_268.9G                   | 350                | 49.02                                   |
| 2    | c2d2_lite_pt                         | pt_C2D2lite_CC20_512_512_6.86G                        | 350                | 22.48                                   |
| 3    | centerpoint                          | pt_centerpoint_astyx_2560_40_54G                      | 350                | 146.98                                  |
| 4    | CLOCs                                | pt_CLOCs_kitti_2.0                                    | 350                | 11.71                                   |
| 5    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 350                | 94.14                                   |
| 6    | efficientnet-b0_tf2                  | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 350                | 426.84                                  |
| 7    | efficientNet-edgetpu-L_tf            | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 350                | 393.64                                  |
| 8    | efficientNet-edgetpu-M_tf            | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 350                | 974.60                                  |
| 9    | efficientNet-edgetpu-S_tf            | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 350                | 1663.81                                 |
| 10   | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 350                | 67.35                                   |
| 11   | face_mask_detection_pt               | pt_face-mask-detection_512_512_0.59G                  | 350                | 770.84                                  |
| 12   | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 350                | 16649.70                                |
| 13   | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 350                | 2954.23                                 |
| 14   | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 350                | 11815.20                                |
| 15   | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 350                | 17826.50                                |
| 16   | fadnet                               | pt_fadnet_sceneflow_576_960_441G                      | 350                | 9.60                                    |
| 17   | fadnet_pruned                        | pt_fadnet_sceneflow_576_960_0.65_154G                 | 350                | 10.05                                   |
| 18   | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 350                | 277.87                                  |
| 19   | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 350                | 612.50                                  |
| 20   | HardNet_MSeg_pt                      | pt_HardNet_mixed_352_352_22.78G                       | 350                | 158.09                                  |
| 21   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 350                | 307.80                                  |
| 22   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 350                | 1585.16                                 |
| 23   | inception_v2_tf                      | tf_inceptionv2_imagenet_224_224_3.88G                 | 350                | 210.04                                  |
| 24   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 350                | 496.16                                  |
| 25   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 350                | 541.24                                  |
| 26   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 350                | 591.99                                  |
| 27   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 350                | 490.31                                  |
| 28   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 350                | 494.73                                  |
| 29   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 350                | 564.87                                  |
| 30   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 350                | 586.50                                  |
| 31   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 350                | 667.43                                  |
| 32   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 350                | 749.79                                  |
| 33   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 350                | 280.19                                  |
| 34   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 350                | 914.61                                  |
| 35   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 350                | 49.42                                   |
| 36   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 350                | 1638.84                                 |
| 37   | mobilenet_edge_0_75_tf               | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 350                | 3344.23                                 |
| 38   | mobilenet_edge_1_0_tf                | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 350                | 3101.21                                 |
| 39   | mobilenet_1_0_224_tf2                | tf2_mobilenetv1_imagenet_224_224_1.15G                | 350                | 5864.58                                 |
| 40   | mobilenet_v1_0_25_128_tf             | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 350                | 19353.80                                |
| 41   | mobilenet_v1_0_5_160_tf              | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 350                | 13349.20                                |
| 42   | mobilenet_v1_1_0_224_tf              | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 350                | 6561.03                                 |
| 43   | mobilenet_v1_1_0_224_pruned_0_11_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G        | 350                | 6959.29                                 |
| 44   | mobilenet_v1_1_0_224_pruned_0_12_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G           | 350                | 6934.40                                 |
| 45   | mobilenet_v2_1_0_224_tf              | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 350                | 4058.70                                 |
| 46   | mobilenet_v2_1_4_224_tf              | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 350                | 3137.54                                 |
| 47   | mobilenet_v2_cityscapes_tf           | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 350                | 16.29                                   |
| 48   | mobilenet_v3_small_1_0_tf2           | tf2_mobilenetv3_imagenet_224_224_132M                 | 350                | 1963.18                                 |
| 49   | movenet_ntd_pt                       | pt_movenet_coco_192_192_0.5G                          | 350                | 2750.94                                 |
| 50   | MT-resnet18_mixed_pt                 | pt_MT-resnet18_mixed_320_512_13.65G                   | 350                | 279.93                                  |
| 51   | multi_task_v3_pt                     | pt_multitaskv3_mixed_320_512_25.44G                   | 350                | 136.76                                  |
| 52   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 350                | 24.20                                   |
| 53   | ofa_depthwise_res50_pt               | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 350                | 2948.67                                 |
| 54   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 350                | 33.32                                   |
| 55   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 350                | 711.82                                  |
| 56   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 350                | 941.31                                  |
| 57   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 350                | 971.98                                  |
| 58   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 350                | 1514.09                                 |
| 59   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 350                | 2264.41                                 |
| 60   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 350                | 220.24                                  |
| 61   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 350                | 276.05                                  |
| 62   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 350                | 318.73                                  |
| 63   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 350                | 5444.55                                 |
| 64   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 350                | 2012.23                                 |
| 65   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 350                | 2480.24                                 |
| 66   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 350                | 2610.40                                 |
| 67   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 350                | 2734.31                                 |
| 68   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 350                | 2934.00                                 |
| 69   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 350                | 4782.06                                 |
| 70   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 350                | 2000.55                                 |
| 71   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G                        | 350                | 20.38                                   |
| 72   | pointpillars_kitti                   | pt_pointpillars_kitti_12000_100_10.8G                 | 350                | 3.13                                    |
| 73   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 350                | 39.08                                   |
| 74   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 350                | 35.28                                   |
| 75   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 350                | 200.90                                  |
| 76   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 350                | 200.96                                  |
| 77   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 350                | 332.21                                  |
| 78   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 350                | 418.89                                  |
| 79   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 350                | 536.56                                  |
| 80   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 350                | 555.17                                  |
| 81   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 350                | 1817.21                                 |
| 82   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 350                | 1909.43                                 |
| 83   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 350                | 2310.18                                 |
| 84   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 350                | 1146.44                                 |
| 85   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 350                | 825.95                                  |
| 86   | resnet_v2_50_tf                      | tf_resnetv2_50_imagenet_299_299_13.1G                 | 350                | 427.18                                  |
| 87   | resnet_v2_101_tf                     | tf_resnetv2_101_imagenet_299_299_26.78G               | 350                | 245.96                                  |
| 88   | resnet_v2_152_tf                     | tf_resnetv2_152_imagenet_299_299_40.47G               | 350                | 172.53                                  |
| 89   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 350                | 1533.14                                 |
| 90   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 350                | 1675.05                                 |
| 91   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 350                | 1698.25                                 |
| 92   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 350                | 1765.32                                 |
| 93   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 350                | 1837.51                                 |
| 94   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 350                | 1955.57                                 |
| 95   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 350                | 2068.41                                 |
| 96   | SA_gate_base_pt                      | pt_sa-gate_NYUv2_360_360_178G                         | 350                | 11.28                                   |
| 97   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 350                | 108.73                                  |
| 98   | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 350                | 60.46                                   |
| 99   | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 350                | 63.70                                   |
| 100  | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 350                | 731.00                                  |
| 101  | SemanticFPN_Mobilenetv2_pt           | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 350                | 207.71                                  |
| 102  | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 350                | 113.79                                  |
| 103  | solo_pt                              | pt_SOLO_coco_640_640_107G                             | 350                | 24.77                                   |
| 104  | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 350                | 3375.85                                 |
| 105  | ssd_inception_v2_coco_tf             | tf_ssdinceptionv2_coco_300_300_9.62G                  | 350                | 105.80                                  |
| 106  | ssd_mobilenet_v1_coco_tf             | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 350                | 2286.02                                 |
| 107  | ssd_mobilenet_v2_coco_tf             | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 350                | 1291.50                                 |
| 108  | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 350                | 80.70                                   |
| 109  | ssdlite_mobilenet_v2_coco_tf         | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 350                | 1875.26                                 |
| 110  | ssr_pt                               | pt_SSR_CVC_256_256_39.72G                             | 350                | 70.78                                   |
| 111  | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 350                | 30.81                                   |
| 112  | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 350                | 192.58                                  |
| 113  | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 350                | 592.00                                  |
| 114  | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 350                | 128.55                                  |
| 115  | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 350                | 2839.65                                 |
| 116  | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 350                | 2830.61                                 |
| 117  | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 350                | 2840.23                                 |
| 118  | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 350                | 328.51                                  |
| 119  | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 350                | 596.46                                  |
| 120  | vgg_16_pruned_0_5_tf                 | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 350                | 647.41                                  |
| 121  | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 350                | 293.45                                  |
| 122  | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 350                | 279.33                                  |
| 123  | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 350                | 276.70                                  |
| 124  | yolov4_416_tf                        | tf_yolov4_coco_416_416_60.3G                          | 350                | 240.10                                  |
| 125  | yolov4_512_tf                        | tf_yolov4_coco_512_512_91.2G                          | 350                | 137.83                                  |


The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H-aieDWC running at 6PE@350
MHz in Gen3x16:


| No\. | Model                                | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | bcc_pt                               | pt_BCC_shanghaitech_800_1000_268.9G                   | 350                | 79.10                                   |
| 2    | c2d2_lite_pt                         | pt_C2D2lite_CC20_512_512_6.86G                        | 350                | 36.73                                   |
| 3    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 350                | 153.31                                  |
| 4    | efficientNet-edgetpu-M_tf            | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 350                | 1377.58                                 |
| 5    | efficientNet-edgetpu-S_tf            | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 350                | 2488.34                                 |
| 6    | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 350                | 141.03                                  |
| 7    | face_mask_detection_pt               | pt_face-mask-detection_512_512_0.59G                  | 350                | 1335.01                                 |
| 8    | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 350                | 30992.50                                |
| 9    | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 350                | 4411.13                                 |
| 10   | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 350                | 20918.30                                |
| 11   | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 350                | 32349.60                                |
| 12   | fadnet                               | pt_fadnet_sceneflow_576_960_441G                      | 350                | 9.23                                    |
| 13   | fadnet_pruned                        | pt_fadnet_sceneflow_576_960_0.65_154G                 | 350                | 10.35                                   |
| 14   | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 350                | 426.19                                  |
| 15   | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 350                | 958.15                                  |
| 16   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 350                | 491.97                                  |
| 17   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 350                | 3500.30                                 |
| 18   | inception_v2_tf                      | tf_inceptionv2_imagenet_224_224_3.88G                 | 350                | 330.52                                  |
| 19   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 350                | 914.35                                  |
| 20   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 350                | 1001.82                                 |
| 21   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 350                | 1078.26                                 |
| 22   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 350                | 963.51                                  |
| 23   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 350                | 909.00                                  |
| 24   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 350                | 1062.79                                 |
| 25   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 350                | 1064.91                                 |
| 26   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 350                | 1253.56                                 |
| 27   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 350                | 1406.71                                 |
| 28   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 350                | 503.54                                  |
| 29   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 350                | 1352.53                                 |
| 30   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 350                | 73.27                                   |
| 31   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 350                | 3403.95                                 |
| 32   | mobilenet_edge_0_75_tf               | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 350                | 5689.70                                 |
| 33   | mobilenet_edge_1_0_tf                | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 350                | 5176.39                                 |
| 34   | mobilenet_1_0_224_tf2                | tf2_mobilenetv1_imagenet_224_224_1.15G                | 350                | 7820.25                                 |
| 35   | mobilenet_v1_0_25_128_tf             | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 350                | 20510.00                                |
| 36   | mobilenet_v1_0_5_160_tf              | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 350                | 14825.30                                |
| 37   | mobilenet_v1_1_0_224_tf              | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 350                | 8058.30                                 |
| 38   | mobilenet_v1_1_0_224_pruned_0_11_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G        | 350                | 8241.64                                 |
| 39   | mobilenet_v1_1_0_224_pruned_0_12_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G           | 350                | 8330.08                                 |
| 40   | mobilenet_v2_1_0_224_tf              | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 350                | 6504.78                                 |
| 41   | mobilenet_v2_1_4_224_tf              | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 350                | 4971.92                                 |
| 42   | mobilenet_v2_cityscapes_tf           | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 350                | 18.42                                   |
| 43   | movenet_ntd_pt                       | pt_movenet_coco_192_192_0.5G                          | 350                | 2267.16                                 |
| 44   | MT-resnet18_mixed_pt                 | pt_MT-resnet18_mixed_320_512_13.65G                   | 350                | 421.09                                  |
| 45   | multi_task_v3_pt                     | pt_multitaskv3_mixed_320_512_25.44G                   | 350                | 203.77                                  |
| 46   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 350                | 11.79                                   |
| 47   | ofa_depthwise_res50_pt               | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 350                | 3237.38                                 |
| 48   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 350                | 50.14                                   |
| 49   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 350                | 1206.24                                 |
| 50   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 350                | 1767.11                                 |
| 51   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 350                | 1784.54                                 |
| 52   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 350                | 2813.63                                 |
| 53   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 350                | 4013.12                                 |
| 54   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 350                | 285.31                                  |
| 55   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 350                | 376.22                                  |
| 56   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 350                | 441.71                                  |
| 57   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 350                | 9448.00                                 |
| 58   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 350                | 3750.97                                 |
| 59   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 350                | 4676.61                                 |
| 60   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 350                | 5033.46                                 |
| 61   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 350                | 5360.20                                 |
| 62   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 350                | 6037.33                                 |
| 63   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 350                | 8047.31                                 |
| 64   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 350                | 3423.98                                 |
| 65   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G_2.5                    | 350                | 18.09                                   |
| 66   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 350                | 37.10                                   |
| 67   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 350                | 53.41                                   |
| 68   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 350                | 307.64                                  |
| 69   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 350                | 307.59                                  |
| 70   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 350                | 528.82                                  |
| 71   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 350                | 663.61                                  |
| 72   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 350                | 957.42                                  |
| 73   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 350                | 998.94                                  |
| 74   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 350                | 3736.48                                 |
| 75   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 350                | 4023.41                                 |
| 76   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 350                | 5432.02                                 |
| 77   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 350                | 2244.46                                 |
| 78   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 350                | 1598.02                                 |
| 79   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 350                | 3152.91                                 |
| 80   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 350                | 3432.18                                 |
| 81   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 350                | 3504.63                                 |
| 82   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 350                | 3718.85                                 |
| 83   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 350                | 3982.00                                 |
| 84   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 350                | 4381.54                                 |
| 85   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 350                | 4804.25                                 |
| 86   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 350                | 158.09                                  |
| 87   | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 350                | 91.22                                   |
| 88   | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 350                | 114.93                                  |
| 89   | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 350                | 1049.65                                 |
| 90   | SemanticFPN_Mobilenetv2_pt           | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 350                | 234.42                                  |
| 91   | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 350                | 185.87                                  |
| 92   | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 350                | 7062.16                                 |
| 93   | ssd_inception_v2_coco_tf             | tf_ssdinceptionv2_coco_300_300_9.62G                  | 350                | 166.21                                  |
| 94   | ssd_mobilenet_v1_coco_tf             | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 350                | 2471.66                                 |
| 95   | ssd_mobilenet_v2_coco_tf             | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 350                | 1879.99                                 |
| 96   | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 350                | 108.15                                  |
| 97   | ssdlite_mobilenet_v2_coco_tf         | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 350                | 2552.13                                 |
| 98   | ssr_pt                               | pt_SSR_CVC_256_256_39.72G                             | 350                | 90.06                                   |
| 99   | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 350                | 41.83                                   |
| 100  | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 350                | 261.90                                  |
| 101  | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 350                | 1039.88                                 |
| 102  | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 350                | 201.48                                  |
| 103  | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 350                | 5301.04                                 |
| 104  | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 350                | 5284.14                                 |
| 105  | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 350                | 5300.12                                 |
| 106  | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 350                | 500.50                                  |
| 107  | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 350                | 965.05                                  |
| 108  | vgg_16_pruned_0_5_tf                 | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 350                | 1047.76                                 |
| 109  | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 350                | 446.08                                  |
| 110  | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 350                | 389.71                                  |
| 111  | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 350                | 385.25                                  |

  
The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H-aieMISC running at 6PE@350
MHz in Gen3x16:


| No\. | Model                                | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | centerpoint                          | pt_centerpoint_astyx_2560_40_54G                      | 350                | 10.83                                   |
| 2    | CLOCs                                | pt_CLOCs_kitti_2.0                                    | 350                | 17.10                                   |
| 3    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 350                | 141.07                                  |
| 4    | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 350                | 91.51                                   |
| 5    | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 350                | 24426.30                                |
| 6    | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 350                | 4397.97                                 |
| 7    | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 350                | 17183.60                                |
| 8    | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 350                | 25900.50                                |
| 9    | fadnet                               | pt_fadnet_sceneflow_576_960_441G                      | 350                | 8.91                                    |
| 10   | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 350                | 381.48                                  |
| 11   | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 350                | 880.96                                  |
| 12   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 350                | 448.06                                  |
| 13   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 350                | 2224.44                                 |
| 14   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 350                | 711.53                                  |
| 15   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 350                | 781.03                                  |
| 16   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 350                | 842.44                                  |
| 17   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 350                | 719.18                                  |
| 18   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 350                | 710.52                                  |
| 19   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 350                | 817.96                                  |
| 20   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 350                | 833.27                                  |
| 21   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 350                | 962.62                                  |
| 22   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 350                | 1087.39                                 |
| 23   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 350                | 397.71                                  |
| 24   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 350                | 1294.82                                 |
| 25   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 350                | 67.65                                   |
| 26   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 350                | 2454.78                                 |
| 27   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 350                | 33.85                                   |
| 28   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 350                | 49.92                                   |
| 29   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 350                | 1033.25                                 |
| 30   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 350                | 1376.67                                 |
| 31   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 350                | 1402.76                                 |
| 32   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 350                | 2216.75                                 |
| 33   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 350                | 3305.77                                 |
| 34   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 350                | 279.77                                  |
| 35   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 350                | 366.23                                  |
| 36   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 350                | 424.64                                  |
| 37   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 350                | 8060.23                                 |
| 38   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 350                | 3012.43                                 |
| 39   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 350                | 3697.90                                 |
| 40   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 350                | 3885.44                                 |
| 41   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 350                | 4073.29                                 |
| 42   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 350                | 4374.36                                 |
| 43   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 350                | 7050.56                                 |
| 44   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 350                | 2887.85                                 |
| 45   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G_2.5                    | 350                | 15.55                                   |
| 46   | pointpillars_kitti                   | pt_pointpillars_kitti_12000_100_10.8G                 | 350                | 3.10                                    |
| 47   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 350                | 34.46                                   |
| 48   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 350                | 52.80                                   |
| 49   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 350                | 292.19                                  |
| 50   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 350                | 292.72                                  |
| 51   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 350                | 486.09                                  |
| 52   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 350                | 597.70                                  |
| 53   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 350                | 800.49                                  |
| 54   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 350                | 828.60                                  |
| 55   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 350                | 2718.97                                 |
| 56   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 350                | 2852.94                                 |
| 57   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 350                | 3448.88                                 |
| 58   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 350                | 1715.80                                 |
| 59   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 350                | 1237.54                                 |
| 60   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 350                | 2306.45                                 |
| 61   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 350                | 2507.43                                 |
| 62   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 350                | 2635.89                                 |
| 63   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 350                | 2744.08                                 |
| 64   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 350                | 2918.96                                 |
| 65   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 350                | 3086.25                                 |
| 66   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 350                | 2504.13                                 |
| 67   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 350                | 152.58                                  |
| 68   | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 350                | 83.71                                   |
| 69   | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 350                | 80.99                                   |
| 70   | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 350                | 992.31                                  |
| 71   | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 350                | 170.61                                  |
| 72   | solo_pt                              | pt_SOLO_coco_640_640_107G                             | 350                | 24.13                                   |
| 73   | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 350                | 4617.91                                 |
| 74   | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 350                | 105.05                                  |
| 75   | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 350                | 40.61                                   |
| 76   | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 350                | 257.27                                  |
| 77   | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 350                | 870.67                                  |
| 78   | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 350                | 185.08                                  |
| 79   | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 350                | 4166.42                                 |
| 80   | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 350                | 4161.00                                 |
| 81   | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 350                | 4168.73                                 |
| 82   | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 350                | 478.68                                  |
| 83   | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 350                | 887.03                                  |
| 84   | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 350                | 958.77                                  |
| 85   | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 350                | 428.68                                  |
| 86   | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 350                | 386.94                                  |
| 87   | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 350                | 382.30                                  |
| 88   | yolov4_416_tf                        | tf_yolov4_coco_416_416_60.3G                          | 350                | 315.31                                  |
| 89   | yolov4_512_tf                        | tf_yolov4_coco_512_512_91.2G                          | 350                | 171.52                                  |


The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H running at 8PE@350
MHz in Gen3x16:


| No\. | Model                                | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 350                | 204.59                                  |
| 2    | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 350                | 148.32                                  |
| 3    | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 350                | 31080.90                                |
| 4    | fadnet                               | pt_fadnet_sceneflow_576_960_441G                      | 350                | 9.65                                    |
| 5    | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 350                | 506.42                                  |
| 6    | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 350                | 1177.00                                 |
| 7    | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 350                | 4224.61                                 |
| 8    | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 350                | 1504.80                                 |
| 9    | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 350                | 76.78                                   |
| 10   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 350                | 4519.23                                 |
| 11   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 350                | 12.69                                   |
| 12   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 350                | 66.81                                   |
| 13   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 350                | 1490.15                                 |
| 14   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 350                | 2288.00                                 |
| 15   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 350                | 2166.96                                 |
| 16   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 350                | 6182.04                                 |
| 17   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 350                | 6635.75                                 |
| 18   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 350                | 7057.42                                 |
| 19   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 350                | 7852.00                                 |
| 20   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 350                | 71.19                                   |
| 21   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 350                | 399.88                                  |
| 22   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 350                | 399.94                                  |
| 23   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 350                | 692.54                                  |
| 24   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 350                | 738.23                                  |
| 25   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 350                | 1220.63                                 |
| 26   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 350                | 1265.40                                 |
| 27   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 350                | 4944.30                                 |
| 28   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 350                | 5335.53                                 |
| 29   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 350                | 7077.62                                 |
| 30   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 350                | 2979.15                                 |
| 31   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 350                | 2122.98                                 |
| 32   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 350                | 4167.80                                 |
| 33   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 350                | 4544.06                                 |
| 34   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 350                | 4651.83                                 |
| 35   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 350                | 4942.05                                 |
| 36   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 350                | 5280.03                                 |
| 37   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 350                | 5813.44                                 |
| 38   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 350                | 6370.42                                 |
| 39   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 350                | 159.51                                  |
| 40   | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 350                | 89.41                                   |
| 41   | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 350                | 117.98                                  |
| 42   | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 350                | 1084.85                                 |
| 43   | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 350                | 233.24                                  |
| 44   | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 350                | 7536.39                                 |
| 45   | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 350                | 114.53                                  |
| 46   | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 350                | 49.48                                   |
| 47   | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 350                | 1359.92                                 |
| 48   | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 350                | 247.93                                  |
| 49   | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 350                | 6526.42                                 |
| 50   | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 350                | 6910.37                                 |
| 51   | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 350                | 6945.23                                 |
| 52   | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 350                | 476.97                                  |
| 53   | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 350                | 469.79                                  |


</details>


### Performance on U50lv
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for each model on the `Alveo U50lv` board with 10 DPUCAHX8H kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                                | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 165                | 336.95                                  |
| 2    | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 165                | 87.66                                   |
| 3    | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 165                | 21381.20                                |
| 4    | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 165                | 1251.82                                 |
| 5    | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 165                | 7323.87                                 |
| 6    | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 165                | 21136.40                                |
| 7    | fadnet_pt                            | pt_fadnet_sceneflow_576_960_441G                      | 165                | 1.16                                    |
| 8    | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 165                | 138.17                                  |
| 9    | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 165                | 212.52                                  |
| 10   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 165                | 158.87                                  |
| 11   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 165                | 1172.51                                 |
| 12   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 165                | 376.13                                  |
| 13   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 165                | 408.94                                  |
| 14   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 165                | 496.03                                  |
| 15   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 165                | 381.25                                  |
| 16   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 165                | 436.47                                  |
| 17   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 165                | 449.85                                  |
| 18   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 165                | 493.43                                  |
| 19   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 165                | 550.77                                  |
| 20   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 165                | 632.56                                  |
| 21   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 165                | 170.86                                  |
| 22   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 165                | 1068.04                                 |
| 23   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 165                | 511.80                                  |
| 24   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 165                | 14.11                                   |
| 25   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 165                | 4.85                                    |
| 26   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 165                | 45.04                                   |
| 27   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 165                | 306.88                                  |
| 28   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 165                | 476.08                                  |
| 29   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 165                | 598.32                                  |
| 30   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 165                | 909.32                                  |
| 31   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 165                | 1548.17                                 |
| 32   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 165                | 98.38                                   |
| 33   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 165                | 124.95                                  |
| 34   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 165                | 159.48                                  |
| 35   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 165                | 5858.88                                 |
| 36   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 165                | 825.99                                  |
| 37   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 165                | 1078.24                                 |
| 38   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 165                | 1181.30                                 |
| 39   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 165                | 1338.94                                 |
| 40   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 165                | 1500.32                                 |
| 41   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 165                | 3502.12                                 |
| 42   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 165                | 1005.87                                 |
| 43   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G_2.5                    | 165                | 11.36                                   |
| 44   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 165                | 21.16                                   |
| 45   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 165                | 46.73                                   |
| 46   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 165                | 72.55                                   |
| 47   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 165                | 73.04                                   |
| 48   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 165                | 135.76                                  |
| 49   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 165                | 246.23                                  |
| 50   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 165                | 377.29                                  |
| 51   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 165                | 430.81                                  |
| 52   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 165                | 593.21                                  |
| 53   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 165                | 770.88                                  |
| 54   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 165                | 1033.57                                 |
| 55   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 165                | 307.93                                  |
| 56   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 165                | 205.48                                  |
| 57   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 165                | 593.40                                  |
| 58   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 165                | 511.61                                  |
| 59   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 165                | 616.01                                  |
| 60   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 165                | 669.76                                  |
| 61   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 165                | 740.15                                  |
| 62   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 165                | 841.86                                  |
| 63   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 165                | 939.27                                  |
| 64   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 165                | 148.27                                  |
| 65   | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 165                | 32.70                                   |
| 66   | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 165                | 56.95                                   |
| 67   | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 165                | 432.47                                  |
| 68   | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 165                | 188.92                                  |
| 69   | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 165                | 4127.72                                 |
| 70   | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 165                | 32.34                                   |
| 71   | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 165                | 4.34                                    |
| 72   | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 165                | 71.97                                   |
| 73   | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 165                | 247.12                                  |
| 74   | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 165                | 86.41                                   |
| 75   | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 165                | 1316.57                                 |
| 76   | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 165                | 1314.66                                 |
| 77   | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 165                | 1099.59                                 |
| 78   | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 165                | 151.61                                  |
| 79   | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 165                | 283.28                                  |
| 80   | vgg_16_pruned_0_5_tf                 | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 165                | 306.23                                  |
| 81   | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 165                | 125.71                                  |
| 82   | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 165                | 78.62                                   |
| 83   | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 165                | 77.83                                   |


The following table lists the performance number including end-to-end throughput for each model on the `Alveo U50lv` board with 8 DPUCAHX8H-DWC kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                                | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | bcc_pt                               | pt_BCC_shanghaitech_800_1000_268.9G                   | 165                | 16.67                                   |
| 2    | c2d2_lite_pt                         | pt_C2D2lite_CC20_512_512_6.86G                        | 165                | 17.55                                   |
| 3    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 165                | 268.16                                  |
| 4    | efficientNet-edgetpu-M_tf            | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 165                | 543.66                                  |
| 5    | efficientNet-edgetpu-S_tf            | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 165                | 883.96                                  |
| 6    | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 165                | 69.98                                   |
| 7    | face_mask_detection_pt               | pt_face-mask-detection_512_512_0.59G                  | 165                | 737.28                                  |
| 8    | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 165                | 20109.10                                |
| 9    | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 165                | 1001.98                                 |
| 10   | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 165                | 5908.28                                 |
| 11   | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 165                | 17368.60                                |
| 12   | fadnet_pt                            | pt_fadnet_sceneflow_576_960_441G                      | 165                | 3.86                                    |
| 13   | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 165                | 110.50                                  |
| 14   | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 165                | 170.04                                  |
| 15   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 165                | 127.05                                  |
| 16   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 165                | 938.83                                  |
| 17   | inception_v2_tf                      | tf_inceptionv2_imagenet_224_224_3.88G                 | 165                | 322.89                                  |
| 18   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 165                | 300.88                                  |
| 19   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 165                | 327.95                                  |
| 20   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 165                | 396.64                                  |
| 21   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 165                | 305.50                                  |
| 22   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 165                | 300.75                                  |
| 23   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 165                | 360.21                                  |
| 24   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 165                | 394.91                                  |
| 25   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 165                | 440.54                                  |
| 26   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 165                | 507.14                                  |
| 27   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 165                | 136.73                                  |
| 28   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 165                | 855.48                                  |
| 29   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 165                | 11.29                                   |
| 30   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 165                | 409.83                                  |
| 31   | mobilenet_edge_0_75_tf               | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 165                | 1954.19                                 |
| 32   | mobilenet_edge_1_0_tf                | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 165                | 1583.69                                 |
| 33   | mobilenet_1_0_224_tf2                | tf2_mobilenetv1_imagenet_224_224_1.15G                | 165                | 2415.83                                 |
| 34   | mobilenet_v1_0_25_128_tf             | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 165                | 14834.80                                |
| 35   | mobilenet_v1_0_5_160_tf              | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 165                | 8837.20                                 |
| 36   | mobilenet_v1_1_0_224_tf              | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 165                | 2415.28                                 |
| 37   | mobilenet_v1_1_0_224_pruned_0_11_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G        | 165                | 2487.63                                 |
| 38   | mobilenet_v1_1_0_224_pruned_0_12_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G           | 165                | 2507.15                                 |
| 39   | mobilenet_v2_1_0_224_tf              | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 165                | 2226.09                                 |
| 40   | mobilenet_v2_1_4_224_tf              | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 165                | 1498.50                                 |
| 41   | movenet_ntd_pt                       | pt_movenet_coco_192_192_0.5G                          | 165                | 1747.97                                 |
| 42   | MT-resnet18_mixed_pt                 | pt_MT-resnet18_mixed_320_512_13.65G                   | 165                | 217.50                                  |
| 43   | multi_task_v3_pt                     | pt_multitaskv3_mixed_320_512_25.44G                   | 165                | 118.96                                  |
| 44   | ocr_pt                               | pt_OCR_ICDAR2015_960_960_875G                         | 165                | 4.72                                    |
| 45   | ofa_depthwise_res50_pt               | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 165                | 2269.73                                 |
| 46   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 165                | 36.06                                   |
| 47   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 165                | 245.50                                  |
| 48   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 165                | 379.84                                  |
| 49   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 165                | 479.30                                  |
| 50   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 165                | 729.18                                  |
| 51   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 165                | 1242.57                                 |
| 52   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 165                | 78.87                                   |
| 53   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 165                | 99.68                                   |
| 54   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 165                | 127.52                                  |
| 55   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 165                | 4727.32                                 |
| 56   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 165                | 661.48                                  |
| 57   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 165                | 865.41                                  |
| 58   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 165                | 946.77                                  |
| 59   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 165                | 1074.36                                 |
| 60   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 165                | 1204.93                                 |
| 61   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 165                | 2813.31                                 |
| 62   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 165                | 806.20                                  |
| 63   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G_2.5                    | 165                | 11.33                                   |
| 64   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 165                | 20.55                                   |
| 65   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 165                | 37.41                                   |
| 66   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 165                | 58.01                                   |
| 67   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 165                | 58.51                                   |
| 68   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 165                | 108.58                                  |
| 69   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 165                | 196.84                                  |
| 70   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 165                | 302.12                                  |
| 71   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 165                | 345.01                                  |
| 72   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 165                | 475.39                                  |
| 73   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 165                | 618.43                                  |
| 74   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 165                | 828.47                                  |
| 75   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 165                | 246.71                                  |
| 76   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 165                | 164.32                                  |
| 77   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 165                | 475.35                                  |
| 78   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 165                | 409.64                                  |
| 79   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 165                | 493.81                                  |
| 80   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 165                | 536.85                                  |
| 81   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 165                | 592.43                                  |
| 82   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 165                | 674.89                                  |
| 83   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 165                | 753.52                                  |
| 84   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 165                | 139.21                                  |
| 85   | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 165                | 28.22                                   |
| 86   | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 165                | 47.44                                   |
| 87   | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 165                | 346.75                                  |
| 88   | SemanticFPN_Mobilenetv2_pt           | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 165                | 135.54                                  |
| 89   | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 165                | 151.05                                  |
| 90   | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 165                | 2884.24                                 |
| 91   | ssd_inception_v2_coco_tf             | tf_ssdinceptionv2_coco_300_300_9.62G                  | 165                | 156.87                                  |
| 92   | ssd_mobilenet_v1_coco_tf             | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 165                | 1127.75                                 |
| 93   | ssd_mobilenet_v2_coco_tf             | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 165                | 635.81                                  |
| 94   | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 165                | 25.93                                   |
| 95   | ssdlite_mobilenet_v2_coco_tf         | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 165                | 1045.29                                 |
| 96   | ssr_pt                               | pt_SSR_CVC_256_256_39.72G                             | 165                | 26.84                                   |
| 97   | textmountain_pt                      | pt_textmountain_ICDAR_960_960_575.2G                  | 165                | 7.83                                    |
| 98   | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 165                | 57.60                                   |
| 99   | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 165                | 197.57                                  |
| 100  | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 165                | 69.14                                   |
| 101  | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 165                | 1056.01                                 |
| 102  | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 165                | 1054.04                                 |
| 103  | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 165                | 1055.42                                 |
| 104  | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 165                | 121.12                                  |
| 105  | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 165                | 226.44                                  |
| 106  | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 165                | 244.66                                  |
| 107  | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 165                | 100.65                                  |
| 108  | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 165                | 62.93                                   |
| 109  | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 165                | 62.35                                   |

  
</details>


### Performance on U55C DWC
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for each model on the `Alveo U55C` board with 11 DPUCAHX8H-DWC kernels running at 300Mhz in Gen3x4:
  

| No\. | Model                                | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :----------------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | bcc_pt                               | pt_BCC_shanghaitech_800_1000_268.9G                   | 300                | 37.43                                   |
| 2    | c2d2_lite_pt                         | pt_C2D2lite_CC20_512_512_6.86G                        | 300                | 28.25                                   |
| 3    | drunet_pt                            | pt_DRUNet_Kvasir_528_608_0.4G                         | 300                | 469.21                                  |
| 4    | efficientNet-edgetpu-M_tf            | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 300                | 1204.68                                 |
| 5    | efficientNet-edgetpu-S_tf            | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 300                | 2000.42                                 |
| 6    | ENet_cityscapes_pt                   | pt_ENet_cityscapes_512_1024_8.6G                      | 300                | 147.29                                  |
| 7    | face_mask_detection_pt               | pt_face-mask-detection_512_512_0.59G                  | 300                | 1581.35                                 |
| 8    | face-quality_pt                      | pt_face-quality_80_60_61.68M                          | 300                | 30918.60                                |
| 9    | facerec-resnet20_mixed_pt            | pt_facerec-resnet20_mixed_112_96_3.5G                 | 300                | 2423.09                                 |
| 10   | facereid-large_pt                    | pt_facereid-large_96_96_515M                          | 300                | 15138.20                                |
| 11   | facereid-small_pt                    | pt_facereid-small_80_80_90M                           | 300                | 32819.60                                |
| 12   | FairMot_pt                           | pt_FairMOT_mixed_640_480_0.5_36G                      | 300                | 239.29                                  |
| 13   | FPN-resnet18_covid19-seg_pt          | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 300                | 401.98                                  |
| 14   | Inception_resnet_v2_tf               | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 300                | 298.63                                  |
| 15   | Inception_v1_tf                      | tf_inceptionv1_imagenet_224_224_3G                    | 300                | 2185.25                                 |
| 16   | inception_v2_tf                      | tf_inceptionv2_imagenet_224_224_3.88G                 | 300                | 672.69                                  |
| 17   | Inception_v3_tf                      | tf_inceptionv3_imagenet_299_299_11.45G                | 300                | 691.41                                  |
| 18   | inception_v3_pruned_0_2_tf           | tf_inceptionv3_imagenet_299_299_0.2_9.1G              | 300                | 692.45                                  |
| 19   | inception_v3_pruned_0_4_tf           | tf_inceptionv3_imagenet_299_299_0.4_6.9G              | 300                | 809.48                                  |
| 20   | Inception_v3_tf2                     | tf2_inceptionv3_imagenet_299_299_11.5G                | 300                | 709.43                                  |
| 21   | Inception_v3_pt                      | pt_inceptionv3_imagenet_299_299_11.4G                 | 300                | 690.40                                  |
| 22   | inception_v3_pruned_0_3_pt           | pt_inceptionv3_imagenet_299_299_0.3_8G                | 300                | 754.35                                  |
| 23   | inception_v3_pruned_0_4_pt           | pt_inceptionv3_imagenet_299_299_0.4_6.8G              | 300                | 811.57                                  |
| 24   | inception_v3_pruned_0_5_pt           | pt_inceptionv3_imagenet_299_299_0.5_5.7G              | 300                | 911.55                                  |
| 25   | inception_v3_pruned_0_6_pt           | pt_inceptionv3_imagenet_299_299_0.6_4.5G              | 300                | 1045.46                                 |
| 26   | Inception_v4_tf                      | tf_inceptionv4_imagenet_299_299_24.55G                | 300                | 324.68                                  |
| 27   | medical_seg_cell_tf2                 | tf2_2d-unet_nuclei_128_128_5.31G                      | 300                | 1972.68                                 |
| 28   | mlperf_ssd_resnet34_tf               | tf_mlperf_resnet34_coco_1200_1200_433G                | 300                | 25.84                                   |
| 29   | mlperf_resnet50_tf                   | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 300                | 1012.03                                 |
| 30   | mobilenet_edge_0_75_tf               | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 300                | 4716.44                                 |
| 31   | mobilenet_edge_1_0_tf                | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 300                | 3834.40                                 |
| 32   | mobilenet_1_0_224_tf2                | tf2_mobilenetv1_imagenet_224_224_1.15G                | 300                | 5305.78                                 |
| 33   | mobilenet_v1_0_25_128_tf             | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 300                | 18873.80                                |
| 34   | mobilenet_v1_0_5_160_tf              | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 300                | 12862.50                                |
| 35   | mobilenet_v1_1_0_224_tf              | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 300                | 5305.64                                 |
| 36   | mobilenet_v1_1_0_224_pruned_0_11_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G        | 300                | 5432.80                                 |
| 37   | mobilenet_v1_1_0_224_pruned_0_12_tf  | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G           | 300                | 5460.44                                 |
| 38   | MT-resnet18_mixed_pt                 | pt_MT-resnet18_mixed_320_512_13.65G                   | 300                | 511.94                                  |
| 39   | multi_task_v3_pt                     | pt_multitaskv3_mixed_320_512_25.44G                   | 300                | 291.27                                  |
| 40   | ofa_depthwise_res50_pt               | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 300                | 2887.10                                 |
| 41   | ofa_rcan_latency_pt                  | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 300                | 65.53                                   |
| 42   | ofa_resnet50_baseline_pt             | pt_OFA-resnet50_imagenet_224_224_15.0G                | 300                | 597.57                                  |
| 43   | ofa_resnet50_pruned_0_45_pt          | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G            | 300                | 913.97                                  |
| 44   | ofa_resnet50_pruned_0_60_pt          | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G            | 300                | 1138.00                                 |
| 45   | ofa_resnet50_pruned_0_74_pt          | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G            | 300                | 1724.06                                 |
| 46   | ofa_resnet50_0_9B_pt                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G            | 300                | 2922.00                                 |
| 47   | ofa_yolo_pt                          | pt_OFA-yolo_coco_640_640_48.88G                       | 300                | 178.70                                  |
| 48   | ofa_yolo_pruned_0_30_pt              | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 300                | 223.19                                  |
| 49   | ofa_yolo_pruned_0_50_pt              | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 300                | 283.17                                  |
| 50   | person-orientation_pruned_pt         | pt_person-orientation_224_112_558M                    | 300                | 11379.50                                |
| 51   | personreid-res50_pt                  | pt_personreid-res50_market1501_256_128_5.3G           | 300                | 1611.62                                 |
| 52   | personreid_res50_pruned_0_4_pt       | pt_personreid-res50_market1501_256_128_0.4_3.3G       | 300                | 2123.04                                 |
| 53   | personreid_res50_pruned_0_5_pt       | pt_personreid-res50_market1501_256_128_0.5_2.7G       | 300                | 2323.59                                 |
| 54   | personreid_res50_pruned_0_6_pt       | pt_personreid-res50_market1501_256_128_0.6_2.1G       | 300                | 2629.79                                 |
| 55   | personreid_res50_pruned_0_7_pt       | pt_personreid-res50_market1501_256_128_0.7_1.6G       | 300                | 2954.94                                 |
| 56   | personreid-res18_pt                  | pt_personreid-res18_market1501_176_80_1.1G            | 300                | 6648.41                                 |
| 57   | pmg_pt                               | pt_pmg_rp2k_224_224_2.28G                             | 300                | 1989.61                                 |
| 58   | pointpainting_nuscenes               | pt_pointpainting_nuscenes_126G_2.5                    | 300                | 20.11                                   |
| 59   | pointpillars_nuscenes                | pt_pointpillars_nuscenes_40000_64_108G                | 300                | 39.93                                   |
| 60   | rcan_pruned_tf                       | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 300                | 80.06                                   |
| 61   | refinedet_VOC_tf                     | tf_refinedet_VOC_320_320_81.9G                        | 300                | 138.34                                  |
| 62   | RefineDet-Medical_EDD_baseline_tf    | tf_RefineDet-Medical_EDD_320_320_81.28G               | 300                | 139.50                                  |
| 63   | RefineDet-Medical_EDD_pruned_0_5_tf  | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G           | 1300               | 248.60                                  |
| 64   | RefineDet-Medical_EDD_pruned_0_75_tf | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G          | 300                | 433.45                                  |
| 65   | RefineDet-Medical_EDD_pruned_0_85_tf | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G          | 300                | 698.40                                  |
| 66   | RefineDet-Medical_EDD_tf             | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 300                | 789.95                                  |
| 67   | resnet_v1_50_tf                      | tf_resnetv1_50_imagenet_224_224_6.97G                 | 300                | 1175.61                                 |
| 68   | resnet_v1_50_pruned_0_38_tf          | tf_resnetv1_50_imagenet_224_224_0.38_4.3G             | 300                | 1521.60                                 |
| 69   | resnet_v1_50_pruned_0_65_tf          | tf_resnetv1_50_imagenet_224_224_0.65_2.45G            | 300                | 2031.64                                 |
| 70   | resnet_v1_101_tf                     | tf_resnetv1_101_imagenet_224_224_14.4G                | 300                | 610.35                                  |
| 71   | resnet_v1_152_tf                     | tf_resnetv1_152_imagenet_224_224_21.83G               | 300                | 407.06                                  |
| 72   | resnet50_tf2                         | tf2_resnet50_imagenet_224_224_7.76G                   | 300                | 1174.21                                 |
| 73   | resnet50_pt                          | pt_resnet50_imagenet_224_224_8.2G                     | 300                | 1012.40                                 |
| 74   | resnet50_pruned_0_3_pt               | pt_resnet50_imagenet_224_224_0.3_5.8G                 | 300                | 1214.86                                 |
| 75   | resnet50_pruned_0_4_pt               | pt_resnet50_imagenet_224_224_0.4_4.9G                 | 300                | 1321.38                                 |
| 76   | resnet50_pruned_0_5_pt               | pt_resnet50_imagenet_224_224_0.5_4.1G                 | 300                | 1458.08                                 |
| 77   | resnet50_pruned_0_6_pt               | pt_resnet50_imagenet_224_224_0.6_3.3G                 | 300                | 1654.50                                 |
| 78   | resnet50_pruned_0_7_pt               | pt_resnet50_imagenet_224_224_0.7_2.5G                 | 300                | 1845.19                                 |
| 79   | salsanext_pt                         | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 300                | 153.16                                  |
| 80   | salsanext_v2_pt                      | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 300                | 58.41                                   |
| 81   | semantic_seg_citys_tf2               | tf2_erfnet_cityscapes_512_1024_54G                    | 300                | 90.03                                   |
| 82   | SemanticFPN_cityscapes_pt            | pt_SemanticFPN_cityscapes_256_512_10G                 | 300                | 803.50                                  |
| 83   | SemanticFPN_Mobilenetv2_pt           | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 300                | 228.29                                  |
| 84   | SESR_S_pt                            | pt_SESR-S_DIV2K_360_640_7.48G                         | 300                | 282.56                                  |
| 85   | SqueezeNet_pt                        | pt_squeezenet_imagenet_224_224_351.7M                 | 300                | 6398.41                                 |
| 86   | ssd_inception_v2_coco_tf             | tf_ssdinceptionv2_coco_300_300_9.62G                  | 300                | 329.92                                  |
| 87   | ssd_mobilenet_v1_coco_tf             | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 300                | 2115.80                                 |
| 88   | ssd_mobilenet_v2_coco_tf             | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 300                | 1458.46                                 |
| 89   | ssd_resnet_50_fpn_coco_tf            | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 300                | 58.99                                   |
| 90   | ssdlite_mobilenet_v2_coco_tf         | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 300                | 2061.95                                 |
| 91   | ssr_pt                               | pt_SSR_CVC_256_256_39.72G                             | 300                | 62.02                                   |
| 92   | tsd_yolox_pt                         | pt_yolox_TT100K_640_640_73G                           | 300                | 131.89                                  |
| 93   | ultrafast_pt                         | pt_ultrafast_CULane_288_800_8.4G                      | 300                | 474.24                                  |
| 94   | unet_chaos-CT_pt                     | pt_unet_chaos-CT_512_512_23.3G                        | 300                | 136.80                                  |
| 95   | chen_color_resnet18_pt               | pt_vehicle-color-classification_color_224_224_3.63G   | 300                | 2645.11                                 |
| 96   | vehicle_make_resnet18_pt             | pt_vehicle-make-classification_CompCars_224_224_3.63G | 300                | 2634.76                                 |
| 97   | vehicle_type_resnet18_pt             | pt_vehicle-type-classification_CompCars_224_224_3.63G | 300                | 2642.15                                 |
| 98   | vgg_16_tf                            | tf_vgg16_imagenet_224_224_30.96G                      | 300                | 283.80                                  |
| 99   | vgg_16_pruned_0_43_tf                | tf_vgg16_imagenet_224_224_0.43_17.67G                 | 300                | 540.89                                  |
| 100  | vgg_16_pruned_0_5_tf                 | tf_vgg16_imagenet_224_224_0.5_15.64G                  | 300                | 584.30                                  |
| 101  | vgg_19_tf                            | tf_vgg19_imagenet_224_224_39.28G                      | 300                | 238.41                                  |
| 102  | yolov3_voc_tf                        | tf_yolov3_voc_416_416_65.63G                          | 300                | 152.75                                  |
| 103  | yolov3_coco_tf2                      | tf2_yolov3_coco_416_416_65.9G                         | 300                | 150.59                                  |


</details>


### Performance on U200
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U200` board with 2 DPUCADF8H kernels running at 300Mhz:
  

| No\. | Model        | Name                                    | E2E latency (ms) Thread num =20 | E2E throughput \-fps\(Multi Thread) |
| ---- | :----------- | :-------------------------------------- | ------------------------------- | ----------------------------------- |
| 1    | resnet50     | cf_resnet50_imagenet_224_224_7.7G       | 3.8                             | 1054                                |
| 2    | Inception_v1 | tf_inceptionv1_imagenet_224_224_3G      | 2.2                             | 1834                                |
| 3    | Inception_v3 | tf_inceptionv3_imagenet_299_299_11.45G  | 18.4                            | 218                                 |
| 4    | resnetv1_50  | tf_resnetv1_50_imagenet_224_224_6.97G   | 4.2                             | 947                                 |
| 5    | resnetv1_101 | tf_resnetv1_101_imagenet_224_224_14.4G  | 8.5                             | 472                                 |
| 6    | resnetv1_152 | tf_resnetv1_152_imagenet_224_224_21.83G | 12.7                            | 316                                 |

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U200` board with 2 DPUCADX8G kernels running at 350Mhz with xilinx_u200_xdma_201830_2 shell:
  

| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 2.13                            | 470.6                                 | 561.3                                 |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 2.08                             | 481                                 | 1157.8                               |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 2.39                            | 418.5                                 | 1449.4                                 |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 2.11                            | 475.1                                 | 1129.2                               |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 15.67                            | 63.8                                   | 371.6                               |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 10.77                           | 92.8                                   | 221.2                                |
| 7    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 10.99                            | 91                                 | 1157.1                                 |
| 8    | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 8.69                            | 115.1                                 | 667.9                               |
| 9    | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 14.53                           | 68.8                                   | 75.9                                  |
| 10   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 19.90                           | 50.3                                   | 82.1                                |

</details>

### Performance on U250
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U250` board with 4 DPUCADF8H kernels running at 300Mhz:
  

| No\. | Model        | Name                                    | E2E latency (ms) Thread num =20 | E2E throughput \-fps\(Multi Thread) |
| ---- | :----------- | :-------------------------------------- | ------------------------------- | ----------------------------------- |
| 1    | resnet50     | cf_resnet50_imagenet_224_224_7.7G       | 1.94                            | 2134.8                              |
| 2    | Inception_v1 | tf_inceptionv1_imagenet_224_224_3G      | 1.10                            | 3631.7                              |
| 3    | Inception_v3 | tf_inceptionv3_imagenet_299_299_11.45G  | 9.20                            | 434.9                               |
| 4    | resnetv1_50  | tf_resnetv1_50_imagenet_224_224_6.97G   | 2.13                            | 1881.6                              |
| 5    | resnetv1_101 | tf_resnetv1_101_imagenet_224_224_14.4G  | 4.24                            | 941.9                               |
| 6    | resnetv1_152 | tf_resnetv1_152_imagenet_224_224_21.83G | 6.35                            | 630.3                               |

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U250` board with 4 DPUCADX8G kernels running at 350Mhz with xilinx_u250_xdma_201830_1 shell:
  

| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 1.68                            | 595.5                                 | 1223.95                                 |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 1.67                             | 600.5                                 | 2422.5                               |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 1.93                            | 517.1                                 | 4059.8                                 |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 1.65                            | 607.8                                 | 23221                               |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 6.18                            | 161.8                                   | 743.8                               |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 5.77                           | 173.4                                   | 452.4                                |
| 7    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 5.44                            | 183.7                                 | 2349.7                                 |
| 8    | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 7.43                            | 167.2                                 | 898.5                               |
| 9    | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 14.27                           | 70.1                                   | 146.7                                  |
| 10   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 9.46                           | 105.7                                   | 139.4                                |

Note: For xilinx_u250_gen3x16_xdma_shell_3_1_202020_1 latest shell U250 xclbins, Alveo U250 board would be having only 3 DPUCADF8H kernels instead of 4, thereby the performance numbers for 1 Alveo U250 board with xilinx_u250_gen3x16_xdma_shell_3_1 shell xclbins would be 75% of the above reported performance numbers which is for 4 DPUCADF8H kernels. 

</details>


### Performance on Ultra96  
The performance number shown below was measured with the previous AI SDK v2.0.4 on Ultra96 v1.
The Vitis platform of Ultra96 v2 has not been released yet. So the performance numbers are therefore not reported for this Model Zoo release.  

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Ultra96` board with a `1 * B1600  @ 287MHz   V1.4.0` DPU configuration:

**Note:** The original power supply of Ultra96 is not designed for high performance AI workload. The board may occasionally hang to run few models, When multi-thread is used. For such situations, `NA` is specified in the following table.


| No\. | Model                          | Name                                                | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
|------|--------------------------------|-----------------------------------------------------|---------------------------------|-----------------------------------------|----------------------------------------|
| 1    | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G             | 30\.8                           | 32\.4667                                | 33\.4667                               |
| 2    | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G         | 13\.98                          | 71\.55                                  | 75\.0667                               |
| 3    | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G             | 17\.16                          | 58\.2667                                | 61\.2833                               |
| 4    | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G         | 44\.05                          | 22\.7                                   | 23\.4333                               |
| 5    | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G         | 7\.34                           | 136\.183                                | NA                                     |
| 6    | tf\_resnet50                   | tf\_resnet50\_imagenet\_224\_224\_6\.97G            | 28\.02                          | 35\.6833                                | 36\.6                                  |
| 7    | tf\_inception\_v1              | tf\_inceptionv1\_imagenet\_224\_224\_3G             | 16\.96                          | 58\.9667                                | 61\.2833                               |
| 8    | tf\_mobilenet\_v2              | tf\_mobilenetv2\_imagenet\_224\_224\_1\.17G         | 10\.17                          | 98\.3                                   | 104\.25                                |
| 9    | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G            | 24\.3                           | 41\.15                                  | 46\.2                                  |
| 10   | ssd\_pedestrian\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | 23\.29                          | 42\.9333                                | 50\.8                                  |
| 11   | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G              | 35\.5                           | 28\.1667                                | 31\.8                                  |
| 12   | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 60\.79                          | 16\.45                                  | 27\.8167                               |
| 13   | tf\_ssd\_voc                   | tf\_ssd\_voc\_300\_300\_64\.81G                     | 186\.92                         | 5\.35                                   | 5\.81667                               |
| 14   | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G               | 4\.17                           | 239\.883                                | 334\.167                               |
| 15   | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G               | 8\.55                           | 117                                     | 167\.2                                 |
| 16   | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G      | 22\.79                          | 43\.8833                                | 49\.6833                               |
| 17   | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G                  | 185\.19                         | 5\.4                                    | 5\.53                                  |
| 18   | tf\_yolov3\_voc                | tf\_yolov3\_voc\_416\_416\_65\.63G                  | 199\.34                         | 5\.01667                                | 5\.1                                   |
| 19   | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G            | 66\.37                          | 15\.0667                                | NA                                     |
| 20   | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G       | 32\.17                          | 31\.0883                                | 33\.6667                               |
| 21   | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G        | 20\.29                          | 49\.2833                                | 55\.25                                 |
| 22   | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G                | 36\.34                          | 27\.5167                                | NA                                     |
| 23   | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G     | 13\.9                           | 71\.9333                                | NA                                     |
| 24   | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G           | 3\.82                           | 261\.55                                 | 277\.4                                 |
| 25   | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G | 560\.75                         | 1\.78333                                | NA                                     |
| 26   | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G                      | 118\.11                         | 8\.46667                                | 8\.9                                   |
| 27   | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G           | 37\.5                           | 26\.6667                                | 30\.65                                 |
| 28   | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G            | 30\.99                          | 32\.2667                                | 38\.35                                 |
| 29   | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G            | 26\.29                          | 38\.03333                               | 46\.8333                               |
| 30   | Inception\-v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G         | 88\.76                          | 11\.2667                                | 11\.5333                               |
| 31   | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G             | 5\.96                           | 167\.867                                | 283\.583                               |
| 32   | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G                | 2\.95                           | 339\.183                                | 347\.633                               |
| 33   | reid                           | cf\_reid\_market1501\_160\_80\_0\.95G               | 6\.28                           | 159\.15                                 | 166\.633                               |
| 34   | yolov3\_bdd                    | dk\_yolov3\_bdd\_288\_512\_53\.7G                   | 193\.55                         | 5\.16667                                | 5\.31667                               |
| 35   | tf\_mobilenet\_v1              | tf\_mobilenetv1\_imagenet\_224\_224\_1\.14G         | 5\.97                           | 167\.567                                | 186\.55                                |
| 36   | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | 13\.47                          | 74\.2167                                | 77\.8167                               |
| 37   | resnet18\_wide                 | tf\_resnet18\_imagenet\_224\_224\_28G               | 97\.72                          | 10\.2333                                | 10\.3833                               |
</details>

### Performance on ZCU102 (0432055-04)  
This version of ZCU102 is out of stock. The performance number shown below was measured with the previous AI SDK v2.0.4. Now this form has stopped updating.
So the performance numbers are therefore not reported for this Model Zoo release.

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-04)` board with a  `3 * B4096  @ 287MHz   V1.4.0` DPU configuration:


| No\. | Model                          | Name                                                | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
|------|--------------------------------|-----------------------------------------------------|---------------------------------|-----------------------------------------|----------------------------------------|
| 1    | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G             | 12\.85                          | 77\.8                                         | 179\.3                                 |
| 2    | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G         | 5\.47                           | 182\.683                                | 485\.533                               |
| 3    | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G             | 6\.76                           | 147\.933                                | 373\.267                               |
| 4    | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G         | 17                              | 58\.8333                                | 155\.4                                 |
| 5    | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G         | 4\.09                           | 244\.617                                | 638\.067                               |
| 6    | tf\_resnet50                   | tf\_resnet50\_imagenet\_224\_224\_6\.97G            | 11\.94                          | 83\.7833                                | 191\.417                               |
| 7    | tf\_inception\_v1              | tf\_inceptionv1\_imagenet\_224\_224\_3G             | 6\.72                           | 148\.867                                | 358\.283                               |
| 8    | tf\_mobilenet\_v2              | tf\_mobilenetv2\_imagenet\_224\_224\_1\.17G         | 5\.46                           | 183\.117                                | 458\.65                                |
| 9    | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G            | 11\.33                          | 88\.2667                                | 320\.5                                 |
| 10   | ssd\_pedestrian\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | 12\.96                          | 77\.1833                                | 314\.717                               |
| 11   | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G              | 17\.49                          | 57\.1833                                | 218\.183                               |
| 12   | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 24\.21                          | 41\.3                                         | 141\.233                               |
| 13   | tf\_ssd\_voc                   | tf\_ssd\_voc\_300\_300\_64\.81G                     | 69\.28                          | 14\.4333                                | 46\.7833                               |
| 14   | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G               | 2\.43                           | 412\.183                                | 1416\.63                               |
| 15   | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G               | 5\.01                           | 199\.717                                | 719\.75                                |
| 16   | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G      | 11\.09                          | 90\.1667                                | 259\.65                                |
| 17   | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G                  | 70\.51                          | 14\.1833                                | 44\.4                                  |
| 18   | tf\_yolov3\_voc                | tf\_yolov3\_voc\_416\_416\_65\.63G                  | 70\.75                          | 14\.1333                                | 44\.0167                               |
| 19   | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G            | 29\.91                          | 33\.4333                                | 109\.067                               |
| 20   | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G       | 15\.39                          | 64\.9667                                | 216\.317                               |
| 21   | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G        | 11\.04                          | 90\.5833                                | 312                                    |
| 22   | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G                | 16\.58                          | 60\.3                                         | 203\.867                               |
| 23   | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G     | 9\.44                           | 105\.9                                         | 424\.667                               |
| 24   | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G           | 1\.73                           | 579\.067                                | 1620\.67                               |
| 25   | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G | 279\.07                         | 3\.58333                                | 38\.5                                  |
| 26   | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G                      | 39\.76                          | 25\.15                                         | 86\.35                                 |
| 27   | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G           | 18\.42                          | 54\.2833                                | 211\.217                               |
| 28   | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G            | 16\.42                          | 60\.9167                                | 242\.433                               |
| 29   | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G            | 14\.46                          | 69\.1667                                | 286\.733                               |
| 30   | Inception\-v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G         | 34\.25                          | 29\.2                                         | 84\.25                                 |
| 31   | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G             | 3\.6                            | 277\.65                                 | 1080\.77                               |
| 32   | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G                | 1\.13                           | 885\.033                                | 1623\.3                                |
| 33   | reid                           | cf\_reid\_marketcuhk\_160\_80\_0\.95G               | 2\.67                           | 375                                           | 773\.533                               |
| 34   | yolov3\_bdd                    | dk\_yolov3\_bdd\_288\_512\_53\.7G                   | 73\.89                          | 13\.5333                                | 42\.8833                               |
| 35   | tf\_mobilenet\_v1              | tf\_mobilenetv1\_imagenet\_224\_224\_1\.14G         | 3\.2                            | 312\.067                                | 875\.967                               |
| 36   | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | 5\.1                            | 195\.95                                 | 524\.433                               |
| 37   | resnet18\_wide                 | tf\_resnet18\_imagenet\_224\_224\_28G               | 33\.28                          | 30\.05                                         | 83\.4167                               |
</details>


## Contributing
We welcome community contributions. When contributing to this repository, first discuss the change you wish to make via:

* [GitHub Issues](https://github.com/Xilinx/Vitis-AI/issues)
* [Forum](https://forums.xilinx.com/t5/AI-and-Vitis-AI/bd-p/AI)
* <a href="mailto:xilinx_ai_model_zoo@xilinx.com">Email</a>

You can also submit a pull request with details on how to improve the product. Prior to submitting your pull request, ensure that you can build the product and run all the demos with your patch. In case of a larger feature, provide a relevant demo.

## License

Xilinx AI Model Zoo is licensed under [Apache License Version 2.0](reference-files/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

<hr/>
<p align="center"><sup>Copyright&copy; 2019 Xilinx</sup></p>
