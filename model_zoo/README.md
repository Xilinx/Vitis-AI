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

- Computation in the table are counted as FLOPs
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
  
| No.  | Application                                             | Name                                                      |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | General                                                 | tf\_inceptionresnetv2\_imagenet\_299\_299\_26\.35G_2.5    |                           0\.8037                            |                           0\.7946                            |                           299\*299                           |                26\.35G                |
| 2    | General                                                 | tf\_inceptionv1\_imagenet\_224\_224\_3G_2.5               |                           0\.6976                            |                           0\.6794                            |                           224\*224                           |                  3G                   |
| 3    | General                                                 | tf_inceptionv2_imagenet_224_224_3.88G_2.5                 |                            0.7399                            |                            07331                             |                           224*224                            |                 3.88G                 |
| 4    | General                                                 | tf\_inceptionv3\_imagenet\_299\_299\_11\.45G_2.5          |                           0\.7798                            |                           0\.7735                            |                           299\*299                           |                11\.45G                |
| 5    | General                                                 | tf_inceptionv3_imagenet_299_299_0.2_9.1G_2.5              |                            0.7786                            |                            0.7668                            |                           299\*299                           |                 9.1G                  |
| 6    | General                                                 | tf_inceptionv3_imagenet_299_299_0.4_6.9G_2.5              |                            0.7669                            |                            0.7561                            |                           299\*299                           |                 6.9G                  |
| 7    | General                                                 | tf\_inceptionv4\_imagenet\_299\_299\_24\.55G_2.5          |                           0\.8018                            |                           0\.7928                            |                           299\*299                           |                24\.55G                |
| 8    | General                                                 | tf\_mobilenetv1\_0\.25\_imagenet\_128\_128\_27M_2.5       |                           0\.4144                            |                           0\.3464                            |                           128\*128                           |                27\.15M                |
| 9    | General                                                 | tf\_mobilenetv1\_0\.5\_imagenet\_160\_160\_150M\_2.5      |                           0\.5903                            |                           0\.5195                            |                           160\*160                           |                 150M                  |
| 10   | General                                                 | tf\_mobilenetv1\_1\.0\_imagenet\_224\_224\_1\.14G_2.5     |                           0\.7102                            |                           0\.6780                            |                           224\*224                           |                1\.14G                 |
| 11   | General                                                 | tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G_2.5        |                            0.7056                            |                            0.6822                            |                           224\*224                           |                 1.02G                 |
| 12   | General                                                 | tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G_2.5           |                            0.7060                            |                            0.6850                            |                           224\*224                           |                  1G                   |
| 13   | General                                                 | tf\_mobilenetv2\_1\.0\_imagenet\_224\_224\_602M_2.5       |                           0\.7013                            |                           0\.6767                            |                           224\*224                           |                 602M                  |
| 14   | General                                                 | tf\_mobilenetv2\_1\.4\_imagenet\_224\_224\_1\.16G_2.5     |                           0\.7411                            |                           0\.7194                            |                           224\*224                           |                1\.16G                 |
| 15   | General                                                 | tf\_resnetv1\_50\_imagenet\_224\_224\_6\.97G_2.5          |                           0\.7520                            |                           0\.7436                            |                           224\*224                           |                6\.97G                 |
| 16   | General                                                 | tf_resnetv1_50_imagenet_224_224_0.38_4.3G_2.5             |                            0.7442                            |                            0.7375                            |                           224\*224                           |                 4.3G                  |
| 17   | General                                                 | tf_resnetv1_50_imagenet_224_224_0.65_2.45G_2.5            |                            0.7279                            |                            0.7167                            |                           224\*224                           |                 2.45G                 |
| 18   | General                                                 | tf\_resnetv1\_101\_imagenet\_224\_224\_14\.4G_2.5         |                           0\.7640                            |                           0\.7560                            |                           224\*224                           |                14\.4G                 |
| 19   | General                                                 | tf\_resnetv1\_152\_imagenet\_224\_224\_21\.83G_2.5        |                           0\.7681                            |                           0\.7463                            |                           224\*224                           |                21\.83G                |
| 20   | General                                                 | tf\_vgg16\_imagenet\_224\_224\_30\.96G_2.5                |                           0\.7089                            |                           0\.7069                            |                           224\*224                           |                30\.96G                |
| 21   | General                                                 | tf_vgg16_imagenet_224_224_0.43_17.67G_2.5                 |                            0.6929                            |                            0.6823                            |                           224\*224                           |                17.67G                 |
| 22   | General                                                 | tf_vgg16_imagenet_224_224_0.5_15.64G_2.5                  |                            0.6857                            |                            0.6729                            |                           224\*224                           |                15.64G                 |
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
| 36   | General                                                 | tf2_efficientnet-b0_imagenet_224_224_0.36G_2.5            |                        0.7690/0.9320                         |                        0.7515/0.9273                         |                           224*224                            |                 0.36G                 |
| 37   | General                                                 | tf2_mobilenetv3_imagenet_224_224_132M_2.5                 |                        0.6756/0.8728                         |                        0.6536/0.8544                         |                           224*224                            |                 132M                  |
| 38   | General                                                 | pt_inceptionv3_imagenet_299_299_11.4G_2.5                 |                         0.775/0.936                          |                         0.771/0.935                          |                           299*299                            |                 11.4G                 |
| 39   | General                                                 | pt_inceptionv3_imagenet_299_299_0.3_8G_2.5                |                         0.775/0.936                          |                         0.772/0.935                          |                           299*299                            |                  8G                   |
| 40   | General                                                 | pt_inceptionv3_imagenet_299_299_0.4_6.8G_2.5              |                         0.768/0.931                          |                         0.764/0.929                          |                           299*299                            |                 6.8G                  |
| 41   | General                                                 | pt_inceptionv3_imagenet_299_299_0.5_5.7G_2.5              |                         0.757/0.921                          |                         0.752/0.918                          |                           299*299                            |                 5.7G                  |
| 42   | General                                                 | pt_inceptionv3_imagenet_299_299_0.6_4.5G_2.5              |                         0.739/0.911                          |                         0.732/0.908                          |                           299*299                            |                 4.5G                  |
| 43   | General                                                 | pt_squeezenet_imagenet_224_224_703.5M_2.5                 |                         0.582/0.806                          |                         0.582/0.806                          |                           224*224                            |                703.5M                 |
| 44   | General                                                 | pt_resnet50_imagenet_224_224_8.2G_2.5                     |                         0.761/0.929                          |                         0.760/0.928                          |                           224*224                            |                 8.2G                  |
| 45   | General                                                 | pt_resnet50_imagenet_224_224_0.3_5.8G_2.5                 |                         0.760/0.929                          |                         0.757/0.928                          |                           224*224                            |                 5.8G                  |
| 46   | General                                                 | pt_resnet50_imagenet_224_224_0.4_4.9G_2.5                 |                         0.755/0.926                          |                         0.752/0.925                          |                           224*224                            |                 4.9G                  |
| 47   | General                                                 | pt_resnet50_imagenet_224_224_0.5_4.1G_2.5                 |                         0.748/0.921                          |                         0.745/0.920                          |                           224*224                            |                 4.1G                  |
| 48   | General                                                 | pt_resnet50_imagenet_224_224_0.6_3.3G_2.5                 |                         0.742/0.917                          |                         0.738/0.915                          |                           224*224                            |                 3.3G                  |
| 49   | General                                                 | pt_resnet50_imagenet_224_224_0.7_2.5G_2.5                 |                         0.726/0.908                          |                         0.720/0.906                          |                           224*224                            |                 2.5G                  |
| 50   | General                                                 | pt_OFA-resnet50_imagenet_224_224_15.0G_2.5                |                         0.799/0.948                          |                         0.789/0.944                          |                           224*224                            |                 15.0G                 |
| 51   | General                                                 | pt_OFA-resnet50_imagenet_224_224_0.45_8.2G_2.5            |                         0.795/0.945                          |                         0.784/0.941                          |                           224*224                            |                 8.2G                  |
| 52   | General                                                 | pt_OFA-resnet50_imagenet_224_224_0.60_6.0G_2.5            |                         0.791/0.943                          |                         0.780/0.939                          |                           224*224                            |                 6.0G                  |
| 53   | General                                                 | pt_OFA-resnet50_imagenet_192_192_0.74_3.6G_2.5            |                         0.777/0.937                          |                         0.770/0.933                          |                           192*192                            |                 3.6G                  |
| 54   | General                                                 | pt_OFA-resnet50_imagenet_160_160_0.88_1.8G_2.5            |                         0.752/0.918                          |                         0.744/0.918                          |                           160*160                            |                 1.8G                  |
| 55   | General                                                 | pt_OFA-depthwise-res50_imagenet_176_176_2.49G_2.5         |                        0.7633/0.9292                         |                        0.7629/0.9306                         |                           176*176                            |                 2.49G                 |
| 56   | General                                                 | tf_ViT_imagenet_352_352_21.3G_2.5                         |                            0.8282                            |                            0.8254                            |                           352*352                            |                 21.3G                 |
| 57   | Car type classification                                 | pt_vehicle-type-classification_CompCars_224_224_3.63G_2.5 |                            0.9025                            |                            0.9011                            |                           224*224                            |                 3.63G                 |
| 58   | Car make classification                                 | pt_vehicle-make-classification_CompCars_224_224_3.63G_2.5 |                            0.8991                            |                            0.8939                            |                           224*224                            |                 3.63G                 |
| 59   | Car color classification                                | pt_vehicle-color-classification_color_224_224_3.63G_2.5   |                            0.9549                            |                            0.9549                            |                           224*224                            |                 3.63G                 |

</font>
<br>

## Detection
<font size=2> 

| No.  | Application                                             | Name                                                      |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |                          Input Size                          |                  OPS                  |
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
| 16   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_81.28G_2.5               |                            0.7866                            |                            0.7857                            |                           320*320                            |                81.28G                 |
| 17   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_0.5_41.42G_2.5           |                            0.7798                            |                            0.7772                            |                           320*320                            |                41.42G                 |
| 18   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_0.75_20.54G_2.5          |                            0.7885                            |                            0.7826                            |                           320*320                            |                20.54G                 |
| 19   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_0.85_12.32G_2.5          |                            0.7898                            |                            0.7877                            |                           320*320                            |                12.32G                 |
| 20   | Medical Detection                                       | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G_2.5           |                            0.7839                            |                            0.8002                            |                           320*320                            |                 9.83G                 |
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
  
| No.  | Application                                             | Name                                                      |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |                          Input Size                          |                  OPS                  |
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
  
| No.  | Application                                             | Name                                                      |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
  | 1    | Question and Answering                                  | tf_bert-base_SQuAD_128_11.17G_2.5                         |                            0.8694                            |                            0.8656                            |                             128                              |                11.17G                 |
| 2    | Sentiment Detection                                     | tf2_sentiment-detection_IMDB_500_32_53.3M_2.5             |                            0.8708                            |                            0.8695                            |                            500*32                            |                 53.3M                 |
| 3    | Customer Satisfaction Assessment                        | tf2_customer-satisfaction_Cars4U_25_32_2.7M_2.5           |                            0.9565                            |                            0.9565                            |                            25*32                             |                 2.7M                  |
| 4    | Open Information Extraction                             | pt_open-information-extraction_qasrl_100_200_1.5G_2.5     |               Acc/F1-score:<br/>58.70%/77.12%                |               Acc/F1-score:<br/>58.70%/77.19%                |                           100*200                            |                 1.5G                  |
</font>
<br>

## Text-OCR
<font size=2> 
  
| No.  | Application                                             | Name                                                      |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | Text Detection                                          | pt_textmountain_ICDAR_960_960_575.2G_2.5                  |                            0.8863                            |                            0.8851                            |                           960*960                            |                575.2G                 |
| 2    | E2E OCR                                                 | pt_OCR_ICDAR2015_960_960_875G_2.5                         |                            0.6758                            |                            0.6776                            |                           960*960                            |                 875G                  |
</font>
<br>

## Surveillance
<font size=2> 
  
| No.  | Application                                             | Name                                                      |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |                          Input Size                          |                  OPS                  |
| ---- | :------------------------------------------------------ | :-------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------: |
| 1    | Face Recognition                                        | pt_facerec-resnet20_mixed_112_96_3.5G_2.5                 |                            0.9955                            |                            0.9947                            |                            112*96                            |                 3.5G                  |
| 2    | Face Quality                                            | pt_face-quality_80_60_61.68M_2.5                          |                            0.1233                            |                            0.1258                            |                            80*60                             |                61.68M                 |
| 3    | Face ReID                                               | pt_facereid-large_96_96_515M_2.5                          |                    mAP:0.794  Rank1:0.955                    |                    mAP:0.790  Rank1:0.953                    |                            96*96                             |                 515M                  |
| 4    | Face ReID                                               | pt_facereid-small_80_80_90M_2.5                           |                    mAP:0.560  Rank1:0.865                    |                    mAP:0.559  Rank1:0.865                    |                            80*80                             |                 90M5                  |
| 5    | ReID                                                    | pt_personreid-res18_market1501_176_80_1.1G_2.5            |                    mAP:0.753  Rank1:0.898                    |                    mAP:0.746  Rank1:0.893                    |                            176*80                            |                 1.1G                  |
| 6    | ReID                                                    | pt_personreid-res50_market1501_256_128_5.3G_2.5           |                    mAP:0.866  Rank1:0.951                    |                    mAP:0.869  Rank1:0.948                    |                           256*128                            |                 5.3G                  |
| 7    | ReID                                                    | pt_personreid-res50_market1501_256_128_0.4_3.3G_2.5       |                    mAP:0.869  Rank1:0.948                    |                    mAP:0.869  Rank1:0.948                    |                           256*128                            |                 3.3G                  |
| 8    | ReID                                                    | pt_personreid-res50_market1501_256_128_0.5_2.7G_2.5       |                    mAP:0.864  Rank1:0.944                    |                    mAP:0.864  Rank1:0.944                    |                           256*128                            |                 2.7G                  |
| 9    | ReID                                                    | pt_personreid-res50_market1501_256_128_0.6_2.1G_2.5       |                    mAP:0.863  Rank1:0.946                    |                    mAP:0.859  Rank1:0.942                    |                           256*128                            |                 2.1G                  |
| 10   | ReID                                                    | pt_personreid-res50_market1501_256_128_0.7_1.6G_2.5       |                    mAP:0.850  Rank1:0.940                    |                    mAP:0.848  Rank1:0.938                    |                           256*128                            |                 1.6G                  |
| 11   | Person orientation estimation                           | pt_person-orientation_224_112_558M_2.5                    |                            0.930                             |                            0.929                             |                           224*112                            |                 558M                  |
| 12   | Joint detection and Tracking                            | pt_FairMOT_mixed_640_480_0.5_36G_2.5                      |                  MOTA 59.1%<br/>IDF1 62.5%                   |                  MOTA 58.1%<br/>IDF1 60.5%                   |                           640*480                            |                  36G                  |
| 13   | Crowd Counting                                          | pt_BCC_shanghaitech_800_1000_268.9G_2.5                   |                  MAE: 65.83<br>MSE: 111.75                   |             MAE: 67.60<br>MSE: 117.36                        |                           800*1000                           |                268.9G                 |
| 14   | Face Mask Detection                                     | pt_face-mask-detection_512_512_0.59G_2.5                  |                            0.8860                            |                            0.8810                            |                           512*512                            |                 0.59G                 |
| 15   | Pose Estimation                                         | pt_movenet_coco_192_192_0.5G_2.5                          |                            0.7972                            |                            0.7984                            |                           192*192                            |                 0.5G                  |

</font>
<br>

## Industrial-Vision-Robotics
<font size=2> 
  
| No.  | Application                                             | Name                                                      |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |                          Input Size                          |                  OPS                  |
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

| No.  | Application                                             | Name                                                      |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |                          Input Size                          |                  OPS                  |
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


| No\. | Model                        | GPU Model Standard Name                               | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | bcc_pt                       | pt_BCC_shanghaitech_800_1000_268.9G                   | 3.31                                     | 10.86                                   |
| 2    | c2d2_lite_pt                 | pt_C2D2lite_CC20_512_512_6.86G                        | 2.84                                     | 5.24                                    |
| 3    | centerpoint                  | pt_centerpoint_astyx_2560_40_54G                      | 16.02                                    | 47.9                                    |
| 4    | CLOCs                        | pt_CLOCs_kitti_2.0                                    | 2.85                                     | 10.18                                   |
| 5    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 60.77                                    | 189.53                                  |
| 6    | efficientdet_d2_tf           | tf_efficientdet-d2_coco_768_768_11.06G                | 3.10                                     | 6.05                                    |
| 7    | efficientnet-b0_tf2          | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 78.02                                    | 152.61                                  |
| 8    | efficientNet-edgetpu-L_tf    | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 34.99                                    | 90.14                                   |
| 9    | efficientNet-edgetpu-M_tf    | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 79.88                                    | 205.22                                  |
| 10   | efficientNet-edgetpu-S_tf    | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 115.05                                   | 308.15                                  |
| 11   | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 10.10                                    | 36.43                                   |
| 12   | face_mask_detection_pt       | pt_face-mask-detection_512_512_0.59G                  | 115.80                                   | 401.08                                  |
| 13   | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 2988.30                                  | 8791.90                                 |
| 14   | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 168.33                                   | 337.68                                  |
| 15   | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 941.81                                   | 2275.70                                 |
| 16   | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 2240.00                                  | 6336.00                                 |
| 17   | fadnet                       | pt_fadnet_sceneflow_576_960_441G                      | 1.17                                     | 1.59                                    |
| 18   | fadnet_pruned                | pt_fadnet_sceneflow_576_960_0.65_154G                 | 1.76                                     | 2.67                                    |
| 19   | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 22.59                                    | 66.03                                   |
| 20   | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 37.08                                    | 107.18                                  |
| 21   | HardNet_MSeg_pt              | pt_HardNet_mixed_352_352_22.78G                       | 24.71                                    | 56.77                                   |
| 22   | HFnet_tf                     | tf_HFNet_mixed_960_960_20.09G                         | 3.57                                     | 15.90                                   |
| 23   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 23.89                                    | 51.93                                   |
| 24   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 191.26                                   | 470.88                                  |
| 25   | inception_v2_tf              | tf_inceptionv2_imagenet_224_224_3.88G                 | 93.04                                    | 229.67                                  |
| 26   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 59.75                                    | 135.30                                  |
| 27   | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 59.18                                    | 136.24                                  |
| 28   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 59.83                                    | 136.02                                  |
| 29   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 28.83                                    | 68.57                                   |
| 30   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 154.32                                   | 393.59                                  |
| 31   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 1.87                                     | 7.12                                    |
| 32   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 78.85                                    | 173.51                                  |
| 33   | mobilenet_edge_0_75_tf       | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 262.66                                   | 720.24                                  |
| 34   | mobilenet_edge_1_0_tf        | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 214.78                                   | 554.55                                  |
| 35   | mobilenet_1_0_224_tf2        | tf2_mobilenetv1_imagenet_224_224_1.15G                | 321.47                                   | 939.61                                  |
| 36   | mobilenet_v1_0_25_128_tf     | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 1332.90                                  | 4754.20                                 |
| 37   | mobilenet_v1_0_5_160_tf      | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 914.84                                   | 3116.80                                 |
| 38   | mobilenet_v1_1_0_224_tf      | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 326.62                                   | 951.00                                  |
| 39   | mobilenet_v2_1_0_224_tf      | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 268.07                                   | 707.03                                  |
| 40   | mobilenet_v2_1_4_224_tf      | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 191.08                                   | 473.15                                  |
| 41   | mobilenet_v2_cityscapes_tf   | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 1.75                                     | 5.28                                    |
| 42   | mobilenet_v3_small_1_0_tf2   | tf2_mobilenetv3_imagenet_224_224_132M                 | 336.62                                   | 965.76                                  |
| 43   | movenet_ntd_pt               | pt_movenet_coco_192_192_0.5G                          | 94.72                                    | 390.02                                  |
| 44   | MT-resnet18_mixed_pt         | pt_MT-resnet18_mixed_320_512_13.65G                   | 32.79                                    | 102.96                                  |
| 45   | multi_task_v3_pt             | pt_multitaskv3_mixed_320_512_25.44G                   | 17.10                                    | 61.35                                   |
| 46   | ocr_pt                       | pt_OCR_ICDAR2015_960_960_875G                         | 1.10                                     | 3.41                                    |
| 47   | ofa_depthwise_res50_pt       | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 106.03                                   | 370.18                                  |
| 48   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 17.00                                    | 28.03                                   |
| 49   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 183.81                                   | 354.73                                  |
| 50   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 16.87                                    | 42.75                                   |
| 51   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 21.53                                    | 54.59                                   |
| 52   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 27.71                                    | 71.35                                   |
| 53   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 661.42                                   | 1428.2                                  |
| 54   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 107.54                                   | 237.66                                  |
| 55   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 370.36                                   | 690.85                                  |
| 56   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 151.69                                   | 366.62                                  |
| 57   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 1.28                                     | 4.25                                    |
| 58   | pointpillars_kitti           | pt_pointpillars_kitti_12000_100_10.8G                 | 19.63                                    | 49.20                                   |
| 59   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 2.23                                     | 9.61                                    |
| 60   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 8.61                                     | 18.05                                   |
| 61   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 11.31                                    | 34.50                                   |
| 62   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 67.62                                    | 229.37                                  |
| 63   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 87.95                                    | 191.05                                  |
| 64   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 46.38                                    | 110.72                                  |
| 65   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 31.64                                    | 77.23                                   |
| 66   | resnet_v2_50_tf              | tf_resnetv2_50_imagenet_299_299_13.1G                 | 45.05                                    | 99.64                                   |
| 67   | resnet_v2_101_tf             | tf_resnetv2_101_imagenet_299_299_26.78G               | 23.60                                    | 55.60                                   |
| 68   | resnet_v2_152_tf             | tf_resnetv2_152_imagenet_299_299_40.47G               | 16.07                                    | 38.18                                   |
| 69   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 87.33                                    | 192.94                                  |
| 70   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 77.58                                    | 173.16                                  |
| 71   | SA_gate_base_pt              | pt_sa-gate_NYUv2_360_360_178G                         | 3.29                                     | 9.45                                    |
| 72   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 5.61                                     | 21.28                                   |
| 73   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 4.15                                     | 10.99                                   |
| 74   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 7.44                                     | 23.88                                   |
| 75   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 35.49                                    | 162.79                                  |
| 76   | SemanticFPN_Mobilenetv2_pt   | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 10.52                                    | 52.46                                   |
| 77   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 88.30                                    | 140.63                                  |
| 78   | solo_pt                      | pt_SOLO_coco_640_640_107G                             | 1.45                                     | 4.84                                    |
| 79   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 575.41                                   | 1499.60                                 |
| 80   | ssd_inception_v2_coco_tf     | tf_ssdinceptionv2_coco_300_300_9.62G                  | 39.41                                    | 102.23                                  |
| 81   | ssd_mobilenet_v1_coco_tf     | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 111.06                                   | 332.23                                  |
| 82   | ssd_mobilenet_v2_coco_tf     | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 81.77                                    | 213.25                                  |
| 83   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 2.92                                     | 5.17                                    |
| 84   | ssdlite_mobilenet_v2_coco_tf | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 106.06                                   | 304.19                                  |
| 85   | ssr_pt                       | pt_SSR_CVC_256_256_39.72G                             | 6.03                                     | 14.42                                   |
| 86   | superpoint_tf                | tf_superpoint_mixed_480_640_52.4G                     | 12.61                                    | 53.79                                   |
| 87   | textmountain_pt              | pt_textmountain_ICDAR_960_960_575.2G                  | 1.67                                     | 4.68                                    |
| 88   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 13.13                                    | 33.70                                   |
| 89   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 35.08                                    | 95.82                                   |
| 90   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 22.80                                    | 69.63                                   |
| 91   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 204.70                                   | 499.15                                  |
| 92   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 203.44                                   | 498.51                                  |
| 93   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 205.03                                   | 499.64                                  |
| 94   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 20.15                                    | 40.94                                   |
| 95   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 17.37                                    | 36.47                                   |
| 96   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 13.55                                    | 35.12                                   |
| 97   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 13.24                                    | 34.87                                   |
| 98   | yolov4_416_tf                | tf_yolov4_coco_416_416_60.3G                          | 13.52                                    | 33.96                                   |
| 99   | yolov4_512_tf                | tf_yolov4_coco_512_512_91.2G                          | 10.25                                    | 25.24                                   |

</details>


### Performance on ZCU104
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 300MHz` DPU configuration:


| No\. | Model                        | GPU Model Standard Name                               | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | bcc_pt                       | pt_BCC_shanghaitech_800_1000_268.9G                   | 3.51                                     | 7.98                                    |
| 2    | c2d2_lite_pt                 | pt_C2D2lite_CC20_512_512_6.86G                        | 3.13                                     | 3.55                                    |
| 3    | centerpoint                  | pt_centerpoint_astyx_2560_40_54G                      | 16.84                                    | 20.72                                   |
| 4    | CLOCs                        | pt_CLOCs_kitti_2.0                                    | 2.89                                     | 10.24                                   |
| 5    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 63.75                                    | 152.73                                  |
| 6    | efficientdet_d2_tf           | tf_efficientdet-d2_coco_768_768_11.06G                | 63.75                                    | 152.73                                  |
| 7    | efficientnet-b0_tf2          | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 83.77                                    | 146.26                                  |
| 8    | efficientNet-edgetpu-L_tf    | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 37.33                                    | 73.44                                   |
| 9    | efficientNet-edgetpu-M_tf    | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 85.42                                    | 168.85                                  |
| 10   | efficientNet-edgetpu-S_tf    | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 122.86                                   | 248.87                                  |
| 11   | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 10.44                                    | 39.45                                   |
| 12   | face_mask_detection_pt       | pt_face-mask-detection_512_512_0.59G                  | 120.72                                   | 326.94                                  |
| 13   | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 3050.46                                  | 8471.64                                 |
| 14   | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 179.71                                   | 314.42                                  |
| 15   | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 992.52                                   | 2122.75                                 |
| 16   | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 2304.48                                  | 5927.11                                 |
| 17   | fadnet                       | pt_fadnet_sceneflow_576_960_441G                      | 1.66                                     | 3.46                                    |
| 18   | fadnet_pruned                | pt_fadnet_sceneflow_576_960_0.65_154G                 | 2.65                                     | 5.28                                    |
| 19   | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 23.83                                    | 52.39                                   |
| 20   | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 39.44                                    | 81.02                                   |
| 21   | HardNet_MSeg_pt              | pt_HardNet_mixed_352_352_22.78G                       | 26.42                                    | 49.36                                   |
| 22   | HFnet_tf                     | tf_HFNet_mixed_960_960_20.09G                         | 3.21                                     | 14.39                                   |
| 23   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 25.54                                    | 47.05                                   |
| 24   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 202.49                                   | 412.79                                  |
| 25   | inception_v2_tf              | tf_inceptionv2_imagenet_224_224_3.88G                 | 98.99                                    | 195.15                                  |
| 26   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 63.43                                    | 121.49                                  |
| 27   | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 62.84                                    | 121.59                                  |
| 28   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 63.53                                    | 122.11                                  |
| 29   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 30.72                                    | 59.03                                   |
| 30   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 163.55                                   | 338.55                                  |
| 31   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 1.87                                     | 5.22                                    |
| 32   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 84.21                                    | 157.65                                  |
| 33   | mobilenet_edge_0_75_tf       | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 278.38                                   | 604.87                                  |
| 34   | mobilenet_edge_1_0_tf        | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 228.03                                   | 476.52                                  |
| 35   | mobilenet_1_0_224_tf2        | tf2_mobilenetv1_imagenet_224_224_1.15G                | 339.56                                   | 770.94                                  |
| 36   | mobilenet_v1_0_25_128_tf     | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 1375.03                                  | 4263.73                                 |
| 37   | mobilenet_v1_0_5_160_tf      | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 950.02                                   | 2614.83                                 |
| 38   | mobilenet_v1_1_0_224_tf      | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 344.85                                   | 783.96                                  |
| 39   | mobilenet_v2_1_0_224_tf      | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 284.18                                   | 609.18                                  |
| 40   | mobilenet_v2_1_4_224_tf      | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 203.20                                   | 413.59                                  |
| 41   | mobilenet_v2_cityscapes_tf   | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 1.81                                     | 5.53                                    |
| 42   | mobilenet_v3_small_1_0_tf2   | tf2_mobilenetv3_imagenet_224_224_132M                 | 363.63                                   | 819.63                                  |
| 43   | movenet_ntd_pt               | pt_movenet_coco_192_192_0.5G                          | 95.38                                    | 362.92                                  |
| 44   | MT-resnet18_mixed_pt         | pt_MT-resnet18_mixed_320_512_13.65G                   | 34.31                                    | 93.07                                   |
| 45   | multi_task_v3_pt             | pt_multitaskv3_mixed_320_512_25.44G                   | 17.76                                    | 55.32                                   |
| 46   | ocr_pt                       | pt_OCR_ICDAR2015_960_960_875G                         | 1.04                                     | 2.56                                    |
| 47   | ofa_depthwise_res50_pt       | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 107.8                                    | 343.72                                  |
| 48   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 17.93                                    | 28.87                                   |
| 49   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 198.24                                   | 354.63                                  |
| 50   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 17.89                                    | 37.46                                   |
| 51   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 22.75                                    | 48.69                                   |
| 52   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 29.17                                    | 64.39                                   |
| 53   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 700.79                                   | 1369.03                                 |
| 54   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 114.82                                   | 216.34                                  |
| 55   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 395.46                                   | 700.55                                  |
| 56   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 161.53                                   | 319.21                                  |
| 57   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 1.31                                     | 4.51                                    |
| 58   | pointpillars_kitti           | pt_pointpillars_kitti_12000_100_10.8G                 | 20.13                                    | 49.83                                   |
| 59   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 2.28                                     | 8.91                                    |
| 60   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 9.17                                     | 17.15                                   |
| 61   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 10.8                                     | 25.87                                   |
| 62   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 71.3                                     | 169.42                                  |
| 63   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 93.84                                    | 175                                     |
| 64   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 49.54                                    | 94.71                                   |
| 65   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 33.81                                    | 64.91                                   |
| 66   | resnet_v2_50_tf              | tf_resnetv2_50_imagenet_299_299_13.1G                 | 48.10                                    | 90.57                                   |
| 67   | resnet_v2_101_tf             | tf_resnetv2_101_imagenet_299_299_26.78G               | 25.23                                    | 47.89                                   |
| 68   | resnet_v2_152_tf             | tf_resnetv2_152_imagenet_299_299_40.47G               | 17.18                                    | 32.61                                   |
| 69   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 93.23                                    | 175.39                                  |
| 70   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 83.56                                    | 157.2                                   |
| 71   | SA_gate_base_pt              | pt_sa-gate_NYUv2_360_360_178G                         | 3.46                                     | 8.52                                    |
| 72   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 5.68                                     | 21.45                                   |
| 73   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 4.30                                     | 11.74                                   |
| 74   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 7.68                                     | 25.54                                   |
| 75   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 36.37                                    | 149.18                                  |
| 76   | SemanticFPN_Mobilenetv2_pt   | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 10.73                                    | 51.97                                   |
| 77   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 94.66                                    | 146.02                                  |
| 78   | solo_pt                      | pt_SOLO_coco_640_640_107G                             | 1.47                                     | 4.82                                    |
| 79   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 599.88                                   | 1315.93                                 |
| 80   | ssd_inception_v2_coco_tf     | tf_ssdinceptionv2_coco_300_300_9.62G                  | 41.64                                    | 87.86                                   |
| 81   | ssd_mobilenet_v1_coco_tf     | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 115.34                                   | 307.17                                  |
| 82   | ssd_mobilenet_v2_coco_tf     | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 85.56                                    | 196.25                                  |
| 83   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 2.91                                     | 5.22                                    |
| 84   | ssdlite_mobilenet_v2_coco_tf | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 110.49                                   | 280.15                                  |
| 85   | ssr_pt                       | pt_SSR_CVC_256_256_39.72G                             | 6.46                                     | 12.37                                   |
| 86   | superpoint_tf                | tf_superpoint_mixed_480_640_52.4G                     | 11.11                                    | 40.32                                   |
| 87   | textmountain_pt              | pt_textmountain_ICDAR_960_960_575.2G                  | 1.77                                     | 3.68                                    |
| 88   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 13.99                                    | 27.78                                   |
| 89   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 37.25                                    | 78.39                                   |
| 90   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 23.63                                    | 59.35                                   |
| 91   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 217.68                                   | 437.02                                  |
| 92   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 216.22                                   | 435.90                                  |
| 93   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 218.02                                   | 437.32                                  |
| 94   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 21.49                                    | 37.10                                   |
| 95   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 18.54                                    | 32.69                                   |
| 96   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 14.44                                    | 28.82                                   |
| 97   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 14.09                                    | 28.64                                   |
| 98   | yolov4_416_tf                | tf_yolov4_coco_416_416_60.3G                          | 14.36                                    | 28.89                                   |
| 99   | yolov4_512_tf                | tf_yolov4_coco_512_512_91.2G                          | 10.93                                    | 21.76                                   |



</details>


### Performance on VCK190
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Versal` board with 96 AIEs running at 1250 MHz:
  

| No\. | Model                        | GPU Model Standard Name                               | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | bcc_pt                       | pt_BCC_shanghaitech_800_1000_268.9G                   | 37.01                                    | 72.51                                   |
| 2    | c2d2_lite_pt                 | pt_C2D2lite_CC20_512_512_6.86G                        | 20.57                                    | 26.49                                   |
| 3    | centerpoint                  | pt_centerpoint_astyx_2560_40_54G                      | 128.12                                   | 236.99                                  |
| 4    | CLOCs                        | pt_CLOCs_kitti_2.0                                    | 8.11                                     | 14.63                                   |
| 5    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 256.13                                   | 459.87                                  |
| 6    | efficientdet_d2_tf           | tf_efficientdet-d2_coco_768_768_11.06G                | 18.1                                     | 40.38                                   |
| 7    | efficientnet-b0_tf2          | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 1090.29                                  | 1811.8                                  |
| 8    | efficientNet-edgetpu-L_tf    | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 397.35                                   | 513.32                                  |
| 9    | efficientNet-edgetpu-M_tf    | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 889.46                                   | 1394.48                                 |
| 10   | efficientNet-edgetpu-S_tf    | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 1209.7                                   | 2192.61                                 |
| 11   | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 25.64                                    | 54.58                                   |
| 12   | face_mask_detection_pt       | pt_face-mask-detection_512_512_0.59G                  | 451.11                                   | 914.73                                  |
| 13   | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 13746.3                                  | 29425.4                                 |
| 14   | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 3849.25                                  | 5691.74                                 |
| 15   | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 7574.57                                  | 19419.9                                 |
| 16   | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 11369.1                                  | 26161.2                                 |
| 17   | fadnet                       | pt_fadnet_sceneflow_576_960_441G                      | 8.02                                     | 13.46                                   |
| 18   | fadnet_pruned                | pt_fadnet_sceneflow_576_960_0.65_154G                 | 9.01                                     | 16.24                                   |
| 19   | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 194.87                                   | 374.48                                  |
| 20   | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 481.49                                   | 788.38                                  |
| 21   | HardNet_MSeg_pt              | pt_HardNet_mixed_352_352_22.78G                       | 205.76                                   | 274.23                                  |
| 22   | HFnet_tf                     | tf_HFNet_mixed_960_960_20.09G                         | 9.37                                     | 22.34                                   |
| 23   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 387.99                                   | 499.77                                  |
| 24   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 1303.24                                  | 2458.23                                 |
| 25   | inception_v2_tf              | tf_inceptionv2_imagenet_224_224_3.88G                 | 827.14                                   | 1190.76                                 |
| 26   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 595.41                                   | 903.47                                  |
| 27   | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 657.37                                   | 1061.04                                 |
| 28   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 592.62                                   | 896.7                                   |
| 29   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 341.84                                   | 424.03                                  |
| 30   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 1528.81                                  | 3034.36                                 |
| 31   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 11.94                                    | 20.92                                   |
| 32   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 1367.1                                   | 2744.24                                 |
| 33   | mobilenet_edge_0_75_tf       | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 1909.12                                  | 4995.3                                  |
| 34   | mobilenet_edge_1_0_tf        | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 1839.19                                  | 4879.14                                 |
| 35   | mobilenet_1_0_224_tf2        | tf2_mobilenetv1_imagenet_224_224_1.15G                | 2026.52                                  | 5049.85                                 |
| 36   | mobilenet_v1_0_25_128_tf     | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 5019.74                                  | 10348.4                                 |
| 37   | mobilenet_v1_0_5_160_tf      | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 3604.67                                  | 7997.22                                 |
| 38   | mobilenet_v1_1_0_224_tf      | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 2022.61                                  | 5015.41                                 |
| 39   | mobilenet_v2_1_0_224_tf      | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 1891.80                                  | 4927.03                                 |
| 40   | mobilenet_v2_1_4_224_tf      | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 1640.06                                  | 4189.87                                 |
| 41   | mobilenet_v2_cityscapes_tf   | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 5.33                                     | 12.03                                   |
| 42   | mobilenet_v3_small_1_0_tf2   | tf2_mobilenetv3_imagenet_224_224_132M                 | 2052.58                                  | 4977.33                                 |
| 43   | movenet_ntd_pt               | pt_movenet_coco_192_192_0.5G                          | 245.96                                   | 445.52                                  |
| 44   | MT-resnet18_mixed_pt         | pt_MT-resnet18_mixed_320_512_13.65G                   | 144.05                                   | 261.02                                  |
| 45   | multi_task_v3_pt             | pt_multitaskv3_mixed_320_512_25.44G                   | 77.73                                    | 173.24                                  |
| 46   | ocr_pt                       | pt_OCR_ICDAR2015_960_960_875G                         | 8.67                                     | 18.67                                   |
| 47   | ofa_depthwise_res50_pt       | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 306.23                                   | 454.3                                   |
| 48   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 60.15                                    | 81.08                                   |
| 49   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 2082.78                                  | 4027.03                                 |
| 50   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 128.26                                   | 223.24                                  |
| 51   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 146.2                                    | 255.8                                   |
| 52   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 167.96                                   | 297.74                                  |
| 53   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 5617.05                                  | 12583.8                                 |
| 54   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 1822.71                                  | 3816.15                                 |
| 55   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 4173.59                                  | 8691.82                                 |
| 56   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 1779.41                                  | 3635.07                                 |
| 57   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 3.8                                      | 6.75                                    |
| 58   | pointpillars_kitti           | pt_pointpillars_kitti_12000_100_10.8G                 | 24.6                                     | 35.29                                   |
| 59   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 7.65                                     | 15.97                                   |
| 60   | psmnet                       | pt_psmnet_sceneflow_576_960_0.68_696G                 | 0.36                                     | 0.71                                    |
| 61   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 45.29                                    | 56.99                                   |
| 62   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 101.95                                   | 224.59                                  |
| 63   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 502.46                                   | 1282.02                                 |
| 64   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 1452.5                                   | 3054.88                                 |
| 65   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 1063.41                                  | 1756.03                                 |
| 66   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 838.21                                   | 1215.98                                 |
| 67   | resnet_v2_50_tf              | tf_resnetv2_50_imagenet_299_299_13.1G                 | 549.19                                   | 794.83                                  |
| 68   | resnet_v2_101_tf             | tf_resnetv2_101_imagenet_299_299_26.78G               | 413.38                                   | 538.87                                  |
| 69   | resnet_v2_152_tf             | tf_resnetv2_152_imagenet_299_299_40.47G               | 331.27                                   | 407.21                                  |
| 70   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 1462.41                                  | 3095.02                                 |
| 71   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 1374.31                                  | 2773.11                                 |
| 72   | SA_gate_base_pt              | pt_sa-gate_NYUv2_360_360_178G                         | 7.61                                     | 9.62                                    |
| 73   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 12.27                                    | 24.06                                   |
| 74   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 10.61                                    | 22.3                                    |
| 75   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 20.36                                    | 47.78                                   |
| 76   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 110.51                                   | 224.58                                  |
| 77   | SemanticFPN_Mobilenetv2_pt   | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 27.15                                    | 55.92                                   |
| 78   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 343.09                                   | 639.38                                  |
| 79   | solo_pt                      | pt_SOLO_coco_640_640_107G                             | 4.36                                     | 7.47                                    |
| 80   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 3143.04                                  | 5797.62                                 |
| 81   | ssd_inception_v2_coco_tf     | tf_ssdinceptionv2_coco_300_300_9.62G                  | 273.59                                   | 381.22                                  |
| 82   | ssd_mobilenet_v1_coco_tf     | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 433.55                                   | 517.66                                  |
| 83   | ssd_mobilenet_v2_coco_tf     | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 387.64                                   | 508.52                                  |
| 84   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 11.2                                     | 12.19                                   |
| 85   | ssdlite_mobilenet_v2_coco_tf | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 408.27                                   | 523.56                                  |
| 86   | ssr_pt                       | pt_SSR_CVC_256_256_39.72G                             | 74.48                                    | 78.02                                   |
| 87   | superpoint_tf                | tf_superpoint_mixed_480_640_52.4G                     | 49.86                                    | 106.38                                  |
| 88   | textmountain_pt              | pt_textmountain_ICDAR_960_960_575.2G                  | 20.92                                    | 29.6                                    |
| 89   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 132.05                                   | 193.27                                  |
| 90   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 392.04                                   | 962.59                                  |
| 91   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 80.52                                    | 242.28                                  |
| 92   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 2036.82                                  | 5191.25                                 |
| 93   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 1986.11                                  | 5191.13                                 |
| 94   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 2053.59                                  | 5251.65                                 |
| 95   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 530.06                                   | 655.44                                  |
| 96   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 480.75                                   | 582.57                                  |
| 97   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 192.68                                   | 292.35                                  |
| 98   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 218.53                                   | 291.4                                   |
| 99   | yolov4_416_tf                | tf_yolov4_coco_416_416_60.3G                          | 141.81                                   | 214.67                                  |
| 100  | yolov4_512_tf                | tf_yolov4_coco_512_512_91.2G                          | 105.09                                   | 154.65                                  |

</details>


### Performance on Kria KV260 SOM
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Kria KV260` board with a `1 * B4096F  @ 300MHz` DPU configuration:
  

| No\. | Model                        | GPU Model Standard Name                               | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | bcc_pt                       | pt_BCC_shanghaitech_800_1000_268.9G                   | 3.55                                     | 4.01                                    |
| 2    | c2d2_lite_pt                 | pt_C2D2lite_CC20_512_512_6.86G                        | 3.24                                     | 3.46                                    |
| 3    | centerpoint                  | pt_centerpoint_astyx_2560_40_54G                      | 17.21                                    | 20.2                                    |
| 4    | CLOCs                        | pt_CLOCs_kitti_2.0                                    | 3.08                                     | 9.05                                    |
| 5    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 65.17                                    | 78.01                                   |
| 6    | efficientdet_d2_tf           | tf_efficientdet-d2_coco_768_768_11.06G                | 3.29                                     | 4.24                                    |
| 7    | efficientnet-b0_tf2          | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 84.26                                    | 87.73                                   |
| 8    | efficientNet-edgetpu-L_tf    | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 37.52                                    | 38.64                                   |
| 9    | efficientNet-edgetpu-M_tf    | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 86.03                                    | 90.05                                   |
| 10   | efficientNet-edgetpu-S_tf    | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 124.06                                   | 131.85                                  |
| 11   | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 11.18                                    | 30.00                                   |
| 12   | face_mask_detection_pt       | pt_face-mask-detection_512_512_0.59G                  | 124.23                                   | 170.38                                  |
| 13   | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 3316.38                                  | 5494.16                                 |
| 14   | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 179.9                                    | 184.56                                  |
| 15   | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 1021.38                                  | 1134.19                                 |
| 16   | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 2472.62                                  | 3560.3                                  |
| 17   | fadnet                       | pt_fadnet_sceneflow_576_960_441G                      | 1.67                                     | 2.26                                    |
| 18   | fadnet_pruned                | pt_fadnet_sceneflow_576_960_0.65_154G                 | 2.78                                     | 4.29                                    |
| 19   | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 24.17                                    | 27.04                                   |
| 20   | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 39.64                                    | 41.25                                   |
| 21   | HardNet_MSeg_pt              | pt_HardNet_mixed_352_352_22.78G                       | 26.66                                    | 27.63                                   |
| 22   | HFnet_tf                     | tf_HFNet_mixed_960_960_20.09G                         | 3.39                                     | 11.24                                   |
| 23   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 25.65                                    | 26.19                                   |
| 24   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 205.17                                   | 227.17                                  |
| 25   | inception_v2_tf              | tf_inceptionv2_imagenet_224_224_3.88G                 | 99.05                                    | 104.45                                  |
| 26   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 63.91                                    | 67.38                                   |
| 27   | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 63.34                                    | 66.68                                   |
| 28   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 64.03                                    | 67.46                                   |
| 29   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 30.83                                    | 31.6                                    |
| 30   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 165.01                                   | 179.4                                   |
| 31   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 1.91                                     | 2.64                                    |
| 32   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 84.83                                    | 88.69                                   |
| 33   | mobilenet_edge_0_75_tf       | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 282.1                                    | 328.78                                  |
| 34   | mobilenet_edge_1_0_tf        | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 232.17                                   | 260.7                                   |
| 35   | mobilenet_1_0_224_tf2        | tf2_mobilenetv1_imagenet_224_224_1.15G                | 347.97                                   | 416.6                                   |
| 36   | mobilenet_v1_0_25_128_tf     | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 1470.42                                  | 2428.88                                 |
| 37   | mobilenet_v1_0_5_160_tf      | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 1002.62                                  | 1492.12                                 |
| 38   | mobilenet_v1_1_0_224_tf      | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 354.19                                   | 425.57                                  |
| 39   | mobilenet_v2_1_0_224_tf      | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 290.62                                   | 336.82                                  |
| 40   | mobilenet_v2_1_4_224_tf      | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 206.04                                   | 228.2                                   |
| 41   | mobilenet_v2_cityscapes_tf   | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 1.90                                     | 3.32                                    |
| 42   | mobilenet_v3_small_1_0_tf2   | tf2_mobilenetv3_imagenet_224_224_132M                 | 374.47                                   | 453.97                                  |
| 43   | movenet_ntd_pt               | pt_movenet_coco_192_192_0.5G                          | 102.8                                    | 351.28                                  |
| 44   | MT-resnet18_mixed_pt         | pt_MT-resnet18_mixed_320_512_13.65G                   | 35.45                                    | 50.52                                   |
| 45   | multi_task_v3_pt             | pt_multitaskv3_mixed_320_512_25.44G                   | 18.57                                    | 29.95                                   |
| 46   | ocr_pt                       | pt_OCR_ICDAR2015_960_960_875G                         | 1.06                                     | 1.29                                    |
| 47   | ofa_depthwise_res50_pt       | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 116.42                                   | 264.3                                   |
| 48   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 18.10                                    | 19.27                                   |
| 49   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 199.09                                   | 214.92                                  |
| 50   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 17.95                                    | 18.89                                   |
| 51   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 22.98                                    | 26.44                                   |
| 52   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 29.19                                    | 35.48                                   |
| 53   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 712.72                                   | 804.15                                  |
| 54   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 115.82                                   | 122.43                                  |
| 55   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 399.80                                   | 436.59                                  |
| 56   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 162.79                                   | 173.08                                  |
| 57   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 1.34                                     | 2.78                                    |
| 58   | pointpillars_kitti           | pt_pointpillars_kitti_12000_100_10.8G                 | 20.81                                    | 32.24                                   |
| 59   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 2.30                                     | 5.46                                    |
| 60   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 9.21                                     | 9.5                                     |
| 61   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 10.90                                    | 13.11                                   |
| 62   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 71.75                                    | 85.60                                   |
| 63   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 94.21                                    | 99.01                                   |
| 64   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 49.86                                    | 51.07                                   |
| 65   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 33.98                                    | 34.54                                   |
| 66   | resnet_v2_50_tf              | tf_resnetv2_50_imagenet_299_299_13.1G                 | 48.42                                    | 50.46                                   |
| 67   | resnet_v2_101_tf             | tf_resnetv2_101_imagenet_299_299_26.78G               | 25.09                                    | 25.87                                   |
| 68   | resnet_v2_152_tf             | tf_resnetv2_152_imagenet_299_299_40.47G               | 17.25                                    | 17.48                                   |
| 69   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 93.80                                    | 98.21                                   |
| 70   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 84.44                                    | 87.96                                   |
| 71   | SA_gate_base_pt              | pt_sa-gate_NYUv2_360_360_178G                         | 3.55                                     | 4.64                                    |
| 72   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 6.11                                     | 19.55                                   |
| 73   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 4.55                                     | 11.46                                   |
| 74   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 8.11                                     | 15.38                                   |
| 75   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 37.36                                    | 86.73                                   |
| 76   | SemanticFPN_Mobilenetv2_pt   | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 11.52                                    | 33.01                                   |
| 77   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 95.84                                    | 105.55                                  |
| 78   | solo_pt                      | pt_SOLO_coco_640_640_107G                             | 1.46                                     | 4.44                                    |
| 79   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 619.72                                   | 762.62                                  |
| 80   | ssd_inception_v2_coco_tf     | tf_ssdinceptionv2_coco_300_300_9.62G                  | 42.06                                    | 47.23                                   |
| 81   | ssd_mobilenet_v1_coco_tf     | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 119.11                                   | 172.23                                  |
| 82   | ssd_mobilenet_v2_coco_tf     | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 87.71                                    | 112.72                                  |
| 83   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 2.96                                     | 5.26                                    |
| 84   | ssdlite_mobilenet_v2_coco_tf | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 111.50                                   | 161.41                                  |
| 85   | ssr_pt                       | pt_SSR_CVC_256_256_39.72G                             | 6.47                                     | 6.50                                    |
| 86   | superpoint_tf                | tf_superpoint_mixed_480_640_52.4G                     | 11.46                                    | 20.57                                   |
| 87   | textmountain_pt              | pt_textmountain_ICDAR_960_960_575.2G                  | 1.78                                     | 1.89                                    |
| 88   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 13.99                                    | 14.62                                   |
| 89   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 37.58                                    | 41.32                                   |
| 90   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 23.88                                    | 30.82                                   |
| 91   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 219.52                                   | 239.61                                  |
| 92   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 217.70                                   | 239.21                                  |
| 93   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 220.38                                   | 239.78                                  |
| 94   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 21.50                                    | 21.72                                   |
| 95   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 18.55                                    | 18.71                                   |
| 96   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 14.48                                    | 14.86                                   |
| 97   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 14.09                                    | 14.76                                   |
| 98   | yolov4_416_tf                | tf_yolov4_coco_416_416_60.3G                          | 14.43                                    | 15.35                                   |
| 99   | yolov4_512_tf                | tf_yolov4_coco_512_512_91.2G                          | 10.98                                    | 11.65                                   |

</details>


### Performance on VCK5000
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H running at 4PE@350
MHz in Gen3x16:

| No\. | Model                        | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | bcc_pt                       | pt_BCC_shanghaitech_800_1000_268.9G                   | 350                | 49.02                                   |
| 2    | c2d2_lite_pt                 | pt_C2D2lite_CC20_512_512_6.86G                        | 350                | 22.48                                   |
| 3    | centerpoint                  | pt_centerpoint_astyx_2560_40_54G                      | 350                | 146.98                                  |
| 4    | CLOCs                        | pt_CLOCs_kitti_2.0                                    | 350                | 11.71                                   |
| 5    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 350                | 94.14                                   |
| 6    | efficientnet-b0_tf2          | tf2_efficientnet-b0_imagenet_224_224_0.36G            | 350                | 426.84                                  |
| 7    | efficientNet-edgetpu-L_tf    | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G     | 350                | 393.64                                  |
| 8    | efficientNet-edgetpu-M_tf    | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 350                | 974.60                                  |
| 9    | efficientNet-edgetpu-S_tf    | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 350                | 1663.81                                 |
| 10   | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 350                | 67.35                                   |
| 11   | face_mask_detection_pt       | pt_face-mask-detection_512_512_0.59G                  | 350                | 770.84                                  |
| 12   | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 350                | 16649.70                                |
| 13   | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 350                | 2954.23                                 |
| 14   | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 350                | 11815.20                                |
| 15   | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 350                | 17826.50                                |
| 16   | fadnet                       | pt_fadnet_sceneflow_576_960_441G                      | 350                | 9.60                                    |
| 17   | fadnet_pruned                | pt_fadnet_sceneflow_576_960_0.65_154G                 | 350                | 10.05                                   |
| 18   | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 350                | 277.87                                  |
| 19   | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 350                | 612.50                                  |
| 20   | HardNet_MSeg_pt              | pt_HardNet_mixed_352_352_22.78G                       | 350                | 158.09                                  |
| 21   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 350                | 307.80                                  |
| 22   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 350                | 1585.16                                 |
| 23   | inception_v2_tf              | tf_inceptionv2_imagenet_224_224_3.88G                 | 350                | 210.04                                  |
| 24   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 350                | 496.16                                  |
| 25   | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 350                | 490.31                                  |
| 26   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 350                | 494.73                                  |
| 27   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 350                | 280.19                                  |
| 28   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 350                | 914.61                                  |
| 29   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 350                | 49.42                                   |
| 30   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 350                | 1638.84                                 |
| 31   | mobilenet_edge_0_75_tf       | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 350                | 3344.23                                 |
| 32   | mobilenet_edge_1_0_tf        | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 350                | 3101.21                                 |
| 33   | mobilenet_1_0_224_tf2        | tf2_mobilenetv1_imagenet_224_224_1.15G                | 350                | 5864.58                                 |
| 34   | mobilenet_v1_0_25_128_tf     | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 350                | 19353.80                                |
| 35   | mobilenet_v1_0_5_160_tf      | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 350                | 13349.20                                |
| 36   | mobilenet_v1_1_0_224_tf      | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 350                | 6561.03                                 |
| 37   | mobilenet_v2_1_0_224_tf      | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 350                | 4058.70                                 |
| 38   | mobilenet_v2_1_4_224_tf      | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 350                | 3137.54                                 |
| 39   | mobilenet_v2_cityscapes_tf   | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 350                | 16.29                                   |
| 40   | mobilenet_v3_small_1_0_tf2   | tf2_mobilenetv3_imagenet_224_224_132M                 | 350                | 1963.18                                 |
| 41   | movenet_ntd_pt               | pt_movenet_coco_192_192_0.5G                          | 350                | 2750.94                                 |
| 42   | MT-resnet18_mixed_pt         | pt_MT-resnet18_mixed_320_512_13.65G                   | 350                | 279.93                                  |
| 43   | multi_task_v3_pt             | pt_multitaskv3_mixed_320_512_25.44G                   | 350                | 136.76                                  |
| 44   | ocr_pt                       | pt_OCR_ICDAR2015_960_960_875G                         | 350                | 24.20                                   |
| 45   | ofa_depthwise_res50_pt       | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 350                | 2948.67                                 |
| 46   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 350                | 33.32                                   |
| 47   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 350                | 2264.41                                 |
| 48   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 350                | 220.24                                  |
| 49   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 350                | 276.05                                  |
| 50   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 350                | 318.73                                  |
| 51   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 350                | 5444.55                                 |
| 52   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 350                | 2012.23                                 |
| 53   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 350                | 4782.06                                 |
| 54   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 350                | 2000.55                                 |
| 55   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 350                | 20.38                                   |
| 56   | pointpillars_kitti           | pt_pointpillars_kitti_12000_100_10.8G                 | 350                | 3.13                                    |
| 57   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 350                | 39.08                                   |
| 58   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 350                | 35.28                                   |
| 59   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 350                | 200.90                                  |
| 60   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 350                | 555.17                                  |
| 61   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 350                | 1817.21                                 |
| 62   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 350                | 1146.44                                 |
| 63   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 350                | 825.95                                  |
| 64   | resnet_v2_50_tf              | tf_resnetv2_50_imagenet_299_299_13.1G                 | 350                | 427.18                                  |
| 65   | resnet_v2_101_tf             | tf_resnetv2_101_imagenet_299_299_26.78G               | 350                | 245.96                                  |
| 66   | resnet_v2_152_tf             | tf_resnetv2_152_imagenet_299_299_40.47G               | 350                | 172.53                                  |
| 67   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 350                | 1533.14                                 |
| 68   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 350                | 1675.05                                 |
| 69   | SA_gate_base_pt              | pt_sa-gate_NYUv2_360_360_178G                         | 350                | 11.28                                   |
| 70   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 350                | 108.73                                  |
| 71   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 350                | 60.46                                   |
| 72   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 350                | 63.70                                   |
| 73   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 350                | 731.00                                  |
| 74   | SemanticFPN_Mobilenetv2_pt   | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 350                | 207.71                                  |
| 75   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 350                | 113.79                                  |
| 76   | solo_pt                      | pt_SOLO_coco_640_640_107G                             | 350                | 24.77                                   |
| 77   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 350                | 3375.85                                 |
| 78   | ssd_inception_v2_coco_tf     | tf_ssdinceptionv2_coco_300_300_9.62G                  | 350                | 105.80                                  |
| 79   | ssd_mobilenet_v1_coco_tf     | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 350                | 2286.02                                 |
| 80   | ssd_mobilenet_v2_coco_tf     | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 350                | 1291.50                                 |
| 81   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 350                | 80.70                                   |
| 82   | ssdlite_mobilenet_v2_coco_tf | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 350                | 1875.26                                 |
| 83   | ssr_pt                       | pt_SSR_CVC_256_256_39.72G                             | 350                | 70.78                                   |
| 84   | textmountain_pt              | pt_textmountain_ICDAR_960_960_575.2G                  | 350                | 30.81                                   |
| 85   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 350                | 192.58                                  |
| 86   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 350                | 592.00                                  |
| 87   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 350                | 128.55                                  |
| 88   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 350                | 2839.65                                 |
| 89   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 350                | 2830.61                                 |
| 90   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 350                | 2840.23                                 |
| 91   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 350                | 328.51                                  |
| 92   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 350                | 293.45                                  |
| 93   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 350                | 279.33                                  |
| 94   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 350                | 276.70                                  |
| 95   | yolov4_416_tf                | tf_yolov4_coco_416_416_60.3G                          | 350                | 240.10                                  |
| 96   | yolov4_512_tf                | tf_yolov4_coco_512_512_91.2G                          | 350                | 137.83                                  |


The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H-aieDWC running at 6PE@350
MHz in Gen3x16:


| No\. | Model                        | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | bcc_pt                       | pt_BCC_shanghaitech_800_1000_268.9G                   | 350                | 79.10                                   |
| 2    | c2d2_lite_pt                 | pt_C2D2lite_CC20_512_512_6.86G                        | 350                | 36.73                                   |
| 3    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 350                | 153.31                                  |
| 4    | efficientNet-edgetpu-M_tf    | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 350                | 1377.58                                 |
| 5    | efficientNet-edgetpu-S_tf    | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 350                | 2488.34                                 |
| 6    | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 350                | 141.03                                  |
| 7    | face_mask_detection_pt       | pt_face-mask-detection_512_512_0.59G                  | 350                | 1335.01                                 |
| 8    | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 350                | 30992.50                                |
| 9    | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 350                | 4411.13                                 |
| 10   | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 350                | 20918.30                                |
| 11   | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 350                | 32349.60                                |
| 12   | fadnet                       | pt_fadnet_sceneflow_576_960_441G                      | 350                | 9.23                                    |
| 13   | fadnet_pruned                | pt_fadnet_sceneflow_576_960_0.65_154G                 | 350                | 10.35                                   |
| 14   | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 350                | 426.19                                  |
| 15   | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 350                | 958.15                                  |
| 16   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 350                | 491.97                                  |
| 17   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 350                | 3500.30                                 |
| 18   | inception_v2_tf              | tf_inceptionv2_imagenet_224_224_3.88G                 | 350                | 330.52                                  |
| 19   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 350                | 914.35                                  |
| 20   | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 350                | 963.51                                  |
| 21   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 350                | 909.00                                  |
| 22   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 350                | 503.54                                  |
| 23   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 350                | 1352.53                                 |
| 24   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 350                | 73.27                                   |
| 25   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 350                | 3403.95                                 |
| 26   | mobilenet_edge_0_75_tf       | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 350                | 5689.70                                 |
| 27   | mobilenet_edge_1_0_tf        | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 350                | 5176.39                                 |
| 28   | mobilenet_1_0_224_tf2        | tf2_mobilenetv1_imagenet_224_224_1.15G                | 350                | 7820.25                                 |
| 29   | mobilenet_v1_0_25_128_tf     | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 350                | 20510.00                                |
| 30   | mobilenet_v1_0_5_160_tf      | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 350                | 14825.30                                |
| 31   | mobilenet_v1_1_0_224_tf      | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 350                | 8058.30                                 |
| 32   | mobilenet_v2_1_0_224_tf      | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 350                | 6504.78                                 |
| 33   | mobilenet_v2_1_4_224_tf      | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 350                | 4971.92                                 |
| 34   | mobilenet_v2_cityscapes_tf   | tf_mobilenetv2_cityscapes_1024_2048_132.74G           | 350                | 18.42                                   |
| 35   | movenet_ntd_pt               | pt_movenet_coco_192_192_0.5G                          | 350                | 2267.16                                 |
| 36   | MT-resnet18_mixed_pt         | pt_MT-resnet18_mixed_320_512_13.65G                   | 350                | 421.09                                  |
| 37   | multi_task_v3_pt             | pt_multitaskv3_mixed_320_512_25.44G                   | 350                | 203.77                                  |
| 38   | ocr_pt                       | pt_OCR_ICDAR2015_960_960_875G                         | 350                | 11.79                                   |
| 39   | ofa_depthwise_res50_pt       | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 350                | 3237.38                                 |
| 40   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 350                | 50.14                                   |
| 41   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 350                | 4013.12                                 |
| 42   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 350                | 285.31                                  |
| 43   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 350                | 376.22                                  |
| 44   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 350                | 441.71                                  |
| 45   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 350                | 9448.00                                 |
| 46   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 350                | 3750.97                                 |
| 47   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 350                | 8047.31                                 |
| 48   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 350                | 3423.98                                 |
| 49   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 350                | 18.09                                   |
| 50   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 350                | 37.10                                   |
| 51   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 350                | 53.41                                   |
| 52   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 350                | 307.64                                  |
| 53   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 350                | 998.94                                  |
| 54   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 350                | 3736.48                                 |
| 55   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 350                | 2244.46                                 |
| 56   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 350                | 1598.02                                 |
| 57   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 350                | 3152.91                                 |
| 58   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 350                | 3432.18                                 |
| 59   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 350                | 158.09                                  |
| 60   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 350                | 91.22                                   |
| 61   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 350                | 114.93                                  |
| 62   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 350                | 1049.65                                 |
| 63   | SemanticFPN_Mobilenetv2_pt   | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 350                | 234.42                                  |
| 64   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 350                | 185.87                                  |
| 65   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 350                | 7062.16                                 |
| 66   | ssd_inception_v2_coco_tf     | tf_ssdinceptionv2_coco_300_300_9.62G                  | 350                | 166.21                                  |
| 67   | ssd_mobilenet_v1_coco_tf     | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 350                | 2471.66                                 |
| 68   | ssd_mobilenet_v2_coco_tf     | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 350                | 1879.99                                 |
| 69   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 350                | 108.15                                  |
| 70   | ssdlite_mobilenet_v2_coco_tf | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 350                | 2552.13                                 |
| 71   | ssr_pt                       | pt_SSR_CVC_256_256_39.72G                             | 350                | 90.06                                   |
| 72   | textmountain_pt              | pt_textmountain_ICDAR_960_960_575.2G                  | 350                | 41.83                                   |
| 73   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 350                | 261.90                                  |
| 74   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 350                | 1039.88                                 |
| 75   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 350                | 201.48                                  |
| 76   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 350                | 5301.04                                 |
| 77   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 350                | 5284.14                                 |
| 78   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 350                | 5300.12                                 |
| 79   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 350                | 500.50                                  |
| 80   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 350                | 446.08                                  |
| 81   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 350                | 389.71                                  |
| 82   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 350                | 385.25                                  |

  
The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H-aieMISC running at 6PE@350
MHz in Gen3x16:


| No\. | Model                        | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | centerpoint                  | pt_centerpoint_astyx_2560_40_54G                      | 350                | 10.83                                   |
| 2    | CLOCs                        | pt_CLOCs_kitti_2.0                                    | 350                | 17.10                                   |
| 3    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 350                | 141.07                                  |
| 4    | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 350                | 91.51                                   |
| 5    | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 350                | 24426.30                                |
| 6    | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 350                | 4397.97                                 |
| 7    | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 350                | 17183.60                                |
| 8    | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 350                | 25900.50                                |
| 9    | fadnet                       | pt_fadnet_sceneflow_576_960_441G                      | 350                | 8.91                                    |
| 10   | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 350                | 381.48                                  |
| 11   | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 350                | 880.96                                  |
| 12   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 350                | 448.06                                  |
| 13   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 350                | 2224.44                                 |
| 14   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 350                | 711.53                                  |
|      | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 350                | 719.18                                  |
| 16   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 350                | 710.52                                  |
| 17   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 350                | 397.71                                  |
| 18   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 350                | 1294.82                                 |
| 19   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 350                | 67.65                                   |
| 20   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 350                | 2454.78                                 |
| 21   | ocr_pt                       | pt_OCR_ICDAR2015_960_960_875G                         | 350                | 33.85                                   |
| 22   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 350                | 49.92                                   |
| 23   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 350                | 3305.77                                 |
| 24   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 350                | 279.77                                  |
| 25   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 350                | 366.23                                  |
| 26   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 350                | 424.64                                  |
| 27   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 350                | 8060.23                                 |
| 28   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 350                | 3012.43                                 |
| 29   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 350                | 7050.56                                 |
| 30   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 350                | 2887.85                                 |
| 31   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 350                | 15.55                                   |
| 32   | pointpillars_kitti           | pt_pointpillars_kitti_12000_100_10.8G                 | 350                | 3.10                                    |
| 33   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 350                | 34.46                                   |
| 34   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 350                | 52.80                                   |
| 35   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 350                | 292.19                                  |
| 36   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 350                | 828.60                                  |
| 37   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 350                | 2718.97                                 |
| 38   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 350                | 1715.80                                 |
| 39   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 350                | 1237.54                                 |
| 40   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 350                | 2306.45                                 |
| 41   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 350                | 2507.43                                 |
| 42   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 350                | 152.58                                  |
| 43   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 350                | 83.71                                   |
| 44   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 350                | 80.99                                   |
| 45   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 350                | 992.31                                  |
| 46   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 350                | 170.61                                  |
| 47   | solo_pt                      | pt_SOLO_coco_640_640_107G                             | 350                | 24.13                                   |
| 48   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 350                | 4617.91                                 |
| 49   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 350                | 105.05                                  |
| 50   | textmountain_pt              | pt_textmountain_ICDAR_960_960_575.2G                  | 350                | 40.61                                   |
| 51   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 350                | 257.27                                  |
| 52   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 350                | 870.67                                  |
| 53   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 350                | 185.08                                  |
| 54   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 350                | 4166.42                                 |
| 55   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 350                | 4161.00                                 |
| 56   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 350                | 4168.73                                 |
| 57   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 350                | 478.68                                  |
| 58   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 350                | 428.68                                  |
| 59   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 350                | 386.94                                  |
| 60   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 350                | 382.30                                  |
| 61   | yolov4_416_tf                | tf_yolov4_coco_416_416_60.3G                          | 350                | 315.31                                  |
| 62   | yolov4_512_tf                | tf_yolov4_coco_512_512_91.2G                          | 350                | 171.52                                  |


The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H running at 8PE@350
MHz in Gen3x16:


| No\. | Model                       | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :-------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | drunet_pt                   | pt_DRUNet_Kvasir_528_608_0.4G                         | 350                | 204.59                                  |
| 2    | ENet_cityscapes_pt          | pt_ENet_cityscapes_512_1024_8.6G                      | 350                | 148.32                                  |
| 3    | face-quality_pt             | pt_face-quality_80_60_61.68M                          | 350                | 31080.90                                |
| 4    | fadnet                      | pt_fadnet_sceneflow_576_960_441G                      | 350                | 9.65                                    |
| 5    | FairMot_pt                  | pt_FairMOT_mixed_640_480_0.5_36G                      | 350                | 506.42                                  |
| 6    | FPN-resnet18_covid19-seg_pt | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 350                | 1177.00                                 |
| 7    | Inception_v1_tf             | tf_inceptionv1_imagenet_224_224_3G                    | 350                | 4224.61                                 |
| 8    | medical_seg_cell_tf2        | tf2_2d-unet_nuclei_128_128_5.31G                      | 350                | 1504.80                                 |
| 9    | mlperf_ssd_resnet34_tf      | tf_mlperf_resnet34_coco_1200_1200_433G                | 350                | 76.78                                   |
| 10   | mlperf_resnet50_tf          | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 350                | 4519.23                                 |
| 11   | ocr_pt                      | pt_OCR_ICDAR2015_960_960_875G                         | 350                | 12.69                                   |
| 12   | ofa_rcan_latency_pt         | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 350                | 66.81                                   |
| 13   | rcan_pruned_tf              | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 350                | 71.19                                   |
| 14   | refinedet_VOC_tf            | tf_refinedet_VOC_320_320_81.9G                        | 350                | 399.88                                  |
| 15   | RefineDet-Medical_EDD_tf    | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 350                | 1265.40                                 |
| 16   | resnet_v1_50_tf             | tf_resnetv1_50_imagenet_224_224_6.97G                 | 350                | 4944.30                                 |
| 17   | resnet_v1_101_tf            | tf_resnetv1_101_imagenet_224_224_14.4G                | 350                | 2979.15                                 |
| 18   | resnet_v1_152_tf            | tf_resnetv1_152_imagenet_224_224_21.83G               | 350                | 2122.98                                 |
| 19   | resnet50_tf2                | tf2_resnet50_imagenet_224_224_7.76G                   | 350                | 4167.80                                 |
| 20   | resnet50_pt                 | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 350                | 4544.06                                 |
| 21   | salsanext_pt                | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 350                | 159.51                                  |
| 22   | salsanext_v2_pt             | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 350                | 89.41                                   |
| 23   | semantic_seg_citys_tf2      | tf2_erfnet_cityscapes_512_1024_54G                    | 350                | 117.98                                  |
| 24   | SemanticFPN_cityscapes_pt   | pt_SemanticFPN_cityscapes_256_512_10G                 | 350                | 1084.85                                 |
| 25   | SESR_S_pt                   | pt_SESR-S_DIV2K_360_640_7.48G                         | 350                | 233.24                                  |
| 26   | SqueezeNet_pt               | pt_squeezenet_imagenet_224_224_351.7M                 | 350                | 7536.39                                 |
| 27   | ssd_resnet_50_fpn_coco_tf   | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 350                | 114.53                                  |
| 28   | textmountain_pt             | pt_textmountain_ICDAR_960_960_575.2G                  | 350                | 49.48                                   |
| 29   | ultrafast_pt                | pt_ultrafast_CULane_288_800_8.4G                      | 350                | 1359.92                                 |
| 30   | unet_chaos-CT_pt            | pt_unet_chaos-CT_512_512_23.3G                        | 350                | 247.93                                  |
| 31   | chen_color_resnet18_pt      | pt_vehicle-color-classification_color_224_224_3.63G   | 350                | 6526.42                                 |
| 32   | vehicle_make_resnet18_pt    | pt_vehicle-make-classification_CompCars_224_224_3.63G | 350                | 6910.37                                 |
| 33   | vehicle_type_resnet18_pt    | pt_vehicle-type-classification_CompCars_224_224_3.63G | 350                | 6945.23                                 |
| 34   | yolov3_voc_tf               | tf_yolov3_voc_416_416_65.63G                          | 350                | 476.97                                  |
| 35   | yolov3_coco_tf2             | tf2_yolov3_coco_416_416_65.9G                         | 350                | 469.79                                  |


</details>


### Performance on U50lv
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for each model on the `Alveo U50lv` board with 10 DPUCAHX8H kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                        | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 165                | 336.95                                  |
| 2    | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 165                | 87.66                                   |
| 3    | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 165                | 21381.20                                |
| 4    | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 165                | 1251.82                                 |
| 5    | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 165                | 7323.87                                 |
| 6    | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 165                | 21136.40                                |
| 7    | fadnet_pt                    | pt_fadnet_sceneflow_576_960_441G                      | 165                | 1.16                                    |
| 8    | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 165                | 138.17                                  |
| 9    | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 165                | 212.52                                  |
| 10   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 165                | 158.87                                  |
| 11   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 165                | 1172.51                                 |
| 12   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 165                | 376.13                                  |
| 13   | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 165                | 381.25                                  |
| 14   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 165                | 436.47                                  |
| 15   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 165                | 170.86                                  |
| 16   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 165                | 1068.04                                 |
| 17   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 165                | 511.80                                  |
| 18   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 165                | 14.11                                   |
| 19   | ocr_pt                       | pt_OCR_ICDAR2015_960_960_875G                         | 165                | 4.85                                    |
| 20   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 165                | 45.04                                   |
| 21   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 165                | 1548.17                                 |
| 22   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 165                | 98.38                                   |
| 23   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 165                | 124.95                                  |
| 24   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 165                | 159.48                                  |
| 25   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 165                | 5858.88                                 |
| 26   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 165                | 825.99                                  |
| 27   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 165                | 3502.12                                 |
| 28   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 165                | 1005.87                                 |
| 29   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 165                | 11.36                                   |
| 30   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 165                | 21.16                                   |
| 31   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 165                | 46.73                                   |
| 32   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 165                | 72.55                                   |
| 33   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 165                | 430.81                                  |
| 34   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 165                | 593.21                                  |
| 35   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 165                | 307.93                                  |
| 36   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 165                | 205.48                                  |
| 37   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 165                | 593.40                                  |
| 38   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 165                | 511.61                                  |
| 39   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 165                | 148.27                                  |
| 40   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 165                | 32.70                                   |
| 41   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 165                | 56.95                                   |
| 42   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 165                | 432.47                                  |
| 43   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 165                | 188.92                                  |
| 44   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 165                | 4127.72                                 |
| 45   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 165                | 32.34                                   |
| 46   | textmountain_pt              | pt_textmountain_ICDAR_960_960_575.2G                  | 165                | 4.34                                    |
| 47   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 165                | 71.97                                   |
| 48   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 165                | 247.12                                  |
| 49   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 165                | 86.41                                   |
| 50   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 165                | 1316.57                                 |
| 51   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 165                | 1314.66                                 |
| 52   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 165                | 1099.59                                 |
| 53   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 165                | 151.61                                  |
| 54   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 165                | 125.71                                  |
| 55   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 165                | 78.62                                   |
| 56   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 165                | 77.83                                   |


The following table lists the performance number including end-to-end throughput for each model on the `Alveo U50lv` board with 8 DPUCAHX8H-DWC kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                        | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | bcc_pt                       | pt_BCC_shanghaitech_800_1000_268.9G                   | 165                | 16.67                                   |
| 2    | c2d2_lite_pt                 | pt_C2D2lite_CC20_512_512_6.86G                        | 165                | 17.55                                   |
| 3    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 165                | 268.16                                  |
| 4    | efficientNet-edgetpu-M_tf    | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 165                | 543.66                                  |
| 5    | efficientNet-edgetpu-S_tf    | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 165                | 883.96                                  |
| 6    | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 165                | 69.98                                   |
| 7    | face_mask_detection_pt       | pt_face-mask-detection_512_512_0.59G                  | 165                | 737.28                                  |
| 8    | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 165                | 20109.10                                |
| 9    | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 165                | 1001.98                                 |
| 10   | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 165                | 5908.28                                 |
| 11   | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 165                | 17368.60                                |
| 12   | fadnet_pt                    | pt_fadnet_sceneflow_576_960_441G                      | 165                | 3.86                                    |
| 13   | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 165                | 110.50                                  |
| 14   | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 165                | 170.04                                  |
| 15   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 165                | 127.05                                  |
| 16   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 165                | 938.83                                  |
| 17   | inception_v2_tf              | tf_inceptionv2_imagenet_224_224_3.88G                 | 165                | 322.89                                  |
| 18   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 165                | 300.88                                  |
| 19   | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 165                | 305.50                                  |
| 20   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 165                | 300.75                                  |
| 21   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 165                | 136.73                                  |
| 22   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 165                | 855.48                                  |
| 23   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 165                | 11.29                                   |
| 24   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 165                | 409.83                                  |
| 25   | mobilenet_edge_0_75_tf       | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 165                | 1954.19                                 |
| 26   | mobilenet_edge_1_0_tf        | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 165                | 1583.69                                 |
| 27   | mobilenet_1_0_224_tf2        | tf2_mobilenetv1_imagenet_224_224_1.15G                | 165                | 2415.83                                 |
| 28   | mobilenet_v1_0_25_128_tf     | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 165                | 14834.80                                |
| 29   | mobilenet_v1_0_5_160_tf      | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 165                | 8837.20                                 |
| 30   | mobilenet_v1_1_0_224_tf      | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 165                | 2415.28                                 |
| 31   | mobilenet_v2_1_0_224_tf      | tf_mobilenetv2_1.0_imagenet_224_224_602M              | 165                | 2226.09                                 |
| 32   | mobilenet_v2_1_4_224_tf      | tf_mobilenetv2_1.4_imagenet_224_224_1.16G             | 165                | 1498.50                                 |
| 33   | movenet_ntd_pt               | pt_movenet_coco_192_192_0.5G                          | 165                | 1747.97                                 |
| 34   | MT-resnet18_mixed_pt         | pt_MT-resnet18_mixed_320_512_13.65G                   | 165                | 217.50                                  |
| 35   | multi_task_v3_pt             | pt_multitaskv3_mixed_320_512_25.44G                   | 165                | 118.96                                  |
| 36   | ocr_pt                       | pt_OCR_ICDAR2015_960_960_875G                         | 165                | 4.72                                    |
| 37   | ofa_depthwise_res50_pt       | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 165                | 2269.73                                 |
| 38   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 165                | 36.06                                   |
| 39   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 165                | 1242.57                                 |
| 40   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 165                | 78.87                                   |
| 41   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 165                | 99.68                                   |
| 42   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 165                | 127.52                                  |
| 43   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 165                | 4727.32                                 |
| 44   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 165                | 661.48                                  |
| 45   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 165                | 2813.31                                 |
| 46   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 165                | 806.20                                  |
| 47   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 165                | 11.33                                   |
| 48   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 165                | 20.55                                   |
| 49   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 165                | 37.41                                   |
| 50   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 165                | 58.01                                   |
| 51   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 165                | 345.01                                  |
| 52   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 165                | 475.39                                  |
| 53   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 165                | 246.71                                  |
| 54   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 165                | 164.32                                  |
| 55   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 165                | 475.35                                  |
| 56   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 165                | 409.64                                  |
| 57   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 165                | 139.21                                  |
| 58   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 165                | 28.22                                   |
| 59   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 165                | 47.44                                   |
| 60   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 165                | 346.75                                  |
| 61   | SemanticFPN_Mobilenetv2_pt   | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 165                | 135.54                                  |
| 62   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 165                | 151.05                                  |
| 63   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 165                | 2884.24                                 |
| 64   | ssd_inception_v2_coco_tf     | tf_ssdinceptionv2_coco_300_300_9.62G                  | 165                | 156.87                                  |
| 65   | ssd_mobilenet_v1_coco_tf     | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 165                | 1127.75                                 |
| 66   | ssd_mobilenet_v2_coco_tf     | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 165                | 635.81                                  |
| 67   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 165                | 25.93                                   |
| 68   | ssdlite_mobilenet_v2_coco_tf | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 165                | 1045.29                                 |
| 69   | ssr_pt                       | pt_SSR_CVC_256_256_39.72G                             | 165                | 26.84                                   |
| 70   | textmountain_pt              | pt_textmountain_ICDAR_960_960_575.2G                  | 165                | 7.83                                    |
| 71   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 165                | 57.60                                   |
| 72   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 165                | 197.57                                  |
| 73   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 165                | 69.14                                   |
| 74   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 165                | 1056.01                                 |
| 75   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 165                | 1054.04                                 |
| 76   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 165                | 1055.42                                 |
| 77   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 165                | 121.12                                  |
| 78   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 165                | 100.65                                  |
| 79   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 165                | 62.93                                   |
| 80   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 165                | 62.35                                   |

  
</details>


### Performance on U55C DWC
Measured with Vitis AI 2.5 and Vitis AI Library 2.5 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for each model on the `Alveo U55C` board with 11 DPUCAHX8H-DWC kernels running at 300Mhz in Gen3x4:
  

| No\. | Model                        | GPU Model Standard Name                               | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------------- | :---------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | bcc_pt                       | pt_BCC_shanghaitech_800_1000_268.9G                   | 300                | 37.43                                   |
| 2    | c2d2_lite_pt                 | pt_C2D2lite_CC20_512_512_6.86G                        | 300                | 28.25                                   |
| 3    | drunet_pt                    | pt_DRUNet_Kvasir_528_608_0.4G                         | 300                | 469.21                                  |
| 4    | efficientNet-edgetpu-M_tf    | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G      | 300                | 1204.68                                 |
| 5    | efficientNet-edgetpu-S_tf    | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G      | 300                | 2000.42                                 |
| 6    | ENet_cityscapes_pt           | pt_ENet_cityscapes_512_1024_8.6G                      | 300                | 147.29                                  |
| 7    | face_mask_detection_pt       | pt_face-mask-detection_512_512_0.59G                  | 300                | 1581.35                                 |
| 8    | face-quality_pt              | pt_face-quality_80_60_61.68M                          | 300                | 30918.60                                |
| 9    | facerec-resnet20_mixed_pt    | pt_facerec-resnet20_mixed_112_96_3.5G                 | 300                | 2423.09                                 |
| 10   | facereid-large_pt            | pt_facereid-large_96_96_515M                          | 300                | 15138.20                                |
| 11   | facereid-small_pt            | pt_facereid-small_80_80_90M                           | 300                | 32819.60                                |
| 12   | FairMot_pt                   | pt_FairMOT_mixed_640_480_0.5_36G                      | 300                | 239.29                                  |
| 13   | FPN-resnet18_covid19-seg_pt  | pt_FPN-resnet18_covid19-seg_352_352_22.7G             | 300                | 401.98                                  |
| 14   | Inception_resnet_v2_tf       | tf_inceptionresnetv2_imagenet_299_299_26.35G          | 300                | 298.63                                  |
| 15   | Inception_v1_tf              | tf_inceptionv1_imagenet_224_224_3G                    | 300                | 2185.25                                 |
| 16   | inception_v2_tf              | tf_inceptionv2_imagenet_224_224_3.88G                 | 300                | 672.69                                  |
| 17   | Inception_v3_tf              | tf_inceptionv3_imagenet_299_299_11.45G                | 300                | 691.41                                  |
| 18   | Inception_v3_tf2             | tf2_inceptionv3_imagenet_299_299_11.5G                | 300                | 709.43                                  |
| 19   | Inception_v3_pt              | pt_inceptionv3_imagenet_299_299_5.7G                  | 300                | 690.40                                  |
| 20   | Inception_v4_tf              | tf_inceptionv4_imagenet_299_299_24.55G                | 300                | 324.68                                  |
| 21   | medical_seg_cell_tf2         | tf2_2d-unet_nuclei_128_128_5.31G                      | 300                | 1972.68                                 |
| 22   | mlperf_ssd_resnet34_tf       | tf_mlperf_resnet34_coco_1200_1200_433G                | 300                | 25.84                                   |
| 23   | mlperf_resnet50_tf           | tf_mlperf_resnet50_imagenet_224_224_8.19G             | 300                | 1012.03                                 |
| 24   | mobilenet_edge_0_75_tf       | tf_mobilenetEdge0.75_imagenet_224_224_624M            | 300                | 4716.44                                 |
| 25   | mobilenet_edge_1_0_tf        | tf_mobilenetEdge1.0_imagenet_224_224_990M             | 300                | 3834.40                                 |
| 26   | mobilenet_1_0_224_tf2        | tf2_mobilenetv1_imagenet_224_224_1.15G                | 300                | 5305.78                                 |
| 27   | mobilenet_v1_0_25_128_tf     | tf_mobilenetv1_0.25_imagenet_128_128_27M              | 300                | 18873.80                                |
| 28   | mobilenet_v1_0_5_160_tf      | tf_mobilenetv1_0.5_imagenet_160_160_150M              | 300                | 12862.50                                |
| 29   | mobilenet_v1_1_0_224_tf      | tf_mobilenetv1_1.0_imagenet_224_224_1.14G             | 300                | 5305.64                                 |
| 30   | MT-resnet18_mixed_pt         | pt_MT-resnet18_mixed_320_512_13.65G                   | 300                | 511.94                                  |
| 31   | multi_task_v3_pt             | pt_multitaskv3_mixed_320_512_25.44G                   | 300                | 291.27                                  |
| 32   | ofa_depthwise_res50_pt       | pt_OFA-depthwise-res50_imagenet_176_176_2.49G         | 300                | 2887.10                                 |
| 33   | ofa_rcan_latency_pt          | pt_OFA-rcan_DIV2K_360_640_45.7G                       | 300                | 65.53                                   |
| 34   | ofa_resnet50_0_9B_pt         | pt_OFA-resnet50_imagenet_160_160_1.8G                 | 300                | 2922.00                                 |
| 35   | ofa_yolo_pt                  | pt_OFA-yolo_coco_640_640_48.88G                       | 300                | 178.70                                  |
| 36   | ofa_yolo_pruned_0_30_pt      | pt_OFA-yolo_coco_640_640_0.3_34.72G                   | 300                | 223.19                                  |
| 37   | ofa_yolo_pruned_0_50_pt      | pt_OFA-yolo_coco_640_640_0.6_24.62G                   | 300                | 283.17                                  |
| 38   | person-orientation_pruned_pt | pt_person-orientation_224_112_558M                    | 300                | 11379.50                                |
| 39   | personreid-res50_pt          | pt_personreid-res50_market1501_256_128_5.3G           | 300                | 1611.62                                 |
| 40   | personreid-res18_pt          | pt_personreid-res18_market1501_176_80_1.1G            | 300                | 6648.41                                 |
| 41   | pmg_pt                       | pt_pmg_rp2k_224_224_2.28G                             | 300                | 1989.61                                 |
| 42   | pointpainting_nuscenes       | pt_pointpainting_nuscenes_126G_2.5                    | 300                | 20.11                                   |
| 43   | pointpillars_nuscenes        | pt_pointpillars_nuscenes_40000_64_108G                | 300                | 39.93                                   |
| 44   | rcan_pruned_tf               | tf_rcan_DIV2K_360_640_0.98_86.95G                     | 300                | 80.06                                   |
| 45   | refinedet_VOC_tf             | tf_refinedet_VOC_320_320_81.9G                        | 300                | 138.34                                  |
| 46   | RefineDet-Medical_EDD_tf     | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G           | 300                | 789.95                                  |
| 47   | resnet_v1_50_tf              | tf_resnetv1_50_imagenet_224_224_6.97G                 | 300                | 1175.61                                 |
| 48   | resnet_v1_101_tf             | tf_resnetv1_101_imagenet_224_224_14.4G                | 300                | 610.35                                  |
| 49   | resnet_v1_152_tf             | tf_resnetv1_152_imagenet_224_224_21.83G               | 300                | 407.06                                  |
| 50   | resnet50_tf2                 | tf2_resnet50_imagenet_224_224_7.76G                   | 300                | 1174.21                                 |
| 51   | resnet50_pt                  | pt_resnet50_imagenet_224_224_4.1G_2.0                 | 300                | 1012.40                                 |
| 52   | salsanext_pt                 | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G         | 300                | 153.16                                  |
| 53   | salsanext_v2_pt              | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G        | 300                | 58.41                                   |
| 54   | semantic_seg_citys_tf2       | tf2_erfnet_cityscapes_512_1024_54G                    | 300                | 90.03                                   |
| 55   | SemanticFPN_cityscapes_pt    | pt_SemanticFPN_cityscapes_256_512_10G                 | 300                | 803.50                                  |
| 56   | SemanticFPN_Mobilenetv2_pt   | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G   | 300                | 228.29                                  |
| 57   | SESR_S_pt                    | pt_SESR-S_DIV2K_360_640_7.48G                         | 300                | 282.56                                  |
| 58   | SqueezeNet_pt                | pt_squeezenet_imagenet_224_224_351.7M                 | 300                | 6398.41                                 |
| 59   | ssd_inception_v2_coco_tf     | tf_ssdinceptionv2_coco_300_300_9.62G                  | 300                | 329.92                                  |
| 60   | ssd_mobilenet_v1_coco_tf     | tf_ssdmobilenetv1_coco_300_300_2.47G                  | 300                | 2115.80                                 |
| 61   | ssd_mobilenet_v2_coco_tf     | tf_ssdmobilenetv2_coco_300_300_3.75G                  | 300                | 1458.46                                 |
| 62   | ssd_resnet_50_fpn_coco_tf    | tf_ssdresnet50v1_fpn_coco_640_640_178.4G              | 300                | 58.99                                   |
| 63   | ssdlite_mobilenet_v2_coco_tf | tf_ssdlite_mobilenetv2_coco_300_300_1.5G              | 300                | 2061.95                                 |
| 64   | ssr_pt                       | pt_SSR_CVC_256_256_39.72G                             | 300                | 62.02                                   |
| 65   | tsd_yolox_pt                 | pt_yolox_TT100K_640_640_73G                           | 300                | 131.89                                  |
| 66   | ultrafast_pt                 | pt_ultrafast_CULane_288_800_8.4G                      | 300                | 474.24                                  |
| 67   | unet_chaos-CT_pt             | pt_unet_chaos-CT_512_512_23.3G                        | 300                | 136.80                                  |
| 68   | chen_color_resnet18_pt       | pt_vehicle-color-classification_color_224_224_3.63G   | 300                | 2645.11                                 |
| 69   | vehicle_make_resnet18_pt     | pt_vehicle-make-classification_CompCars_224_224_3.63G | 300                | 2634.76                                 |
| 70   | vehicle_type_resnet18_pt     | pt_vehicle-type-classification_CompCars_224_224_3.63G | 300                | 2642.15                                 |
| 71   | vgg_16_tf                    | tf_vgg16_imagenet_224_224_30.96G                      | 300                | 283.80                                  |
| 72   | vgg_19_tf                    | tf_vgg19_imagenet_224_224_39.28G                      | 300                | 238.41                                  |
| 73   | yolov3_voc_tf                | tf_yolov3_voc_416_416_65.63G                          | 300                | 152.75                                  |
| 74   | yolov3_coco_tf2              | tf2_yolov3_coco_416_416_65.9G                         | 300                | 150.59                                  |


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
