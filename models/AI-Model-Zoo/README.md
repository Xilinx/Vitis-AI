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

## Vitis AI 1.4 Model Zoo New Features！
1.New 16 models and total 108 models with diverse deep learning frameworks (Caffe, TensorFlow, TensorFlow 2 and PyTorch) in Vitis AI 1.4.</br>

2.The diversity of AI Model Zoo is significantly improved compared to Vitis AI 1.3.</br>

a.For autonomous driving and ADAS, provide 4D radar detection, image-lidar sensor fusion, surround-view 3D detection, upgraded 3D segmentation and upgraded multi-task models.</br>
b.For medical and industrial vision, provide depth estimation, RGB-D segmentation, super resolution and many other reference models.</br>

3.EoU enhancement: provide automated download script, freely select the version according to model name and hardware platform.</br>

## Model Details
The following table includes comprehensive information about all models, including application, framework, training and validation dataset, input size, computation as well as float and quantized precision.

<details>
 <summary><b>Click here to view details</b></summary>

| No\. | Application                   | Name                                                      | Framework  |                    Input Size                     |             OPS per image             |                      Training Set                       |                          Val Set                          |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |
| :--: | :---------------------------- | :-------------------------------------------------------- | :--------: | :-----------------------------------------------: | :-----------------------------------: | :-----------------------------------------------------: | :-------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  1   | Image Classification          | cf\_resnet50\_imagenet\_224\_224\_7\.7G\_1\.4             |   caffe    |                     224\*224                      |                 7\.7G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7444/0\.9185                        |                       0\.7335/0\.9130                        |
|  2   | Image Classifiction           | cf\_resnet18\_imagenet\_224\_224\_3\.65G\_1\.4            |   caffe    |                     224\*224                      |                3\.65G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.6844/0\.8864                        |                        0.6699/0.8826                         |
|  3   | Image Classification          | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G\_1\.4         |   caffe    |                     224\*224                      |                3\.16G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7030/0\.8971                        |                       0\.6964/0\.8942                        |
|  4   | Image Classification          | cf\_inceptionv2\_imagenet\_224\_224\_4G\_1\.4             |   caffe    |                     224\*224                      |                  4G                   |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7275/0\.9111                        |                       0\.7166/0\.9029                        |
|  5   | Image Classification          | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G\_1\.4         |   caffe    |                      299*299                      |                11\.4G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7701/0\.9329                        |                       0\.7622/0\.9302                        |
|  6   | Image Classification          | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G\_1\.4         |   caffe    |                     299\*299                      |                24\.5G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7958/0\.9470                        |                       0\.7899/0\.9444                        |
|  7   | Image Classification          | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G\_1\.4         |   caffe    |                     224\*224                      |                 608M                  |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.6527/0\.8643                        |                       0\.6399/0\.8540                        |
|  8   | Image Classifiction           | cf_squeezenet_imagenet_227_227_0.76G_1.4                  |   caffe    |                     227\*227                      |                0\.76G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.5432/0\.7824                        |                       0\.5270/0\.7816                        |
|  9   | ADAS Pedstrain Detection      | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G\_1\.4     |   caffe    |                     360\*640                      |                 5\.9G                 |         coco2014\_train\_person and crowndhuman         |                   coco2014\_val\_person                   |                           0\.5903                            |                           0\.5857                            |
|  10  | Object Detection              | cf_refinedet_coco_360_480_123G_1.4                        |   caffe    |                     360\*480                      |                 123G                  |                 coco2014\_train\_person                 |                   coco2014\_val\_person                   |                            0.6928                            |                            0.7042                            |
|  11  | Object Detection              | cf\_refinedet\_coco\_360\_480\_0\.8\_25G\_1\.4            |   caffe    |                     360\*480                      |                  25G                  |                 coco2014\_train\_person                 |                   coco2014\_val\_person                   |                           0\.6794                            |                           0\.6790                            |
|  12  | Object Detection              | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G\_1\.4       |   caffe    |                     360\*480                      |                10\.10G                |                 coco2014\_train\_person                 |                   coco2014\_val\_person                   |                           0\.6489                            |                           0\.6487                            |
|  13  | Object Detection              | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G\_1\.4        |   caffe    |                     360\*480                      |                5\.08G                 |                 coco2014\_train\_person                 |                   coco2014\_val\_person                   |                           0\.6120                            |                           0\.6112                            |
|  14  | ADAS Vehicle Detection        | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G\_1\.4            |   caffe    |                     360\*480                      |                 6\.3G                 |                 bdd100k \+ private data                 |                  bdd100k \+ private data                  |                           0\.4207                            |                           0\.4200                            |
|  15  | Traffic Detection             | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G\_1\.4              |   caffe    |                     360\*480                      |                11\.6G                 |                      private data                       |                       private data                        |                           0\.5982                            |                           0\.5921                            |
|  16  | ADAS Lane Detection           | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G\_1\.4     |   caffe    |                     480\*640                      |                 2\.5G                 |                      caltech lanes                      |                       caltech lanes                       |                    0\.8864 \(F1\-score\)                     |                    0\.8882 \(F1\-score\)                     |
|  17  | Object Detection              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G\_1\.4           |   caffe    |                     360\*480                      |                6\.57G                 |                      bdd100k train                      |                        bdd100k val                        |                           0\.3052                            |                           0\.2752                            |
|  18  | ADAS Segmentation             | cf\_fpn\_cityscapes\_256\_512\_8\.9G\_1\.4                |   caffe    |                     256\*512                      |                 8\.9G                 |                 Cityscapes gtFineTrain                  |                      Cityscapes Val                       |                           0\.5669                            |                           0\.5607                            |
|  19  | Pose Estimation               | cf\_SPnet\_aichallenger\_224\_128\_0\.54G\_1\.4           |   caffe    |                     224\*128                      |                548\.6M                |                     ai\_challenger                      |                      ai\_challenger                       |                     0\.9000 \(PCKh0\.5\)                     |                     0\.8964 \(PCKh0\.5\)                     |
|  20  | Pose Estimation               | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G\_1\.4 |   caffe    |                     368\*368                      |                189.7G                 |                     ai\_challenger                      |                      ai\_challenger                       |                       0\.4507 \(OKs\)                        |                       0\.4429 \(Oks\)                        |
|  21  | Face Detection                | cf\_densebox\_wider\_320\_320\_0\.49G\_1\.4               |   caffe    |                     320\*320                      |                0\.49G                 |                       wider\_face                       |                           FDDB                            |                           0\.8833                            |                           0\.8783                            |
|  22  | Face Detection                | cf\_densebox\_wider\_360\_640\_1\.11G\_1\.4               |   caffe    |                     360\*640                      |                1\.11G                 |                       wider\_face                       |                           FDDB                            |                           0\.8931                            |                           0\.8922                            |
|  23  | Face Recognition              | cf\_landmark\_celeba\_96\_72\_0\.14G\_1\.4                |   caffe    |                      96\*72                       |                0\.14G                 |                         celebA                          |                      processed helen                      |                      0\.1952 (L2 loss\)                      |                      0\.1971 (L2 loss\)                      |
|  24  | Re\-identification            | cf\_reid\_market1501\_160\_80\_0\.95G\_1\.4               |   caffe    |                      160\*80                      |                0\.95G                 |                   Market1501\+CUHK03                    |                        Market1501                         |                mAP:0\.5660     Rank-1:0\.7800                |                mAP:0\.5590     Rank-1:0\.7760                |
|  25  | Detection+Segmentation        | cf\_multitask\_bdd\_288\_512\_14\.8G\_1\.4                |   caffe    |                     288\*512                      |                14\.8G                 |                   BDD100K+Cityscapes                    |                    BDD100K+Cityscapes                     |                   mAP:0\.2228 mIOU:0\.4088                   |                   mAP:0\.2140 mIOU:0\.4058                   |
|  26  | Object Detection              | dk\_yolov3\_bdd\_288\_512\_53\.7G\_1\.4                   |  darknet   |                     288\*512                      |                53\.7G                 |                         bdd100k                         |                          bdd100k                          |                           0\.5058                            |                           0\.4910                            |
|  27  | Object Detection              | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G\_1\.4      |  darknet   |                     256\*512                      |                5\.46G                 |                       Cityscapes                        |                        Cityscapes                         |                           0\.5520                            |                           0\.5300                            |
|  28  | Object Detection              | dk\_yolov3\_voc\_416\_416\_65\.42G\_1\.4                  |  darknet   |                     416\*416                      |                65\.42G                |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.8240                            |                           0\.8130                            |
|  29  | Object Detection              | dk\_yolov2\_voc\_448\_448\_34G\_1\.4                      |  darknet   |                     448\*448                      |                  34G                  |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7845                            |                           0\.7740                            |
|  30  | Object Detection              | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G\_1\.4           |  darknet   |                     448\*448                      |                11\.56G                |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7700                            |                           0\.7610                            |
|  31  | Object Detection              | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G\_1\.4            |  darknet   |                     448\*448                      |                9\.86G                 |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7670                            |                           0\.7540                            |
|  32  | Object Detection              | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G\_1\.4            |  darknet   |                     448\*448                      |                7\.82G                 |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7576                            |                           0\.7482                            |
|  33  | Face Recognition              | cf_facerec-resnet20_112_96_3.5G_1.4                       |   caffe    |                      112*96                       |                 3.5G                  |                      private data                       |                       private data                        |                        0.9610 (1e-06)                        |                        0.9480 (1e-06)                        |
|  34  | Face Recognition              | cf_facerec-resnet64_112_96_11G_1.4                        |   caffe    |                      112*96                       |                  11G                  |                      private data                       |                       private data                        |                        0.9830 (1e-06)                        |                        0.9820 (1e-06)                        |
|  35  | Medical Segmentation          | cf_FPN-resnet18_EDD_320_320_45.3G_1.4                     |   caffe    |                      320*320                      |                45.30G                 |                         EDD_seg                         |                          EDD_seg                          | mean dice=0.8203                mean jaccard=0.7925                                                                       F2-score=0.8075 | mean dice=0.8055                     mean jaccard=0.7772                                                                              F2- score=0.7925 |
|  36  | Plate Detection               | cf_plate-detection_320_320_0.49G_1.4                      |   caffe    |                      320*320                      |                 0.49G                 |                      private data                       |                       private data                        |                       Recall1: 0.9660                        |                       Recall1: 0.9650                        |
|  37  | Plate Recognition             | cf_plate-recognition_96_288_1.75G_1.4                     |   caffe    |                      96*288                       |                 1.75G                 |                      private data                       |                       private data                        |             plate number:99.51% plate color:100%             |             plate number:99.51% plate color:100%             |
|  38  | Face Detection                | cf_retinaface_wider_360_640_1.11G_1.4                     |   caffe    |                      360*640                      |                 1.11G                 |                    wideface and FDDB                    |                    widerface and FDDB                     |                        0.9140@fp=100                         |                        0.8940@fp=100                         |
|  39  | Face Quality                  | cf_face-quality_80_60_61.68M_1.4                          |   caffe    |                       80*60                       |                61.68M                 |                      private data                       |                       private data                        |                      12.5481 (L1 loss)                       |                      12.6863 (L1 loss)                       |
|  40  | Robot Instrument Segmentation | cf_FPN-resnet18_Endov_240_320_13.75G_1.4                  |   caffe    |                      240*320                      |                13.75G                 |                       EndoVis'15                        |                        EndoVis'15                         |                  Dice=0.8050 Jaccard=0.7270                  |                  Dice=0.8010 Jaccard=0.7230                  |
|  41  | Pose Estimation               | cf_hourglass_mpii_256_256_10.2G_1.4                       |   caffe    |                      256*256                      |                10.20G                 |                     MPII Human Pose                     |                      MPII Human Pose                      |                     0.8718 \(PCKh0\.5\)                      |                     0.8661 \(PCKh0\.5\)                      |
|  42  | 2D Detection                  | dk_tiny-yolov3_416_416_5.46G_1.4                          |  darknet   |                      416*416                      |                 5.46G                 |                      private data                       |                       private data                        |                            0.9739                            |                            0.9650                            |
|  43  | 2D Detection                  | dk_yolov4_coco_416_416_60.1G_1.4                          |  darknet   |                      416*416                      |                 60.1G                 |                          coco                           |                        coco2014-5k                        |                            0.3950                            |                            0.3730                            |
|  44  | 2D Detection                  | dk_yolov4_coco_416_416_0.36_38.2G_1.4                     |  darknet   |                      416*416                      |                 38.2G                 |                          coco                           |                        coco2014-5k                        |                            0.3810                            |                            0.3590                            |
|  45  | Classifiction                 | tf\_inceptionresnetv2\_imagenet\_299\_299\_26\.35G\_1\.4  | tensorflow |                     299\*299                      |                26\.35G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.8037                            |                           0\.7946                            |
|  46  | Classifiction                 | tf\_inceptionv1\_imagenet\_224\_224\_3G\_1\.4             | tensorflow |                     224\*224                      |                  3G                   |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.6976                            |                           0\.6794                            |
|  47  | Classifiction                 | tf\_inceptionv3\_imagenet\_299\_299\_11\.45G\_1\.4        | tensorflow |                     299\*299                      |                11\.45G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7798                            |                           0\.7607                            |
|  48  | Classifiction                 | tf\_inceptionv4\_imagenet\_299\_299\_24\.55G\_1\.4        | tensorflow |                     299\*299                      |                24\.55G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.8018                            |                           0\.7928                            |
|  49  | Classifiction                 | tf\_mobilenetv1\_0\.25\_imagenet\_128\_128\_27M\_1\.4     | tensorflow |                     128\*128                      |                27\.15M                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.4144                            |                           0\.3464                            |
|  50  | Classifiction                 | tf\_mobilenetv1\_0\.5\_imagenet\_160\_160\_150M\_1\.4     | tensorflow |                     160\*160                      |               150\.07M                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.5903                            |                           0\.5195                            |
|  51  | Classifiction                 | tf\_mobilenetv1\_1\.0\_imagenet\_224\_224\_1\.14G\_1\.4   | tensorflow |                     224\*224                      |                1\.14G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7102                            |                           0\.6779                            |
|  52  | Classifiction                 | tf\_mobilenetv2\_1\.0\_imagenet\_224\_224\_602M_1\.4      | tensorflow |                     224\*224                      |                0\.59G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7013                            |                           0\.6767                            |
|  53  | Classifiction                 | tf\_mobilenetv2\_1\.4\_imagenet\_224\_224\_1\.16G\_1\.4   | tensorflow |                     224\*224                      |                1\.16G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7411                            |                           0\.7194                            |
|  54  | Classifiction                 | tf\_resnetv1\_50\_imagenet\_224\_224\_6\.97G\_1\.4        | tensorflow |                     224\*224                      |                6\.97G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7520                            |                           0\.7478                            |
|  55  | Classifiction                 | tf\_resnetv1\_101\_imagenet\_224\_224\_14\.4G\_1\.4       | tensorflow |                     224\*224                      |                14\.4G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7640                            |                           0\.7560                            |
|  56  | Classifiction                 | tf\_resnetv1\_152\_imagenet\_224\_224\_21\.83G\_1\.4      | tensorflow |                     224\*224                      |                21\.83G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7681                            |                           0\.7463                            |
|  57  | Classifiction                 | tf\_vgg16\_imagenet\_224\_224\_30\.96G\_1\.4              | tensorflow |                     224\*224                      |                30\.96G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7089                            |                           0\.7069                            |
|  58  | Classifiction                 | tf\_vgg19\_imagenet\_224\_224\_39\.28G\_1\.4              | tensorflow |                     224\*224                      |                39\.28G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7100                            |                           0\.7026                            |
|  59  | Object Detection              | tf\_ssdmobilenetv1\_coco\_300\_300\_2\.47G\_1\.4          | tensorflow |                     300\*300                      |                2\.47G                 |                        coco2017                         |                     coco2014 minival                      |                           0\.2080                            |                           0\.2100                            |
|  60  | Object Detection              | tf\_ssdmobilenetv2\_coco\_300\_300\_3\.75G\_1\.4          | tensorflow |                     300\*300                      |                3\.75G                 |                        coco2017                         |                     coco2014 minival                      |                           0\.2150                            |                           0\.2110                            |
|  61  | Object Detection              | tf\_ssdresnet50v1\_fpn\_coco\_640\_640\_178\.4G\_1\.4     | tensorflow |                     300\*300                      |                178\.4G                |                        coco2017                         |                     coco2014 minival                      |                           0\.3010                            |                           0\.2900                            |
|  62  | Object Detection              | tf\_yolov3\_voc\_416\_416\_65\.63G\_1\.4                  | tensorflow |                     416\*416                      |                65\.63G                |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7846                            |                           0\.7744                            |
|  63  | Object Detection              | tf\_mlperf_resnet34\_coco\_1200\_1200\_433G\_1\.4         | tensorflow |                    1200\*1200                     |                 433G                  |                        coco2017                         |                         coco2017                          |                           0\.2250                            |                           0\.2150                            |
|  64  | Classifiction                 | tf_inceptionv2_imagenet_224_224_3.88G_1.4                 | tensorflow |                      224*224                      |                 3.88G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7399                            |                            07331                             |
|  65  | Classifiction                 | tf_resnetv2_50_imagenet_299_299_13.1G_1.4                 | tensorflow |                      299*299                      |                 13.1G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7559                            |                            0.7445                            |
|  66  | Classifiction                 | tf_resnetv2_101_imagenet_299_299_26.78G_1.4               | tensorflow |                      299*299                      |                26.78G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7695                            |                            0.7506                            |
|  67  | Classifiction                 | tf_resnetv2_152_imagenet_299_299_40.47G_1.4               | tensorflow |                      299*299                      |                40.47G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7779                            |                            0.7432                            |
|  68  | Object Detection              | tf_ssdlite_mobilenetv2_coco_300_300_1.5G_1.4              | tensorflow |                      300*300                      |                 1.5G                  |                        coco2017                         |                     coco2014 minival                      |                            0.2170                            |                            0.2090                            |
|  69  | Object Detection              | tf_ssdinceptionv2_coco_300_300_9.62G_1.4                  | tensorflow |                      300*300                      |                 9.62G                 |                        coco2017                         |                     coco2014 minival                      |                            0.2390                            |                            0.2360                            |
|  70  | Segmentation                  | tf_mobilenetv2_cityscapes_1024_2048_132.74G_1.4           | tensorflow |                     1024*2048                     |                132.74G                |                       Cityscapes                        |                        Cityscapes                         |                            0.6263                            |                            0.4578                            |
|  71  | Classification                | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G_1.4      | tensorflow |                      224*224                      |                 4.72G                 |                     ImageNet Train                      |                       ImageNet Val                        |                        0.7702/0.9377                         |                        0.7660/0.9337                         |
|  72  | Classification                | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G_1.4      | tensorflow |                      240*240                      |                 7.34G                 |                     ImageNet Train                      |                       ImageNet Val                        |                        0.7862/0.9440                         |                        0.7798/0.9406                         |
|  73  | Classification                | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G_1.4     | tensorflow |                      300*300                      |                19.36G                 |                     ImageNet Train                      |                       ImageNet Val                        |                        0.8026/0.9514                         |                        0.7996/0.9491                         |
|  74  | Classifiction                 | tf_mlperf_resnet50_imagenet_224_224_8.19G_1.4             | tensorflow |                      224*224                      |                 8.19G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7652                            |                            0.7606                            |
|  75  | General Detection             | tf_refinedet_VOC_320_320_81.9G_1.4                        | tensorflow |                      320*320                      |                 81.9G                 |                   voc07\+12\_trainval                   |                        voc07\_test                        |                            0.8015                            |                            0.7999                            |
|  76  | Classifiction                 | tf_mobilenetEdge1.0_imagenet_224_224_990M_1.4             | tensorflow |                      224*224                      |                 990M                  |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7227                            |                            0.6775                            |
|  77  | Classifiction                 | tf_mobilenetEdge0.75_imagenet_224_224_624M_1.4            | tensorflow |                      224*224                      |                 624M                  |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7201                            |                            0.6489                            |
|  78  | Medical Detection             | tf_RefineDet-Medical_EDD_320_320_9.83G_1.4                | tensorflow |                      320*320                      |                 9.83G                 |                           EDD                           |                            EDD                            |                            0.7839                            |                            0.8014                            |
|  79  | Super Resolution              | tf_rcan_DIV2K_360_640_0.98_86.95G_1.4                     | tensorflow |                      360*640                      |                86.95G                 |                          DIV2K                          |                           DIV2K                           |             Set5 Y_PSNR : 37.6402 SSIM : 0.9592              |             Set5 Y_PSNR : 37.2495 SSIM : 0.9556              |
|  80  | Classifiction                 | tf2_resnet50_imagenet_224_224_7.76G_1.4                   | tensorflow |                      224*224                      |                 7.76G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7513                            |                            0.7423                            |
|  81  | Classifiction                 | tf2_mobilenetv1_imagenet_224_224_1.15G_1.4                | tensorflow |                      224*224                      |                 1.15G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7005                            |                            0.5603                            |
|  82  | Classifiction                 | tf2_inceptionv3_imagenet_299_299_11.5G_1.4                | tensorflow |                      299*299                      |                 11.5G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7753                            |                            0.7694                            |
|  83  | Medical Segmentation          | tf2_2d-unet_nuclei_128_128_5.31G_1.4                      | tensorflow |                      128*128                      |                 5.31G                 |                       Nuclei Cell                       |                        Nuclei Cell                        |                            0.3968                            |                            0.3968                            |
|  84  | Semantic Segmentation         | tf2_erfnet_cityscapes_512_1024_54G_1.4                    | tensorflow |                     512*1024                      |                  54G                  |                       Cityscapes                        |                        Cityscapes                         |                            0.5298                            |                            0.5167                            |
|  85  | Classifiction                 | tf2_efficientnet-b0_imagenet_224_224_0.36G_1.4            | tensorflow |                      224*224                      |                 0.36G                 |                     ImageNet Train                      |                       ImageNet Val                        |                        0.7690/0.9320                         |                        0.7515/0.9273                         |
|  86  | Segmentation                  | pt_ENet_cityscapes_512_1024_8.6G_1.4                      |  pytorch   |                     512*1024                      |                 8.6G                  |                       Cityscapes                        |                        Cityscapes                         |                            0.6442                            |                            0.6315                            |
|  87  | Segmentation                  | pt_SemanticFPN-resnet18_cityscapes_256_512_10G_1.4        |  pytorch   |                      256*512                      |                  10G                  |                       Cityscapes                        |                        Cityscapes                         |                            0.6290                            |                            0.6230                            |
|  88  | Face Recognition              | pt_facerec-resnet20_mixed_112_96_3.5G_1.4                 |  pytorch   |                      112*96                       |                 3.5G                  |                          mixed                          |                           mixed                           |                            0.9955                            |                            0.9947                            |
|  89  | Face Quality                  | pt_face-quality_80_60_61.68M_1.4                          |  pytorch   |                       80*60                       |                61.68M                 |                      private data                       |                       private data                        |                            0.1233                            |                            0.1258                            |
|  90  | Multi Task                    | pt_MT-resnet18_mixed_320_512_13.65G_1.4                   |  pytorch   |                      320*512                      |                13.65G                 |                          mixed                          |                           mixed                           |                   mAP:0.3951  mIOU:0.4403                    |                   mAP:0.3841  mIOU:0.4271                    |
|  91  | Face ReID                     | pt_facereid-large_96_96_515M_1.4                          |  pytorch   |                       96*96                       |                 515M                  |                      private data                       |                       private data                        |                   mAP:0.7940  Rank1:0.9550                   |                  mAP:0.7900   Rank1:0.9530                   |
|  92  | Face ReID                     | pt_facereid-small_80_80_90M_1.4                           |  pytorch   |                       80*80                       |                  90M                  |                      private data                       |                       private data                        |                  mAP:0.5600   Rank1:0.8650                   |                  mAP:0.5590   Rank1:0.8650                   |
|  93  | Re\-identification            | pt_personreid-res50_market1501_256_128_5.4G_1.4           |  pytorch   |                      256*128                      |                 4.2G                  |                       market1501                        |                        market1501                         |                  mAP:0.8540   Rank1:0.9410                   |                  mAP:0.8380   Rank1:0.9370                   |
|  94  | Re\-identification            | pt_personreid-res18_market1501_176_80_1.1G_1.4            |  pytorch   |                      176*80                       |                 1.1G                  |                       market1501                        |                        market1501                         |                  mAP:0.7530   Rank1:0.8980                   |                  mAP:0.7460   Rank1:0.8930                   |
|  95  | 3D Detection                  | pt_pointpillars_kitti_12000_100_10.8G_1.4                 |  pytorch   |                   12000\*100\*4                   |                 10.8G                 |                          kitti                          |                           kitti                           |   Car 3D AP@0.5(easy, moderate, hard) 90.79, 89.66, 88.78    |   Car 3D AP@0.5(easy, moderate, hard) 90.75, 87.04, 83.44    |
|  96  | 3D point cloud Segmentation   | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G_1.4         |  pytorch   |                      64*2048                      |                 20.4G                 |                     semantic-kitti                      |                      semantic-kitti                       |                Acc avg 0.8860 IoU avg 0.5100                 |                Acc avg 0.8350 IoU avg 0.4540                 |
|  97  | Covid-19 Segmentation         | pt_FPN-resnet18_covid19-seg_352_352_22.7G_1.4             |  pytorch   |                      352*352                      |                 22.7G                 |                       COVID19-seg                       |                        COVID19-seg                        |       2-classes Dice:0.8588     3-classes mIoU:0.5989        |       2-classes Dice:0.8547     3-classes mIoU:0.5957        |
|  98  | CT lung Segmentation          | pt_unet_chaos-CT_512_512_23.3G_1.4                        |  pytorch   |                      512*512                      |                 23.3G                 |                        Chaos-CT                         |                         Chaos-CT                          |                         Dice:0.9758                          |                         Dice:0.9747                          |
|  99  | surround-view 3D Detection    | pt_pointpillars_nuscenes_40000_64_108G_1.4                |  pytorch   |                   40000\*64\*5                    |                 108G                  |                        NuScenes                         |                         NuScenes                          |                    mAP: 42.2<br>NDS: 55.1                    |                    mAP: 40.5<br>NDS: 53.0                    |
| 100  | 3D Segmentation               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G_1.4        |  pytorch   |                    64\*2048\*5                    |                  32G                  |                     semantic-kitti                      |                      semantic-kitti                       |                         mIou: 54.2%                          |                       mIou: 54.2%(QAT)                       |
| 101  | 4D radar based 3D Detection   | pt_centerpoint_astyx_2560_40_54G_1.4                      |  pytorch   |                    2560\*40\*4                    |                  54G                  |                     Astyx 4D radar                      |                      Astyx 4D radar                       |              BEV AP@0.5: 32.843D AP@0.5: 28.27               |            BEV AP@0.5: 33.823D AP@0.5: 18.54(QAT)            |
| 102  | Image-lidar sensor fusion     | pt_pointpainting_nuscenes_126G_1.4                        |  pytorch   | semanticfpn:320*576<br>pointpillars:40000\*64\*16 | semanticfpn:14G<br/>pointpillars:112G | semanticfpn:NuImages v1.0<br>pointpillars:NuScenes v1.0 | semanticfpn:NuImages v1.0 <br/>pointpillars:NuScenes v1.0 |            mIoU: 69.1%<br>mAP: 51.8<br>NDS: 58.7             |         mIoU: 68.6%<br/>mAP: 50.4<br/>NDS: 56.4(QAT)         |
| 103  | Multi Task                    | pt_multitaskv3_mixed_320_512_25.44G_1.4                   |  pytorch   |                      320*512                      |                25.44G                 |                          mixed                          |                           mixed                           | mAP:51.2<br>mIOU:58.14<br/>Drivable mIOU: 82.57<br/>Lane IOU:43.71<br/>Silog: 8.78 | mAP:50.9<br/>mIOU:57.52<br/>Drivable mIOU: 82.30<br/>Lane IOU:44.01<br/>Silog: 9.32 |
| 104  | Depth Estimation              | pt_fadnet_sceneflow_576_960_359G_1.4                      |  pytorch   |                      576*960                      |                 359G                  |                       Scene Flow                        |                        Scene Flow                         |                          EPE: 0.926                          |                          EPE: 1.169                          |
| 105  | RGB-D Segmentation            | pt_sa-gate_NYUv2_360_360_178G_1.4                         |  pytorch   |                      360*360                      |                 178G                  |                          NYUv2                          |                           NYUv2                           |                         miou: 50.41%                         |                         miou: 49.61%                         |
| 106  | Crowd Counting                | pt_BCC_shanghaitech_800_1000_268.9G_1.4                   |  pytorch   |                     800*1000                      |                268.9G                 |                     shanghaitech A                      |                      shanghaitech A                       |                  MAE: 65.83<br>MSE: 111.75                   |                MAE: 67.60<br>MSE: 117.36(QAT)                |
| 107  | Production Recognition        | pt_pmg_rp2k_224_224_2.28G_1.4                             |  pytorch   |                      224*224                      |                 2.28G                 |                          RP2K                           |                           RP2K                            |                            0.9640                            |                            0.9618                            |
| 108  | Segmentation                  | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G_1.4   |  pytorch   |                     512*1024                      |                 5.4G                  |                       Cityscapes                        |                        Cityscapes                         |                            0.6870                            |                            0.6820                            |

</details>

### Naming Rules
Model name: `F_M_(D)_H_W_(P)_C_V`
* `F` specifies training framework: `cf` is Caffe, `tf` is Tensorflow, `tf2` is Tensorflow 2, `dk` is Darknet, `pt` is PyTorch
* `M` specifies the model
* `D` specifies the dataset. It is optional depending on whether the dataset is public or private
* `H` specifies the height of input data
* `W` specifies the width of input data
* `P` specifies the pruning ratio, it means how much computation is reduced. It is optional depending on whether the model is pruned or not
* `C` specifies the computation of the model: how many Gops per image
* `V` specifies the version of Vitis AI


For example, `cf_refinedet_coco_360_480_0.8_25G_1.4` is a `RefineDet` model trained with `Caffe` using `COCO` dataset, input data size is `360*480`, `80%` pruned, the computation per image is `25Gops` and Vitis AI version is `1.4`.


### caffe-xilinx 
This is a custom distribution of caffe. Please use [caffe-xilinx](/models/AI-Model-Zoo/caffe-xilinx) to test and finetune the caffe models listed in this page.
If you use the Vitis AI docker environment to evaluate and develop the caffe models, you will not need to perform this step, as caffe-xilinx has been pre-compiled in docker.

## Model Download
Please visit [model-list](/models/AI-Model-Zoo/model-list) in this page. You will get downloadlink and MD5 of all the released models, including pre-compiled models running on different platforms.                                 

</details>

### Automated Download Script
With downloader.py, you could quickly find the model you are interested in and specify a version to download it immediately. Please make sure that downloader.py and [model-list](/models/AI-Model-Zoo/model-list) folder are at the same level directory. 

 ```
 python3  downloader.py  
 ```
Step1: You need input framework and model name keyword. Use space divide. 

tf: tensorflow1.x,  tf2: tensorflow2.x,  pt: pytorch,  cf: caffe,  dk: darknet

Step2: Select the specified model based on standard name.

Step3: Select the specified hardware platform for your slected model.

For example, after running downloader.py and input `cf  densebox` then you will see the alternatives such as:

```
0:  cf_densebox_wider_320_320_0.49G_1.4
1:  cf_densebox_wider_360_640_1.11G_1.4 
```
After you input the num: 0, you will see the alternatives such as:

```
0:  cf_densebox_wider_320_320_0.49G_1.4    GPU
1:  densebox_320_320    ZCU102 & ZCU104 & KV260
2:  densebox_320_320    VCK190
3:  densebox_320_320    U50
4:  densebox_320_320    U50lv
5:  densebox_320_320    U50-V3ME
......
```
Then you could choose and input the number, the specified version of model will be automatically downloaded to the current directory.

### Model Directory Structure
Download and extract the model archive to your working area on the local hard disk. For details on the various models, download link and MD5 checksum for the zip file of each model, see [model-list](/models/AI-Model-Zoo/model-list).

#### Caffe Model Directory Structure
For a caffe model, you should see the following directory structure:

    ├── code                            # Contains test , training and quantization scripts.
    │                                     
    │                                   
    ├── readme.md                       # Contains the environment requirements, data preprocess and model information.
    │                                     Refer this to know that how to test and train the model with scripts.
    │                                        
    ├── data                            # Contains the dataset that used for model test and training.
    │                                     When test or training scripts run successfully, dataset will be automatically placed in it.
    │                                     In some cases, you may also need to manually place your own dataset here.
    │                                                      
    ├── quantized  
    │    │    
    │    ├── deploy.caffemodel                # Quantized weights, the output of vai_q_caffe without modification.
    │    ├── deploy.prototxt                  # Quantized prototxt, the output of vai_q_caffe without modification.
    │    ├── quantize_test.prototxt           # Used to run evaluation with quantize_train_test.caffemodel. 
    │    │                                      Some models don't have this file if they are converted from Darknet (Yolov2, Yolov3),   
    │    │                                      Pytorch (ReID) or there is no Caffe Test (Densebox). 
    │    │
    │    ├── quantize_train_test.caffemodel   # Quantized weights can be used for quantized-point training and evaluation.       
    │    └── quantize_train_test.prototxt     # Used for quantized-point training and testing with quantize_train_test.caffemodel 
    │                                           on GPU when datalayer modified to user's data path.         
    │                                                 
    └── float                           
         ├── trainval.caffemodel              # Trained float-point weights.
         ├── test.prototxt                    # Used to run evaluation with python test codes released in near future.    
         └── trainval.prorotxt                # Used for training and testing with caffe train/test command 
                                                when datalayer modified to user's data path.Some models don't
                                                have this file if they are converted from Darknet (Yolov2, Yolov3),
                                                Pytorch (ReID) or there is no Caffe Test (Densebox).              
          

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
    │                                     The accuracy of QAT result is better than direct quantization. 
    │                                     Some models but not all supported QAT, and only these models have qat folder. 
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
        
                                          
                                          
**Note:** For more information on `vai_q_caffe`, `vai_q_tensorflow` and `vai_q_pytorch`, please see the [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_4/ug1414-vitis-ai.pdf).



## Model Performance
All the models in the Model Zoo have been deployed on Xilinx hardware with [Vitis AI](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library](https://github.com/Xilinx/Vitis-AI/tree/master/tools/Vitis-AI-Library). The performance number including end-to-end throughput and latency for each model on various boards with different DPU configurations are listed in the following sections.

For more information about DPU, see [DPU IP Product Guide](https://www.xilinx.com/cgi-bin/docs/ipdoc?c=dpu;v=latest;d=pg338-dpu.pdf).


**Note:** The model performance number listed in the following sections is generated with Vitis AI v1.4 and Vitis AI Lirary v1.4. For different boards, different DPU configurations are used. Vitis AI and Vitis AI Library can be downloaded for free from [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library Github](https://github.com/Xilinx/Vitis-AI/tree/master/tools/Vitis-AI-Library).
We will continue to improve the performance with Vitis AI. The performance number reported here is subject to change in the near future.


### Performance on ZCU102 (0432055-05)
Measured with Vitis AI 1.4 and Vitis AI Library 1.4  

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-05)` board with a `3 * B4096  @ 281MHz` DPU configuration:


| No\. | Model                      | Name                                                | E2E latency \(ms\) <br/>Thread Num =1 | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 13.75                                 | 72.7                                     | 173.6                                   |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 5.46                                  | 183.1                                    | 478                                     |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 5.57                                  | 179.5                                    | 469.8                                   |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 7.6                                   | 131.6                                    | 303.2                                   |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 16.99                                 | 58.8                                     | 135.9                                   |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 34.95                                 | 28.6                                     | 68.5                                    |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 4                                     | 250                                      | 734                                     |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 3.69                                  | 270.8                                    | 1069                                    |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 13.14                                 | 76.1                                     | 278.6                                   |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 117.37                                | 8.5                                      | 24.7                                    |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 30.74                                 | 32.5                                     | 100.1                                   |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 15.9                                  | 62.9                                     | 201.3                                   |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 11.46                                 | 87.2                                     | 288                                     |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 11.46                                 | 87.2                                     | 297.3                                   |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 18.22                                 | 54.9                                     | 200.1                                   |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 10.59                                 | 94.4                                     | 351.3                                   |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 24.99                                 | 40                                       | 117.6                                   |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 29.03                                 | 34.4                                     | 151.9                                   |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 1.94                                  | 515.1                                    | 1588.7                                  |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 265.03                                | 3.8                                      | 15.2                                    |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 2.38                                  | 420.2                                    | 1630                                    |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 4.78                                  | 209                                      | 812.9                                   |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 1.19                                  | 841.6                                    | 1547.6                                  |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 2.85                                  | 350.5                                    | 692.7                                   |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 26.35                                 | 37.9                                     | 127                                     |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 80.56                                 | 12.4                                     | 33                                      |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 11.02                                 | 90.7                                     | 263.4                                   |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 78.5                                  | 12.7                                     | 33.5                                    |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 37.88                                 | 26.4                                     | 70                                      |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 15.9                                  | 62.9                                     | 191.6                                   |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 13.81                                 | 72.4                                     | 224                                     |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 11.83                                 | 84.5                                     | 269.3                                   |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 6.11                                  | 163.5                                    | 329.8                                   |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 14.13                                 | 70.7                                     | 176.8                                   |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 81.65                                 | 12.2                                     | 45.8                                    |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 1.97                                  | 506.2                                    | 2009.4                                  |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 6.66                                  | 150.2                                    | 494.5                                   |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 7.94                                  | 125.9                                    | 542.1                                   |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 0.47                                  | 2118.7                                   | 7244.3                                  |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G                | 27.52                                 | 36.3                                     | 155.9                                   |
| 41   | Hourglass                  | cf_hourglass_mpii_256_256_10.2G                     | 54.76                                 | 18.3                                     | 73.5                                    |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 8.51                                  | 117.5                                    | 393                                     |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 75.45                                 | 13.2                                     | 33.8                                    |
| 44   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 55.4                                  | 18                                       | 45.5                                    |
| 45   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 44.67                                 | 22.4                                     | 50.8                                    |
| 46   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 5.47                                  | 182.8                                    | 468.9                                   |
| 47   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 17.03                                 | 58.7                                     | 134.8                                   |
| 48   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 34.97                                 | 28.6                                     | 68.5                                    |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M         | 0.89                                  | 1122.8                                   | 4147.4                                  |
| 50   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 1.4                                   | 712.3                                    | 2657.2                                  |
| 51   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 3.37                                  | 296.8                                    | 940.9                                   |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 4.08                                  | 244.7                                    | 694.9                                   |
| 53   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 5.57                                  | 179.5                                    | 467.9                                   |
| 54   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 12.56                                 | 79.5                                     | 185.6                                   |
| 55   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 23.48                                 | 42.6                                     | 106.4                                   |
| 56   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 34.29                                 | 29.1                                     | 74.1                                    |
| 57   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 49.75                                 | 20.1                                     | 41.1                                    |
| 58   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 57.68                                 | 17.3                                     | 36.6                                    |
| 59   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 9.39                                  | 106.5                                    | 330.5                                   |
| 60   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 12.67                                 | 78.9                                     | 212.4                                   |
| 61   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 346.97                                | 2.9                                      | 5.1                                     |
| 62   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 75.63                                 | 13.2                                     | 34.5                                    |
| 63   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 559.75                                | 1.8                                      | 7.1                                     |
| 64   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G               | 10.97                                 | 91.1                                     | 230                                     |
| 65   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G               | 28.49                                 | 35.1                                     | 95.5                                    |
| 66   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G             | 48.77                                 | 20.5                                     | 54.4                                    |
| 67   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G             | 69.03                                 | 14.5                                     | 37.5                                    |
| 68   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 9.92                                  | 100.7                                    | 305.2                                   |
| 69   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G                | 28.58                                 | 35                                       | 101.5                                   |
| 70   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G         | 620.95                                | 1.6                                      | 4.6                                     |
| 71   | efficientnet-edgetpu-S     | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G    | 10.54                                 | 94.9                                     | 260.9                                   |
| 72   | efficientnet-edgetpu-M     | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G    | 14.53                                 | 68.8                                     | 181.7                                   |
| 73   | efficientnet-edgetpu-L     | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G   | 35.77                                 | 27.9                                     | 78                                      |
| 74   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 14.04                                 | 71.2                                     | 168.5                                   |
| 75   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 90.56                                 | 11                                       | 34.4                                    |
| 76   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M           | 5.04                                  | 198.2                                    | 547.3                                   |
| 77   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M          | 4.21                                  | 237.5                                    | 706.7                                   |
| 78   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G              | 15.13                                 | 66.1                                     | 230.4                                   |
| 79   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 131.87                                | 7.6                                      | 17.1                                    |
| 80   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 14.15                                 | 70.6                                     | 171.2                                   |
| 81   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 3.42                                  | 292.5                                    | 930.9                                   |
| 82   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 17.15                                 | 58.3                                     | 136.6                                   |
| 83   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 6.46                                  | 154.8                                    | 395.1                                   |
| 84   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 141.25                                | 7.1                                      | 24                                      |
| 85   | efficientnet-b0            | tf2_efficientnet-b0_imagenet_224_224_0.36G          | -                                     | -                                        | -                                       |
| 86   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 108.07                                | 9.3                                      | 36.9                                    |
| 87   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 29.34                                 | 34.1                                     | 161.5                                   |
| 88   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 6.12                                  | 163.3                                    | 329.7                                   |
| 89   | face quality               | pt_face-quality_80_60_61.68M                        | 0.48                                  | 2098                                     | 7196.2                                  |
| 90   | multi_task_v2              | pt_MT-resnet18_mixed_320_512_13.65G                 | 32                                    | 31.2                                     | 101.8                                   |
| 91   | face_reid_large            | pt_facereid-large_96_96_515M                        | 1.19                                  | 841.3                                    | 2205.7                                  |
| 92   | face_reid_small            | pt_facereid-small_80_80_90M                         | 0.54                                  | 1861.8                                   | 5973.2                                  |
| 93   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 10.33                                 | 96.8                                     | 228.2                                   |
| 94   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 2.86                                  | 349.2                                    | 684.1                                   |
| 95   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G               | 50.34                                 | 19.8                                     | 50                                      |
| 96   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 181.45                                | 5.5                                      | 21.1                                    |
| 97   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 27.93                                 | 35.8                                     | 106.1                                   |
| 98   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 45.15                                 | 22.1                                     | 69.7                                    |
| 99   | surround-view pointpillars | pt_pointpillars_nuscenes_40000_64_108G              | 440.5                                 | 2.3                                      | 9.9                                     |
| 100  | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 246.30                                | 4.1                                      | 11.1                                    |
| 101  | centerpoint                | pt_centerpoint_astyx_2560_40_54G                    | 62.45                                 | 16                                       | 48                                      |
| 102  | pointpainting              | pt_pointpainting_nuscenes_126G                      | 779.83                                | 1.3                                      | 4.4                                     |
| 103  | multi_task_v3              | pt_multitaskv3_mixed_320_512_25.44G                 | 60.01                                 | 16.6                                     | 60.8                                    |
| 104  | FADnet                     | pt_fadnet_sceneflow_576_960_359G                    | -                                     | -                                        | -                                       |
| 105  | SA-gate                    | pt_sa-gate_NYUv2_360_360_178G                       | -                                     | -                                        | -                                       |
| 106  | Bayesian Crowd Counting    | pt_BCC_shanghaitech_800_1000_268.9G                 | 304.73                                | 3.3                                      | 10.7                                    |
| 107  | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 6.91                                  | 144.7                                    | 363.4                                   |
| 108  | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 98.54                                 | 10.1                                     | 54                                      |
| -    | Inception_v3               | torchvision_inception_v3                            | 16.99                                 | 58.8                                     | 136.1                                   |
| -    | SqueezeNet                 | torchvision_squeezenet                              | 4.51                                  | 221.6                                    | 816.8                                   |
| -    | resnet50                   | torchvision_resnet50                                | 14.54                                 | 68.7                                     | 166.5                                   |

</details>


### Performance on ZCU104
Measured with Vitis AI 1.4 and Vitis AI Library 1.4 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 300MHz` DPU configuration:


| No\. | Model                      | Name                                                | E2E latency \(ms\) <br/>Thread Num =1 | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 12.89                                 | 77.5                                     | 151.8                                   |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 5.16                                  | 193.8                                    | 416.9                                   |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 5.27                                  | 189.8                                    | 405.7                                   |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 7.16                                  | 139.7                                    | 278.6                                   |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 16.01                                 | 62.4                                     | 122                                     |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 32.81                                 | 30.5                                     | 59                                      |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 3.8                                   | 263                                      | 610.6                                   |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 3.62                                  | 276.2                                    | 978.3                                   |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 12.57                                 | 79.5                                     | 215.6                                   |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 110.18                                | 9.1                                      | 18.4                                    |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 28.99                                 | 34.5                                     | 74.4                                    |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 15.08                                 | 66.3                                     | 155                                     |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 10.9                                  | 91.7                                     | 227                                     |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 10.94                                 | 91.3                                     | 238.4                                   |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 17.38                                 | 57.5                                     | 150.4                                   |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 10.2                                  | 98                                       | 300.7                                   |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 39.67                                 | 25.2                                     | 106.2                                   |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 28.33                                 | 35.3                                     | 146.4                                   |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 1.86                                  | 538                                      | 1330.7                                  |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 252.51                                | 4                                        | 11                                      |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 2.32                                  | 430.1                                    | 1560                                    |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 4.67                                  | 214.3                                    | 754.4                                   |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 1.12                                  | 888.5                                    | 1591.8                                  |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 2.68                                  | 372.7                                    | 705.4                                   |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 25.21                                 | 39.6                                     | 109.1                                   |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 75.66                                 | 13.2                                     | 26.8                                    |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 10.53                                 | 94.9                                     | 233.8                                   |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 73.71                                 | 13.6                                     | 27.3                                    |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 35.64                                 | 28                                       | 57.5                                    |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 15.03                                 | 66.5                                     | 153.3                                   |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 13.09                                 | 76.4                                     | 180.9                                   |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 11.23                                 | 89                                       | 217.3                                   |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 5.74                                  | 174.2                                    | 309.5                                   |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 13.26                                 | 75.4                                     | 145                                     |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 77.75                                 | 12.9                                     | 33.5                                    |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 1.94                                  | 514.5                                    | 1926.1                                  |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 6.13                                  | 162.9                                    | 440                                     |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 7.72                                  | 129.4                                    | 481.2                                   |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 0.46                                  | 2172.8                                   | 6828.2                                  |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G                | 26.62                                 | 37.6                                     | 129.7                                   |
| 41   | Hourglass                  | cf_hourglass_mpii_256_256_10.2G                     | 53.69                                 | 18.6                                     | 73.2                                    |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 8.17                                  | 122.3                                    | 329.5                                   |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 71.04                                 | 14.1                                     | 28.5                                    |
| 44   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 52.12                                 | 19.2                                     | 39                                      |
| 45   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 41.73                                 | 23.9                                     | 45.1                                    |
| 46   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 5.17                                  | 193.3                                    | 412.1                                   |
| 47   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 16.04                                 | 62.3                                     | 121.6                                   |
| 48   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 32.82                                 | 30.4                                     | 59                                      |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M         | 0.87                                  | 1151.3                                   | 4155.9                                  |
| 50   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 1.35                                  | 740.4                                    | 2264.2                                  |
| 51   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 3.21                                  | 311.8                                    | 769.7                                   |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 3.87                                  | 258.3                                    | 593.2                                   |
| 53   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 5.25                                  | 190.3                                    | 406.1                                   |
| 54   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 11.78                                 | 84.8                                     | 165.2                                   |
| 55   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 22                                    | 45.4                                     | 89                                      |
| 56   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 32.13                                 | 31.1                                     | 60.8                                    |
| 57   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 46.68                                 | 21.4                                     | 37.1                                    |
| 58   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 54.07                                 | 18.5                                     | 32.7                                    |
| 59   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 8.92                                  | 112.1                                    | 303.5                                   |
| 60   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 12.04                                 | 83                                       | 194.1                                   |
| 61   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 346.53                                | 2.9                                      | 5.2                                     |
| 62   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 71.02                                 | 14.1                                     | 28.2                                    |
| 63   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 533.66                                | 1.9                                      | 5.2                                     |
| 64   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G               | 10.31                                 | 97                                       | 195.2                                   |
| 65   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G               | 26.86                                 | 37.2                                     | 84.3                                    |
| 66   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G             | 45.78                                 | 21.8                                     | 46.1                                    |
| 67   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G             | 64.68                                 | 15.4                                     | 31.5                                    |
| 68   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 9.45                                  | 105.8                                    | 273.1                                   |
| 69   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G                | 24.72                                 | 40.4                                     | 86.8                                    |
| 70   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G         | 596.67                                | 1.7                                      | 4.7                                     |
| 71   | efficientnet-edgetpu-S     | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G    | 9.9                                   | 101                                      | 208.4                                   |
| 72   | efficientnet-edgetpu-M     | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G    | 13.62                                 | 73.4                                     | 148                                     |
| 73   | efficientnet-edgetpu-L     | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G   | 33.73                                 | 29.6                                     | 63.1                                    |
| 74   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 13.17                                 | 75.9                                     | 148.2                                   |
| 75   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 92.92                                 | 10.8                                     | 25.8                                    |
| 76   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M           | 4.77                                  | 209.5                                    | 463.5                                   |
| 77   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M          | 4                                     | 249.9                                    | 581.4                                   |
| 78   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G              | 14.38                                 | 69.5                                     | 169.8                                   |
| 79   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 124.01                                | 8.1                                      | 15.3                                    |
| 80   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 13.28                                 | 75.3                                     | 148                                     |
| 81   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 3.25                                  | 307.4                                    | 755.9                                   |
| 82   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 16.16                                 | 61.8                                     | 121.8                                   |
| 83   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 6.1                                   | 163.9                                    | 342.5                                   |
| 84   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 136.94                                | 7.3                                      | 25.4                                    |
| 85   | efficientnet-b0            | tf2_efficientnet-b0_imagenet_224_224_0.36G          | -                                     | -                                        | -                                       |
| 86   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 105.53                                | 9.5                                      | 39.1                                    |
| 87   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 28.66                                 | 34.9                                     | 159.3                                   |
| 88   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 5.74                                  | 174.1                                    | 309.4                                   |
| 89   | face quality               | pt_face-quality_80_60_61.68M                        | 0.47                                  | 2144.4                                   | 6877.7                                  |
| 90   | multi_task_v2              | pt_MT-resnet18_mixed_320_512_13.65G                 | 30.67                                 | 32.6                                     | 91.8                                    |
| 91   | face_reid_large            | pt_facereid-large_96_96_515M                        | 1.13                                  | 881.6                                    | 2043                                    |
| 92   | face_reid_small            | pt_facereid-small_80_80_90M                         | 0.52                                  | 1909.7                                   | 5661.7                                  |
| 93   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 9.69                                  | 103.2                                    | 201.7                                   |
| 94   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 2.69                                  | 371.4                                    | 689.9                                   |
| 95   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G               | 48.88                                 | 20.4                                     | 49.5                                    |
| 96   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 179.83                                | 5.6                                      | 21.4                                    |
| 97   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 26.35                                 | 37.9                                     | 79.8                                    |
| 98   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 43.61                                 | 22.9                                     | 59.2                                    |
| 99   | surround-view pointpillars | pt_pointpillars_nuscenes_40000_64_108G              | 429.69                                | 2.3                                      | 9.3                                     |
| 100  | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 238.18                                | 4.2                                      | 11.8                                    |
| 101  | centerpoint                | pt_centerpoint_astyx_2560_40_54G                    | 59.45                                 | 16.8                                     | 20.7                                    |
| 102  | pointpainting              | pt_pointpainting_nuscenes_126G                      | 763.33                                | 1.3                                      | 4.7                                     |
| 103  | multi_task_v3              | pt_multitaskv3_mixed_320_512_25.44G                 | 57.69                                 | 17.3                                     | 54.4                                    |
| 104  | FADnet                     | pt_fadnet_sceneflow_576_960_359G                    | -                                     | -                                        | -                                       |
| 105  | SA-gate                    | pt_sa-gate_NYUv2_360_360_178G                       | -                                     | -                                        | -                                       |
| 106  | Bayesian Crowd Counting    | pt_BCC_shanghaitech_800_1000_268.9G                 | 287.24                                | 3.5                                      | 7.9                                     |
| 107  | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 6.5                                   | 153.7                                    | 313.9                                   |
| 108  | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 96.67                                 | 10.3                                     | 53.5                                    |
| -    | Inception_v3               | torchvision_inception_v3                            | 16.01                                 | 62.4                                     | 122.2                                   |
| -    | SqueezeNet                 | torchvision_squeezenet                              | 4.37                                  | 228.5                                    | 744.7                                   |
| -    | resnet50                   | torchvision_resnet50                                | 13.64                                 | 73.3                                     | 144                                     |

</details>


### Performance on VCK190
Measured with Vitis AI 1.4 and Vitis AI Library 1.4 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Versal` board with 96 AIEs running at 1333Mhz:
  

| No\. | Model                      | Name                                                | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 853.5                                    | 1381.7                                  |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 1286.7                                   | 2955.8                                  |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 983.3                                    | 1750.7                                  |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 785.9                                    | 1245.6                                  |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 424.1                                    | 611.4                                  |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 243.1                                    | 295.4                                  |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 1255.1                                   | 2857.2                                  |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 2431.6                                   | 4173.3                                  |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 317.2                                    | 702.8                                  |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 119.6                                    | 149.9                                  |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 267.1                                    | 465.6                                  |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 339.7                                    | 699.1                                  |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 405.6                                    | 853.8                                  |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 383                                      | 816.1                                  |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 282.9                                    | 665.2                                  |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 269.9                                    | 678.8                                  |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 74.9                                     | 167.8                                  |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 82.9                                     | 190.0                                  |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 1813.2                                   | 4022.4                                  |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 22.1                                     | 42.7                                    |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 1018.4                                   | 2161                                    |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 499.4                                    | 1030.8                                  |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 5249.7                                   | 10082                                  |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 2685.6                                   | 5065.2                                  |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 147.3                                    | 302.3                                  |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 150.2                                    | 200.2                                  |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 406.7                                    | 967.4                                  |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 156                                      | 200.9                                  |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 279                                      | 460.3                                  |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 386.8                                    | 846.6                                  |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 406.6                                    | 943.8                                  |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 425.1                                    | 1048.6                                  |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 1970.9                                   | 2555.1                                  |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 1086.7                                   | 1244.7                                  |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 66.7                                     | 188.8                                  |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 1199.3                                   | 2500.2                                  |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 457.3                                    | 916.1                                  |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 335.1                                    | 750.5                                  |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 7858.9                                   | 18874.4                                |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G                | 102.6                                    | 231.9                                  |
| 41   | Hourglass                  | cf_hourglass_mpii_256_256_10.2G                     | 72.8                                     | 132                                    |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 521.9                                    | 1281.2                                  |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 118.4                                    | 167.8                                  |
| 44   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 129.8                                    | 190.2                                  |
| 45   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 251.1                                    | 307                                    |
| 46   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 988.4                                    | 1774.6                                  |
| 47   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 423.5                                    | 610.7                                  |
| 48   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 243.4                                    | 295.3                                  |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M         | 3614.5                                   | 7974.1                                  |
| 50   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 2594.2                                   | 6211.9                                  |
| 51   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 1383.4                                   | 3577.1                                  |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 1253.5                                   | 2825.8                                  |
| 53   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 1045.4                                   | 1974.9                                  |
| 54   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 904.9                                    | 1519.6                                  |
| 55   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 607.3                                    | 834.5                                  |
| 56   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 456.4                                    | 574.8                                  |
| 57   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 318.6                                    | 371.7                                  |
| 58   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 286.1                                    | 328.2                                  |
| 59   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 362.8                                    | 515.9                                  |
| 60   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 316.9                                    | 508.3                                  |
| 61   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 10.1                                     | 11.8                                    |
| 62   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 156.8                                    | 201.5                                  |
| 63   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 10.6                                     | 23.1                                    |
| 64   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G               | 559.1                                    | 746.8                                  |
| 65   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G               | 400.9                                    | 563.6                                  |
| 66   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G             | 276.5                                    | 344.8                                  |
| 67   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G             | 210.5                                    | 247.9                                  |
| 68   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 333.1                                    | 507.9                                  |
| 69   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G                | 157.6                                    | 386.4                                  |
| 70   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G         | 4.5                                      | 11.6                                    |
| 71   | efficientnet-edgetpu-S     | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G    | 697.3                                    | 1015                                    |
| 72   | efficientnet-edgetpu-M     | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G    | 519.5                                    | 700.4                                  |
| 73   | efficientnet-edgetpu-L     | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G   | 232.4                                    | 279.6                                  |
| 74   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 839.9                                    | 1345.7                                  |
| 75   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 77.5                                     | 186.6                                  |
| 76   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M           | 1225.1                                   | 2721.8                                  |
| 77   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M          | 1302                                     | 3117.1                                  |
| 78   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G              | 416.7                                    | 920                                    |
| 79   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 47.5                                     | 61.6                                    |
| 80   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 854                                      | 1379.9                                  |
| 81   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 1382.6                                   | 3578.7                                  |
| 82   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 440.9                                    | 647.4                                  |
| 83   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 1161.8                                   | 2033.1                                  |
| 84   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 17.6                                     | 45.2                                    |
| 85   | efficientnet-b0            | tf2_efficientnet-b0_imagenet_224_224_0.36G          | 687.5                                    | 995.4                                  |
| 86   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 21.1                                     | 48                                      |
| 87   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 88.6                                     | 191                                    |
| 88   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 1967.1                                   | 2554.8                                  |
| 89   | face quality               | pt_face-quality_80_60_61.68M                        | 7780                                     | 18640.7                                |
| 90   | multi_task_v2              | pt_MT-resnet18_mixed_320_512_13.65G                 | 120.8                                    | 265.1                                  |
| 91   | face_reid_large            | pt_facereid-large_96_96_515M                        | 5040.1                                   | 12355.3                                |
| 92   | face_reid_small            | pt_facereid-small_80_80_90M                         | 7286.5                                   | 16493.1                                |
| 93   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 1133.2                                   | 1825.2                                  |
| 94   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 2750.4                                   | 4891.6                                  |
| 95   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G               | 34.7                                     | 53.9                                    |
| 96   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 11                                       | 22.3                                    |
| 97   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 366.1                                    | 577                                    |
| 98   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 83.3                                     | 220                                    |
| 99   | surround-view pointpillars | pt_pointpillars_nuscenes_40000_64_108G              | 7.4                                      | 17.2                                    |
| 100  | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 9.4                                      | 21.5                                    |
| 101  | centerpoint                | pt_centerpoint_astyx_2560_40_54G                    | 111.2                                    | 225.4                                  |
| 102  | pointpainting              | pt_pointpainting_nuscenes_126G                      | 3.5                                      | 6.5                                    |
| 103  | multi_task_v3              | pt_multitaskv3_mixed_320_512_25.44G                 | 71.3                                     | 183.9                                  |
| 104  | FADnet                     | pt_fadnet_sceneflow_576_960_359G                    | 5.79                                     | 7.19                                    |
| 105  | SA-gate                    | pt_sa-gate_NYUv2_360_360_178G                       | 8.2                                      | 14.1                                    |
| 106  | Bayesian Crowd Counting    | pt_BCC_shanghaitech_800_1000_268.9G                 | 30.6                                     | 55.6                                    |
| 107  | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 1174.1                                   | 2057.9                                  |
| 108  | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 21.8                                     | 48.2                                    |
| -    | Inception_v3               | torchvision_inception_v3                            | 424.1                                    | 611.3                                  |
| -    | SqueezeNet                 | torchvision_squeezenet                              | 807.4                                    | 1971.8                                  |
| -    | resnet50                   | torchvision_resnet50                                | 836.9                                    | 1344.6                                  |

</details>


### Performance on Kria KV260 SOM
Measured with Vitis AI 1.4 and Vitis AI Library 1.4 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Kria KV260` board with a `1 * B4096F  @ 300MHz` DPU configuration:
  

| No\. | Model                      | Name                                                | E2E latency \(ms\) <br/>Thread Num =1 | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 13.68                                 | 73                                       | 77.1                                    |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 5.37                                  | 186.1                                    | 213.8                                   |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 5.46                                  | 183.2                                    | 210.6                                   |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 7.53                                  | 132.8                                    | 147.8                                   |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 16.86                                 | 59.3                                     | 63.7                                    |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 34.32                                 | 29.1                                     | 30.1                                    |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 3.95                                  | 252.9                                    | 310.4                                   |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 3.7                                   | 269.9                                    | 557.7                                   |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 12.76                                 | 78.3                                     | 107                                     |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 111.02                                | 9                                        | 9.2                                     |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 29.44                                 | 34                                       | 37                                      |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 15.58                                 | 64.2                                     | 75.9                                    |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 11.3                                  | 88.5                                     | 111.2                                   |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 11.18                                 | 89.4                                     | 119.2                                   |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 17.65                                 | 56.6                                     | 74.3                                    |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 10.6                                  | 94.4                                     | 149.4                                   |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 40.41                                 | 24.7                                     | 61.7                                    |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 29.41                                 | 34                                       | 76.4                                    |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 1.88                                  | 530.4                                    | 680.3                                   |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 274.99                                | 3.6                                      | 5.5                                     |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 2.37                                  | 421.7                                    | 845.4                                   |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 4.85                                  | 206                                      | 418.5                                   |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 1.25                                  | 801.4                                    | 938.1                                   |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 3.02                                  | 331.2                                    | 379.1                                   |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 26.07                                 | 38.3                                     | 53.7                                    |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 77.06                                 | 13                                       | 13.5                                    |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 11.03                                 | 90.6                                     | 122.8                                   |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 74.85                                 | 13.3                                     | 13.8                                    |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 36.91                                 | 27.1                                     | 29                                      |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 15.15                                 | 66                                       | 76.7                                    |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 13.2                                  | 75.7                                     | 90.4                                    |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 11.36                                 | 88                                       | 108.5                                   |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 6.21                                  | 161.1                                    | 167                                     |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 13.71                                 | 72.9                                     | 74.1                                    |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 78.11                                 | 12.8                                     | 16.8                                    |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 1.96                                  | 510.8                                    | 1059.4                                  |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 6.06                                  | 164.9                                    | 263.9                                   |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 7.85                                  | 127.4                                    | 267                                     |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 0.45                                  | 2231                                     | 3736.3                                  |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G                | 28.8                                  | 34.7                                     | 64.1                                    |
| 41   | Hourglass                  | cf_hourglass_mpii_256_256_10.2G                     | 55.73                                 | 17.9                                     | 52.2                                    |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 8.11                                  | 123.3                                    | 165.7                                   |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 72.21                                 | 13.8                                     | 14.8                                    |
| 44   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 53.7                                  | 18.6                                     | 20.5                                    |
| 45   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 44.17                                 | 22.6                                     | 23.2                                    |
| 46   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 5.38                                  | 185.7                                    | 214.7                                   |
| 47   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 16.91                                 | 59.1                                     | 63.5                                    |
| 48   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 34.35                                 | 29.1                                     | 30.1                                    |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M         | 0.88                                  | 1133.2                                   | 2053.8                                  |
| 50   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 1.38                                  | 725.8                                    | 1127.5                                  |
| 51   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 3.3                                   | 302.9                                    | 387.5                                   |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 4.05                                  | 247                                      | 299.9                                   |
| 53   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 5.52                                  | 181.1                                    | 208.2                                   |
| 54   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 12.61                                 | 79.3                                     | 84.1                                    |
| 55   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 23.13                                 | 43.2                                     | 44.6                                    |
| 56   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 33.56                                 | 29.8                                     | 30.4                                    |
| 57   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 52.04                                 | 19.2                                     | 19.5                                    |
| 58   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 59.56                                 | 16.8                                     | 17                                      |
| 59   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 9.09                                  | 110                                      | 163.2                                   |
| 60   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 12.55                                 | 79.7                                     | 102.3                                   |
| 61   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 361.24                                | 2.8                                      | 5.2                                     |
| 62   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 74.85                                 | 13.3                                     | 13.8                                    |
| 63   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 555.45                                | 1.8                                      | 2.6                                     |
| 64   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G               | 10.66                                 | 93.8                                     | 100.6                                   |
| 65   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G               | 27.89                                 | 35.8                                     | 42.8                                    |
| 66   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G             | 48.29                                 | 20.7                                     | 23.6                                    |
| 67   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G             | 67.53                                 | 14.8                                     | 16.1                                    |
| 68   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 9.77                                  | 102.3                                    | 143.1                                   |
| 69   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G                | 25.4                                  | 39.3                                     | 44.9                                    |
| 70   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G         | 621.96                                | 1.6                                      | 2.7                                     |
| 71   | efficientnet-edgetpu-S     | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G    | 10.32                                 | 96.9                                     | 104.4                                   |
| 72   | efficientnet-edgetpu-M     | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G    | 14.22                                 | 70.3                                     | 74.4                                    |
| 73   | efficientnet-edgetpu-L     | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G   | 35.19                                 | 28.4                                     | 31.2                                    |
| 74   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 13.92                                 | 71.8                                     | 75.8                                    |
| 75   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 94.85                                 | 10.5                                     | 13                                      |
| 76   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M           | 5.00                                  | 199.7                                    | 234                                     |
| 77   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M          | 4.14                                  | 241.3                                    | 291.9                                   |
| 78   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G              | 14.5                                  | 69                                       | 84.3                                    |
| 79   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 132.69                                | 7.5                                      | 7.8                                     |
| 80   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 14.11                                 | 70.8                                     | 74.9                                    |
| 81   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 3.35                                  | 298.6                                    | 380.6                                   |
| 82   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 17.03                                 | 58.7                                     | 63                                      |
| 83   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 6.9                                   | 144.8                                    | 158.3                                   |
| 84   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 145.22                                | 6.9                                      | 12.8                                    |
| 85   | efficientnet-b0            | tf2_efficientnet-b0_imagenet_224_224_0.36G          | -                                     | -                                        | -                                       |
| 86   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 111.6                                 | 9                                        | 21                                      |
| 87   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 29.41                                 | 34                                       | 76.6                                    |
| 88   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 6.21                                  | 161                                      | 167.1                                   |
| 89   | face quality               | pt_face-quality_80_60_61.68M                        | 0.49                                  | 2047.3                                   | 3742.3                                  |
| 90   | multi_task_v2              | pt_MT-resnet18_mixed_320_512_13.65G                 | 33.1                                  | 30.2                                     | 42.8                                    |
| 91   | face_reid_large            | pt_facereid-large_96_96_515M                        | 1.19                                  | 839.6                                    | 1058.4                                  |
| 92   | face_reid_small            | pt_facereid-small_80_80_90M                         | 0.53                                  | 1893.6                                   | 3055.3                                  |
| 93   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 10.42                                 | 96                                       | 102.3                                   |
| 94   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 3.03                                  | 329.9                                    | 368.9                                   |
| 95   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G               | 48.95                                 | 20.4                                     | 28.8                                    |
| 96   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 181.06                                | 5.5                                      | 18.6                                    |
| 97   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 26.63                                 | 37.5                                     | 39.9                                    |
| 98   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 54.58                                 | 18.3                                     | 22.8                                    |
| 99   | surround-view pointpillars | pt_pointpillars_nuscenes_40000_64_108G              | 465.21                                | 2.1                                      | 5                                       |
| 100  | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 246.92                                | 4.0                                      | 10.1                                    |
| 101  | centerpoint                | pt_centerpoint_astyx_2560_40_54G                    | 784.6                                 | 1.3                                      | 4.2                                     |
| 102  | pointpainting              | pt_pointpainting_nuscenes_126G                      | 820.21                                | 1.2                                      | 2.6                                     |
| 103  | multi_task_v3              | pt_multitaskv3_mixed_320_512_25.44G                 | 61.39                                 | 16.3                                     | 25.9                                    |
| 104  | FADnet                     | pt_fadnet_sceneflow_576_960_359G                    | -                                     | -                                        | -                                       |
| 105  | SA-gate                    | pt_sa-gate_NYUv2_360_360_178G                       | -                                     | -                                        | -                                       |
| 106  | Bayesian Crowd Counting    | pt_BCC_shanghaitech_800_1000_268.9G                 | 292.42                                | 3.4                                      | 3.9                                     |
| 107  | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 6.82                                  | 146.5                                    | 160.1                                   |
| 108  | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 100.17                                | 10                                       | 28.1                                    |
| -    | Inception_v3               | torchvision_inception_v3                            | 16.85                                 | 59.3                                     | 63.8                                    |
| -    | SqueezeNet                 | torchvision_squeezenet                              | 4.52                                  | 221.2                                    | 385                                     |
| -    | resnet50                   | torchvision_resnet50                                | 14.47                                 | 69.1                                     | 72.7                                    |

</details>


### Performance on VCK5000
Measured with Vitis AI 1.4 and Vitis AI Library 1.4 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H running at 8PE@350
MHz in Gen3x16:


| No\. | Model                  | Name                                          | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :--------------------- | :-------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50               | cf_resnet50_imagenet_224_224_7.7G             | 350                | 3877.2                                  |
| 2    | resnet18               | cf_resnet18_imagenet_224_224_3.65G            | 350                | 6437                                    |
| 3    | Inception_v1           | cf_inceptionv1_imagenet_224_224_3.16G         | 350                | 4124.6                                  |
| 4    | SqueezeNet             | cf_squeezenet_imagenet_227_227_0.76G          | 350                | 8336.9                                  |
| 5    | refinedet\_baseline    | cf_refinedet_coco_360_480_123G                | 350                | 294.6                                   |
| 6    | refinedet_pruned_0_8   | cf_refinedet_coco_360_480_0.8_25G             | 350                | 659                                     |
| 7    | refinedet_pruned_0_92  | cf_refinedet_coco_360_480_0.92_10.10G         | 350                | 849.6                                   |
| 8    | refinedet_pruned_0_96  | cf_refinedet_coco_360_480_0.96_5.08G          | 350                | 892.8                                   |
| 9    | ssd_adas_pruned_0_95   | cf_ssdadas_bdd_360_480_0.95_6.3G              | 350                | 936.1                                   |
| 10   | ssd_traffic_pruned_0_9 | cf_ssdtraffic_360_480_0.9_11.6G               | 350                | 899.2                                   |
| 11   | VPGnet_pruned_0_99     | cf_VPGnet_caltechlane_480_640_0.99_2.5G       | 350                | 603.4                                   |
| 12   | FPN                    | cf_fpn_cityscapes_256_512_8.9G                | 350                | 1120.1                                  |
| 13   | SP_net                 | cf_SPnet_aichallenger_224_128_0.54G           | 350                | 7724                                    |
| 14   | Openpose_pruned_0_3    | cf_openpose_aichallenger_368_368_0.3_189.7G   | 350                | 169.6                                   |
| 15   | densebox_320_320       | cf_densebox_wider_320_320_0.49G               | 350                | 5930                                    |
| 16   | densebox_640_360       | cf_densebox_wider_360_640_1.11G               | 350                | 2814.9                                  |
| 17   | face_landmark          | cf_landmark_celeba_96_72_0.14G                | 350                | 14209.4                                 |
| 18   | reid                   | cf_reid_market1501_160_80_0.95G               | 350                | 10503.3                                 |
| 19   | multi_task             | cf_multitask_bdd_288_512_14.8G                | 350                | 703.7                                   |
| 20   | yolov3_bdd             | dk_yolov3_bdd_288_512_53.7G                   | 350                | 387.1                                   |
| 21   | yolov3_adas_pruned_0_9 | dk_yolov3_cityscapes_256_512_0.9_5.46G        | 350                | 1392.8                                  |
| 22   | yolov3_voc             | dk_yolov3_voc_416_416_65.42G                  | 350                | 465.3                                   |
| 23   | yolov2_voc             | dk_yolov2_voc_448_448_34G                     | 350                | 953.2                                   |
| 24   | yolov2_voc_pruned_0_66 | dk_yolov2_voc_448_448_0.66_11.56G             | 350                | 1579.1                                  |
| 25   | yolov2_voc_pruned_0_71 | dk_yolov2_voc_448_448_0.71_9.86G              | 350                | 1738                                    |
| 26   | yolov2_voc_pruned_0_77 | dk_yolov2_voc_448_448_0.77_7.82G              | 350                | 1707.5                                  |
| 27   | FPN_Res18_segmentation | cf_FPN-resnet18_EDD_320_320_45.3G             | 350                | 538.8                                   |
| 28   | plate detection        | cf_plate-detection_320_320_0.49G              | 350                | 7735.4                                  |
| 29   | tiny-yolov3            | dk_tiny-yolov3_416_416_5.46G                  | 350                | 2614.9                                  |
| 30   | Inception_v1           | tf_inceptionv1_imagenet_224_224_3G            | 350                | 4391.8                                  |
| 31   | resnet_v1_50           | tf_resnetv1_50_imagenet_224_224_6.97G         | 350                | 4131.8                                  |
| 32   | resnet_v1_101          | tf_resnetv1_101_imagenet_224_224_14.4G        | 350                | 2406.1                                  |
| 33   | resnet_v1_152          | tf_resnetv1_152_imagenet_224_224_21.83G       | 350                | 1704.7                                  |
| 34   | yolov3_voc             | tf_yolov3_voc_416_416_65.63G                  | 350                | 465.3                                   |
| 35   | mlperf_resnet50        | tf_mlperf_resnet50_imagenet_224_224_8.19G     | 350                | 3817.4                                  |
| 36   | refinedet              | tf_refinedet_VOC_320_320_81.9G                | 350                | 381.1                                   |
| 37   | refinedet_medical      | tf_RefineDet-Medical_EDD_320_320_9.83G        | 350                | 1296.7                                  |
| 38   | resnet50               | tf2_resnet50_imagenet_224_224_7.76G           | 350                | 3876.7                                  |
| 39   | 2d-unet                | tf2_2d-unet_nuclei_128_128_5.31G              | 350                | 1962.7                                  |
| 40   | ENet                   | pt_ENet_cityscapes_512_1024_8.6G              | 350                | 141.8                                   |
| 41   | SemanticFPN            | pt_SemanticFPN_cityscapes_256_512_10G         | 350                | 1117.2                                  |
| 42   | face quality           | pt_face-quality_80_60_61.68M                  | 350                | 41925.4                                 |
| 43   | salsanext              | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G | 350                | 181.7                                   |
| 44   | FPN-R18 (light-weight) | pt_FPN-resnet18_covid19-seg_352_352_22.7G     | 350                | 1082.7                                  |
| 45   | 2d-unet                | pt_unet_chaos-CT_512_512_23.3G                | 350                | 230.1                                   |
| -    | SqueezeNet             | torchvision_squeezenet                        | 350                | 5162.7                                  |
| -    | resnet50               | torchvision_resnet50                          | 350                | 3818.3                                  |

</details>


### Performance on U50
Measured with Vitis AI 1.4 and Vitis AI Library 1.4 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for each model on the `Alveo U50` board with 6 DPUCAHX8H kernels running at 300Mhz in Gen3x4:
  

| No\. | Model                      | Name                                           | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :--------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G              | 300                | 573.5                                   |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G             | 300                | 1404                                    |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G          | 300                | 1233.8                                  |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G             | 300                | 973.5                                   |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G          | 300                | 406.8                                   |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G          | 300                | 185.2                                   |
| 7    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G           | 300                | 3612.2                                  |
| 8    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G        | 270                | 569.1                                   |
| 9    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                 | 270                | 49.7                                    |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G              | 270                | 193.2                                   |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G          | 270                | 404.7                                   |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G           | 270                | 589.8                                   |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G               | 300                | 665.3                                   |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                | 300                | 430.5                                   |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G        | 270                | 477.2                                   |
| 16   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                 | 300                | 441.2                                   |
| 17   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G            | 300                | 2885.4                                  |
| 18   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G    | 270                | 29.1                                    |
| 19   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                | 300                | 2223.2                                  |
| 20   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                | 300                | 957.4                                   |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                 | 300                | 10017.8                                 |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G                | 300                | 3869.8                                  |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                 | 300                | 336.1                                   |
| 24   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                    | 270                | 74.9                                    |
| 25   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G         | 270                | 612.8                                   |
| 26   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                   | 270                | 77                                      |
| 27   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                      | 270                | 162.5                                   |
| 28   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G              | 270                | 411.1                                   |
| 29   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G               | 270                | 481.8                                   |
| 30   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G               | 270                | 586.4                                   |
| 31   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                | 300                | 1331                                    |
| 32   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                 | 300                | 491.6                                   |
| 33   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G              | 300                | 101.6                                   |
| 34   | plate detection            | cf_plate-detection_320_320_0.49G               | 300                | 6839.3                                  |
| 35   | plate recognition          | cf_plate-recognition_96_288_1.75G              | 270                | 1126.1                                  |
| 36   | face_quality               | cf_face-quality_80_60_61.68M                   | 270                | 21635.6                                 |
| 37   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                   | 270                | 870.8                                   |
| 38   | yolov4                     | dk_yolov4_coco_416_416_60.1G                   | 270                | 81                                      |
| 39   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G              | 300                | 84.6                                    |
| 40   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G   | 300                | 172.2                                   |
| 41   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G             | 300                | 1254.1                                  |
| 42   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G         | 300                | 406.1                                   |
| 43   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G         | 300                | 185.5                                   |
| 44   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G          | 300                | 643.6                                   |
| 45   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G         | 300                | 334.1                                   |
| 46   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G        | 300                | 222.8                                   |
| 47   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G               | 300                | 161.6                                   |
| 48   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G               | 300                | 134.9                                   |
| 49   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G       | 270                | 31.1                                    |
| 50   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                   | 270                | 77.3                                    |
| 51   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G         | 270                | 13.8                                    |
| 52   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G      | 270                | 500.7                                   |
| 53   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                 | 270                | 70.9                                    |
| 54   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G         | 270                | 420.7                                   |
| 55   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G              | 300                | 49.5                                    |
| 56   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G            | 270                | 517                                     |
| 57   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G         | 270                | 370.9                                   |
| 58   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G               | 270                | 1051.5                                  |
| 59   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G             | 270                | 49.6                                    |
| 60   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G               | 270                | 71.7                                    |
| 61   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G          | 270                | 425.7                                   |
| 62   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G          | 270                | 1213.2                                  |
| 63   | face quality               | pt_face-quality_80_60_61.68M                   | 270                | 20692                                   |
| 64   | face_reid_large            | pt_facereid-large_96_96_515M                   | 270                | 7090.6                                  |
| 65   | face_reid_small            | pt_facereid-small_80_80_90M                    | 270                | 19572.8                                 |
| 66   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G    | 270                | 806.1                                   |
| 67   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G     | 270                | 3390.7                                  |
| 68   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G  | 270                | 130.5                                   |
| 69   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G      | 270                | 207.8                                   |
| 70   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                 | 270                | 60.2                                    |
| 71   | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G | 300                | 25.3                                    |
| 72   | PMG                        | pt_pmg_rp2k_224_224_2.28G                      | 300                | 1089.9                                  |
| -    | Inception_v3               | torchvision_inception_v3                       | 300                | 406.4                                   |
| -    | SqueezeNet                 | torchvision_squeezenet                         | 300                | 2075.8                                  |
| -    | resnet50                   | torchvision_resnet50                           | 300                | 554.7                                   |


The following table lists the performance number including end-to-end throughput for each model on the `Alveo U50` board with 1 DPUCAHX8L kernel running at 333Mhz in Gen3x4:
  

| No\. | Model                      | Name                                                | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 333                | 88.3                                    |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 333                | 323.9                                   |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 333                | 350.9                                   |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 333                | 200.8                                   |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 333                | 109.2                                   |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 333                | 56.1                                    |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 333                | 1155                                    |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 333                | 862.2                                   |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 333                | 118.1                                   |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 333                | 30.1                                    |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 333                | 70.5                                    |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 333                | 82.4                                    |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 333                | 101.3                                   |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 333                | 126.4                                   |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 333                | 120.3                                   |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 333                | 77.9                                    |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 333                | 157                                     |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 333                | 39.8                                    |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 333                | 10954.4                                 |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 333                | 16.7                                    |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 333                | 4896.1                                  |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G                     | 333                | 613.6                                   |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 333                | 21.5                                    |
| 24   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 333                | 318.5                                   |
| 25   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 333                | 141.1                                   |
| 26   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 333                | 14.6                                    |
| 27   | plate detection            | cf_plate-detection_320_320_0.49G                    | 333                | 1196.4                                  |
| 28   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 333                | 205.5                                   |
| 29   | face_quality               | cf_face-quality_80_60_61.68M                        | 333                | 7242.8                                  |
| 30   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 333                | 35.5                                    |
| 31   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 333                | 355.6                                   |
| 32   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 333                | 111.1                                   |
| 33   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 333                | 57.1                                    |
| 34   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 333                | 5425.1                                  |
| 35   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 333                | 2005.7                                  |
| 36   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 333                | 1153.4                                  |
| 37   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 333                | 850.2                                   |
| 38   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 333                | 102.5                                   |
| 39   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 333                | 58.2                                    |
| 40   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 333                | 38.6                                    |
| 41   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 333                | 56.7                                    |
| 42   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 333                | 50.6                                    |
| 43   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 333                | 1004.6                                  |
| 44   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 333                | 236.1                                   |
| 45   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 333                | 598.1                                   |
| 46   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 333                | 6.9                                     |
| 47   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 333                | 88.9                                    |
| 48   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 333                | 45.9                                    |
| 49   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G              | 333                | 158.1                                   |
| 50   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 333                | 4.1                                     |
| 51   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 333                | 88.3                                    |
| 52   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 333                | 1948.4                                  |
| 53   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 333                | 102.1                                   |
| 54   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 333                | 118                                     |
| 55   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 333                | 5.6                                     |
| 56   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 333                | 5.9                                     |
| 57   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 333                | 36.5                                    |
| 58   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 333                | 317.1                                   |
| 59   | face quality               | pt_face-quality_80_60_61.68M                        | 333                | 7245.1                                  |
| 60   | face_reid_small            | pt_facereid-small_80_80_90M                         | 333                | 4663                                    |
| 61   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 333                | 107.2                                   |
| 62   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 333                | 12                                      |
| 63   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 333                | 88.7                                    |
| 64   | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 333                | 11.1                                    |
| -    | Inception_v3               | torchvision_inception_v3                            | 333                | 108.8                                   |
| -    | SqueezeNet                 | torchvision_squeezenet                              | 333                | 509.4                                   |
| -    | resnet50                   | torchvision_resnet50                                | 333                | 91.1                                    |


</details>


### Performance on U50 lv
Measured with Vitis AI 1.4 and Vitis AI Library 1.4 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for each model on the `Alveo U50` board with 10 DPUCAHX8H kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                      | Name                                           | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :--------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G              | 247.5              | 788.4                                   |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G             | 247.5              | 1936.6                                  |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G          | 247.5              | 1702.0                                  |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G             | 247.5              | 1342.1                                  |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G          | 220                | 498.7                                   |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G          | 220                | 226.4                                   |
| 7    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G           | 247.5              | 4976                                    |
| 8    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G        | 247.5              | 863.2                                   |
| 9    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                 | 192.5              | 59.2                                    |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G              | 247.5              | 295.7                                   |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G          | 247.5              | 615.8                                   |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G           | 247.5              | 900.7                                   |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G               | 220                | 826.4                                   |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                | 275                | 653.4                                   |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G        | 247.5              | 719.5                                   |
| 16   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                 | 247.5              | 601.2                                   |
| 17   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G            | 275                | 4291.3                                  |
| 18   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G    | 192.5              | 34.8                                    |
| 19   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                | 275                | 3247.4                                  |
| 20   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                | 275                | 1435.5                                  |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                 | 275                | 14849.1                                 |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G                | 275                | 5833.1                                  |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                 | 247.5              | 426.9                                   |
| 24   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                    | 220                | 101.8                                   |
| 25   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G         | 220                | 835.2                                   |
| 26   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                   | 220                | 104.6                                   |
| 27   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                      | 220                | 220.9                                   |
| 28   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G              | 220                | 558.3                                   |
| 29   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G               | 220                | 654.5                                   |
| 30   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G               | 220                | 795.2                                   |
| 31   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                | 275                | 2026.2                                  |
| 32   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                 | 275                | 749.7                                   |
| 33   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G              | 247.5              | 139.8                                   |
| 34   | plate detection            | cf_plate-detection_320_320_0.49G               | 275                | 6997.9                                  |
| 35   | plate recognition          | cf_plate-recognition_96_288_1.75G              | 247.5              | 1697                                    |
| 36   | face_quality               | cf_face-quality_80_60_61.68M                   | 247.5              | 24506.3                                 |
| 37   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                   | 247.5              | 1325.9                                  |
| 38   | yolov4                     | dk_yolov4_coco_416_416_60.1G                   | 220                | 109.9                                   |
| 39   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G              | 220                | 111.8                                   |
| 40   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G   | 220                | 210.7                                   |
| 41   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G             | 247.5              | 1727.1                                  |
| 42   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G         | 220                | 498                                     |
| 43   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G         | 220                | 227.1                                   |
| 44   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G          | 247.5              | 885.2                                   |
| 45   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G         | 247.5              | 459.6                                   |
| 46   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G        | 247.5              | 306.3                                   |
| 47   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G               | 247.5              | 225.1                                   |
| 48   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G               | 247.5              | 187.6                                   |
| 49   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G       | 220                | 42.3                                    |
| 50   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                   | 220                | 104.9                                   |
| 51   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G         | 247.5              | 21                                      |
| 52   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G      | 247.5              | 763.4                                   |
| 53   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                 | 247.5              | 108.2                                   |
| 54   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G         | 247.5              | 641.8                                   |
| 55   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G              | 220                | 60.3                                    |
| 56   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G            | 247.5              | 788.7                                   |
| 57   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G         | 247.5              | 565.7                                   |
| 58   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G               | 247.5              | 1601.1                                  |
| 59   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G             | 247.5              | 75.5                                    |
| 60   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G               | 247.5              | 108.7                                   |
| 61   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G          | 247.5              | 645.4                                   |
| 62   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G          | 247.5              | 1848.3                                  |
| 63   | face quality               | pt_face-quality_80_60_61.68M                   | 247.5              | 24241.3                                 |
| 64   | face_reid_large            | pt_facereid-large_96_96_515M                   | 247.5              | 10591.7                                 |
| 65   | face_reid_small            | pt_facereid-small_80_80_90M                    | 247.5              | 25945.3                                 |
| 66   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G    | 247.5              | 1229                                    |
| 67   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G     | 247.5              | 5149.3                                  |
| 68   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G  | 247.5              | 132.7                                   |
| 69   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G      | 247.5              | 317.3                                   |
| 70   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                 | 247.5              | 91.2                                    |
| 71   | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G | 220                | 36.4                                    |
| 72   | PMG                        | pt_pmg_rp2k_224_224_2.28G                      | 220                | 1334.6                                  |
| -    | Inception_v3               | torchvision_inception_v3                       | 220                | 498.1                                   |
| -    | SqueezeNet                 | torchvision_squeezenet                         | 247.5              | 2853.8                                  |
| -    | resnet50                   | torchvision_resnet50                           | 247.5              | 763.7                                   |


The following table lists the performance number including end-to-end throughput latency for each model on the `Alveo U50` board with 1 DPUCAHX8L kernel running at 250Mhz in Gen3x4:
  

| No\. | Model                      | Name                                                | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 250                | 80.5                                    |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 250                | 296.9                                   |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 250                | 321.5                                   |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 250                | 184.1                                   |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 250                | 99.6                                    |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 250                | 52.2                                    |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 250                | 1012.4                                  |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 250                | 788.9                                   |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 250                | 110.5                                   |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 250                | 26.6                                    |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 250                | 65.7                                    |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 250                | 77.6                                    |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 250                | 95.6                                    |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 250                | 119.2                                   |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 250                | 111.9                                   |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 250                | 73.9                                    |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 250                | 140.8                                   |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 250                | 38.7                                    |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 250                | 10773.9                                 |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 250                | 14.4                                    |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 250                | 4454.8                                  |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G                     | 250                | 569.3                                   |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 250                | 21.1                                    |
| 24   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 250                | 295.3                                   |
| 25   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 250                | 129.1                                   |
| 26   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 250                | 14.1                                    |
| 27   | plate detection            | cf_plate-detection_320_320_0.49G                    | 250                | 1103.4                                  |
| 28   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 250                | 197                                     |
| 29   | face_quality               | cf_face-quality_80_60_61.68M                        | 250                | 6492                                    |
| 30   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 250                | 32.2                                    |
| 31   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 250                | 326.4                                   |
| 32   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 250                | 99.5                                    |
| 33   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 250                | 52.2                                    |
| 34   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 250                | 4707.8                                  |
| 35   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 250                | 1708.9                                  |
| 36   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 250                | 1006.1                                  |
| 37   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 250                | 742.1                                   |
| 38   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 250                | 93.4                                    |
| 39   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 250                | 51.4                                    |
| 40   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 250                | 35.6                                    |
| 41   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 250                | 52.2                                    |
| 42   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 250                | 46.2                                    |
| 43   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 250                | 853.9                                   |
| 44   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 250                | 209.5                                   |
| 45   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 250                | 527                                     |
| 46   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 250                | 6.2                                     |
| 47   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 250                | 81                                      |
| 48   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 250                | 39.8                                    |
| 49   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G              | 250                | 144.7                                   |
| 50   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 250                | 4                                       |
| 51   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 250                | 80.6                                    |
| 52   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 250                | 1674.4                                  |
| 53   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 250                | 92.9                                    |
| 54   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 250                | 113.9                                   |
| 55   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 250                | 5.4                                     |
| 56   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 250                | 5.7                                     |
| 57   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 250                | 35.7                                    |
| 58   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 250                | 294.4                                   |
| 59   | face quality               | pt_face-quality_80_60_61.68M                        | 250                | 6492.1                                  |
| 60   | face_reid_small            | pt_facereid-small_80_80_90M                         | 250                | 4285.2                                  |
| 61   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 250                | 98.9                                    |
| 62   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 250                | 11.6                                    |
| 63   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 250                | 81.7                                    |
| 64   | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 250                | 10.9                                    |
| -    | Inception_v3               | torchvision_inception_v3                            | 250                | 99.3                                    |
| -    | SqueezeNet                 | torchvision_squeezenet                              | 250                | 471                                     |
| -    | resnet50                   | torchvision_resnet50                                | 250                | 81.4                                    |


</details>


### Performance on U200
Measured with Vitis AI 1.4 and Vitis AI Library 1.4 

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
Measured with Vitis AI 1.4 and Vitis AI Library 1.4 

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

### Performance on U280
Measured with Vitis AI 1.4 and Vitis AI Library 1.4 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for each model on the `Alveo U280` board with 14 DPUCAHX8H kernels running at 300Mhz:
  

| No\. | Model                      | Name                                           | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :--------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G              | 150                | 800.1                                   |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G             | 150                | 1658.3                                  |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G          | 150                | 1378.1                                  |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G             | 150                | 1100.6                                  |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G          | 150                | 443.8                                   |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G          | 150                | 207.9                                   |
| 7    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G           | 150                | 3961                                    |
| 8    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G        | 150                | 575.9                                   |
| 9    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                 | 150                | 60.5                                    |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G              | 150                | 220.3                                   |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G          | 150                | 458.3                                   |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G           | 150                | 647.4                                   |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G               | 150                | 656.4                                   |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                | 150                | 435.2                                   |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G        | 150                | 729.8                                   |
| 16   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                 | 150                | 478.4                                   |
| 17   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G            | 150                | 4071.6                                  |
| 18   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G    | 150                | 37.4                                    |
| 19   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                | 150                | 4845.4                                  |
| 20   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                | 150                | 2135.5                                  |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                 | 150                | 12527.7                                 |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G                | 150                | 4770.5                                  |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                 | 150                | 353.8                                   |
| 24   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                    | 150                | 94                                      |
| 25   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G         | 150                | 817.1                                   |
| 26   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                   | 150                | 97.3                                    |
| 27   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                      | 150                | 202.9                                   |
| 28   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G              | 150                | 499.6                                   |
| 29   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G               | 150                | 582.3                                   |
| 30   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G               | 150                | 694.9                                   |
| 31   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                | 150                | 1599.5                                  |
| 32   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                 | 150                | 579.2                                   |
| 33   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G              | 150                | 116.6                                   |
| 34   | plate detection            | cf_plate-detection_320_320_0.49G               | 150                | 8251.1                                  |
| 35   | plate recognition          | cf_plate-recognition_96_288_1.75G              | 150                | 1497.6                                  |
| 36   | face_quality               | cf_face-quality_80_60_61.68M                   | 150                | 28736.4                                 |
| 37   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                   | 150                | 1040.8                                  |
| 38   | yolov4                     | dk_yolov4_coco_416_416_60.1G                   | 150                | 100.3                                   |
| 39   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G              | 150                | 123.5                                   |
| 40   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G   | 150                | 192.5                                   |
| 41   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G             | 150                | 1430.6                                  |
| 42   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G         | 150                | 444.3                                   |
| 43   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G         | 150                | 208.2                                   |
| 44   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G          | 150                | 750                                     |
| 45   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G         | 150                | 389                                     |
| 46   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G        | 150                | 259.4                                   |
| 47   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G               | 150                | 188.6                                   |
| 48   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G               | 150                | 157.5                                   |
| 49   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G       | 150                | 37.2                                    |
| 50   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                   | 150                | 97.2                                    |
| 51   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G         | 150                | 16.4                                    |
| 52   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G      | 150                | 645.7                                   |
| 53   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                 | 150                | 87.9                                    |
| 54   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G         | 150                | 513.7                                   |
| 55   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G              | 150                | 51.5                                    |
| 56   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G            | 150                | 666.2                                   |
| 57   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G         | 150                | 454.1                                   |
| 58   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G               | 150                | 1287.3                                  |
| 59   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G             | 150                | 67.1                                    |
| 60   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G               | 150                | 123                                     |
| 61   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G          | 150                | 540.2                                   |
| 62   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G          | 150                | 1598.9                                  |
| 63   | face quality               | pt_face-quality_80_60_61.68M                   | 150                | 28616.8                                 |
| 64   | face_reid_large            | pt_facereid-large_96_96_515M                   | 150                | 11573.5                                 |
| 65   | face_reid_small            | pt_facereid-small_80_80_90M                    | 150                | 30606.7                                 |
| 66   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G    | 150                | 1042.6                                  |
| 67   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G     | 150                | 4540.5                                  |
| 68   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G  | 150                | 108.5                                   |
| 69   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G      | 150                | 263.5                                   |
| 70   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                 | 150                | 132.5                                   |
| 71   | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G | 150                | 54.2                                    |
| 72   | PMG                        | pt_pmg_rp2k_224_224_2.28G                      | 150                | 1270.3                                  |
| -    | Inception_v3               | torchvision_inception_v3                       | 150                | 444.2                                   |
| -    | SqueezeNet                 | torchvision_squeezenet                         | 150                | 2187.5                                  |
| -    | resnet50                   | torchvision_resnet50                           | 150                | 774.9                                   |


The following table lists the performance number including end-to-end throughput for each model on the `Alveo U280` board with 2 DPUCAHX8L kernels running at 250Mhz:


| No\.  | Model                      | Name                                                | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ----- | :------------------------- | :-------------------------------------------------- | ------------------ | --------------------------------------- |
| 1     | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 250                | 541.6                                   |
| 2     | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 250                | 155.6                                   |
| 3     | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 250                | 640.3                                   |
| 4     | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 250                | 338.9                                   |
| 5     | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 250                | 188.5                                   |
| 6     | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 250                | 100.4                                   |
| 7     | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 250                | 2042.4                                  |
| 8     | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 250                | 1541.1                                  |
| 9     | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 250                | 191.5                                   |
| 10    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 250                | 51.9                                    |
| 11    | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 250                | 111.9                                   |
| 12    | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 250                | 130.7                                   |
| 13    | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 250                | 156.1                                   |
| 14    | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 250                | 200.3                                   |
| 15    | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 250                | 205.3                                   |
| 16    | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 250                | 114.2                                   |
| 17    | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 250                | 239                                     |
| 18    | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 250                | 49.9                                    |
| 19    | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 250                | 13354.9                                 |
| 20    | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 250                | 28                                      |
| 21    | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 250                | 9276.1                                  |
| 22    | reid                       | cf_reid_market1501_160_80_0.95G                     | 250                | 922                                     |
| 23    | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 250                | 24.2                                    |
| 24    | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 250                | 477.3                                   |
| 25    | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 250                | 228.3                                   |
| 26    | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 250                | 16.3                                    |
| 27    | plate detection            | cf_plate-detection_320_320_0.49G                    | 250                | 2269.6                                  |
| 28    | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 250                | 277.1                                   |
| 29    | face_quality               | cf_face-quality_80_60_61.68M                        | 250                | 13653.1                                 |
| 30    | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 250                | 63.8                                    |
| 31    | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 250                | 650.6                                   |
| 32    | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 250                | 188.9                                   |
| 33    | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 250                | 102.8                                   |
| 34    | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 250                | 9803.7                                  |
| 35    | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 250                | 3485.4                                  |
| 36    | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 250                | 2038.4                                  |
| 37    | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 250                | 1489.3                                  |
| 38    | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 250                | 178.6                                   |
| 39    | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 250                | 102                                     |
| 40    | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 250                | 67.6                                    |
| 41    | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 250                | 106.2                                   |
| 42    | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 250                | 93.7                                    |
| 43    | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 250                | 1619.5                                  |
| 44    | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 250                | 374                                     |
| 45    | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 250                | 1017                                    |
| 46    | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 250                | 12.4                                    |
| 47    | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 250                | 156.6                                   |
| 48    | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 250                | 79.2                                    |
| 49    | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G              | 250                | 244                                     |
| 50    | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 250                | 7.7                                     |
| 51    | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 250                | 155.5                                   |
| 52    | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 250                | 3421.8                                  |
| 53    | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 250                | 178.6                                   |
| 54    | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 250                | 136.3                                   |
| 55    | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 250                | 7.7                                     |
| 56    | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 250                | 8.1                                     |
| 57    | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 250                | 43.6                                    |
| 58    | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 250                | 475.7                                   |
| 59    | face quality               | pt_face-quality_80_60_61.68M                        | 250                | 13680.8                                 |
| 60    | face_reid_small            | pt_facereid-small_80_80_90M                         | 250                | 8872.3                                  |
| 61    | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 250                | 186.2                                   |
| 62    | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 250                | 19.3                                    |
| 63    | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 250                | 127.7                                   |
| 64    | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 250                | 15.6                                    |
| -     | Inception_v3               | torchvision_inception_v3                            | 250                | 187.6                                   |
| -     | SqueezeNet                 | torchvision_squeezenet                              | 250                | 1538.8                                  |
| -     | resnet50                   | torchvision_resnet50                                | 250                | 162.5                                   |


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
