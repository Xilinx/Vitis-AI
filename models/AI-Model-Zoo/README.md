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

## Vitis AI 2.0 Model Zoo New Features！
1.New 22 models and total 130 models with diverse deep learning frameworks (Caffe, TensorFlow, TensorFlow 2 and PyTorch) in Vitis AI 2.0.</br>

2.The diversity of AI Model Zoo is significantly improved compared to Vitis AI 1.4.</br>

a.For AD/ADAS, provide Lidar based 3D detection, Instance segmentation, upgraded lane detection and Image-lidar fusion 3D detection, etc.</br>

b.For Medical, provide Image Denoising, upgraded Image Super-resolution, Spectral remove, Polyp segmentation, etc.</br>

c.For Smart city and Industrial Vision, provide Binocular depth estimation, Person orientation estimation, Joint detection and Tracking, etc.</br>

d.In addition, begin to provide NLP reference models and OFA search reference models such as OFA-resnet50 and OFA-depthwise-resnet50.</br>

3.EoU enhancement: Provide upgraded automatic script for users to find their required reference models and download them faster and more efficiently.</br>

## Model Details
The following table includes comprehensive information about all models, including application, framework, training and validation dataset, input size, computation as well as float and quantized precision.

<details>
 <summary><b>Click here to view details</b></summary>

| No\. | Application                     | Name                                                    | Framework  |                          Input Size                          |             OPS per image             |                      Training Set                       |                          Val Set                          |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |
| :--: | :------------------------------ | :------------------------------------------------------ | :--------: | :----------------------------------------------------------: | :-----------------------------------: | :-----------------------------------------------------: | :-------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  1   | Image Classification            | cf\_resnet50\_imagenet\_224\_224\_7\.7G\_2.0            |   caffe    |                           224\*224                           |                 7\.7G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7444/0\.9185                        |                       0\.7335/0\.9130                        |
|  2   | Image Classifiction             | cf\_resnet18\_imagenet\_224\_224\_3\.65G\_2.0           |   caffe    |                           224\*224                           |                3\.65G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.6844/0\.8864                        |                        0.6699/0.8826                         |
|  3   | Image Classification            | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G\_2.0        |   caffe    |                           224\*224                           |                3\.16G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7030/0\.8971                        |                       0\.6964/0\.8942                        |
|  4   | Image Classification            | cf\_inceptionv2\_imagenet\_224\_224\_4G\_2.0            |   caffe    |                           224\*224                           |                  4G                   |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7275/0\.9111                        |                       0\.7166/0\.9029                        |
|  5   | Image Classification            | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G\_2.0        |   caffe    |                           299*299                            |                11\.4G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7701/0\.9329                        |                       0\.7622/0\.9302                        |
|  6   | Image Classification            | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G\_2.0        |   caffe    |                           299\*299                           |                24\.5G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.7958/0\.9470                        |                       0\.7899/0\.9444                        |
|  7   | Image Classification            | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G\_2.0        |   caffe    |                           224\*224                           |                 608M                  |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.6527/0\.8643                        |                       0\.6399/0\.8540                        |
|  8   | Image Classifiction             | cf_squeezenet_imagenet_227_227_0.76G_2.0                |   caffe    |                           227\*227                           |                0\.76G                 |                     ImageNet Train                      |                       ImageNet Val                        |                       0\.5432/0\.7824                        |                       0\.5270/0\.7816                        |
|  9   | ADAS Pedstrain Detection        | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G\_2.0    |   caffe    |                           360\*640                           |                 5\.9G                 |         coco2014\_train\_person and crowndhuman         |                   coco2014\_val\_person                   |                           0\.5903                            |                           0\.5857                            |
|  10  | Object Detection                | cf_refinedet_coco_360_480_123G_2.0                      |   caffe    |                           360\*480                           |                 123G                  |                 coco2014\_train\_person                 |                   coco2014\_val\_person                   |                            0.6928                            |                            0.7042                            |
|  11  | Object Detection                | cf\_refinedet\_coco\_360\_480\_0\.8\_25G_2.0            |   caffe    |                           360\*480                           |                  25G                  |                 coco2014\_train\_person                 |                   coco2014\_val\_person                   |                           0\.6794                            |                           0\.6790                            |
|  12  | Object Detection                | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G_2.0       |   caffe    |                           360\*480                           |                10\.10G                |                 coco2014\_train\_person                 |                   coco2014\_val\_person                   |                           0\.6489                            |                           0\.6487                            |
|  13  | Object Detection                | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G_2.0        |   caffe    |                           360\*480                           |                5\.08G                 |                 coco2014\_train\_person                 |                   coco2014\_val\_person                   |                           0\.6120                            |                           0\.6112                            |
|  14  | ADAS Vehicle Detection          | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G_2.0            |   caffe    |                           360\*480                           |                 6\.3G                 |                 bdd100k \+ private data                 |                  bdd100k \+ private data                  |                           0\.4207                            |                           0\.4200                            |
|  15  | Traffic Detection               | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G_2.0              |   caffe    |                           360\*480                           |                11\.6G                 |                      private data                       |                       private data                        |                           0\.5982                            |                           0\.5921                            |
|  16  | ADAS Lane Detection             | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G_2.0     |   caffe    |                           480\*640                           |                 2\.5G                 |                      caltech lanes                      |                       caltech lanes                       |                    0\.8864 \(F1\-score\)                     |                    0\.8882 \(F1\-score\)                     |
|  17  | Object Detection                | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G_2.0           |   caffe    |                           360\*480                           |                6\.57G                 |                      bdd100k train                      |                        bdd100k val                        |                           0\.3052                            |                           0\.2752                            |
|  18  | ADAS Segmentation               | cf\_fpn\_cityscapes\_256\_512\_8\.9G_2.0                |   caffe    |                           256\*512                           |                 8\.9G                 |                 Cityscapes gtFineTrain                  |                      Cityscapes Val                       |                           0\.5669                            |                           0\.5607                            |
|  19  | Pose Estimation                 | cf\_SPnet\_aichallenger\_224\_128\_0\.54G_2.0           |   caffe    |                           224\*128                           |                548\.6M                |                     ai\_challenger                      |                      ai\_challenger                       |                     0\.9000 \(PCKh0\.5\)                     |                     0\.8964 \(PCKh0\.5\)                     |
|  20  | Pose Estimation                 | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G_2.0 |   caffe    |                           368\*368                           |                189.7G                 |                     ai\_challenger                      |                      ai\_challenger                       |                       0\.4507 \(OKs\)                        |                       0\.4429 \(Oks\)                        |
|  21  | Face Detection                  | cf\_densebox\_wider\_320\_320\_0\.49G_2.0               |   caffe    |                           320\*320                           |                0\.49G                 |                       wider\_face                       |                           FDDB                            |                           0\.8833                            |                           0\.8783                            |
|  22  | Face Detection                  | cf\_densebox\_wider\_360\_640\_1\.11G_2.0               |   caffe    |                           360\*640                           |                1\.11G                 |                       wider\_face                       |                           FDDB                            |                           0\.8931                            |                           0\.8922                            |
|  23  | Face Recognition                | cf\_landmark\_celeba\_96\_72\_0\.14G_2.0                |   caffe    |                            96\*72                            |                0\.14G                 |                         celebA                          |                      processed helen                      |                      0\.1952 (L2 loss\)                      |                      0\.1971 (L2 loss\)                      |
|  24  | Re\-identification              | cf\_reid\_market1501\_160\_80\_0\.95G_2.0               |   caffe    |                           160\*80                            |                0\.95G                 |                   Market1501\+CUHK03                    |                        Market1501                         |                mAP:0\.5660     Rank-1:0\.7800                |                mAP:0\.5590     Rank-1:0\.7760                |
|  25  | Detection+Segmentation          | cf\_multitask\_bdd\_288\_512\_14\.8G_2.0                |   caffe    |                           288\*512                           |                14\.8G                 |                   BDD100K+Cityscapes                    |                    BDD100K+Cityscapes                     |                   mAP:0\.2228 mIOU:0\.4088                   |                   mAP:0\.2140 mIOU:0\.4058                   |
|  26  | Object Detection                | dk\_yolov3\_bdd\_288\_512\_53\.7G_2.0                   |  darknet   |                           288\*512                           |                53\.7G                 |                         bdd100k                         |                          bdd100k                          |                           0\.5058                            |                           0\.4910                            |
|  27  | Object Detection                | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G_2.0      |  darknet   |                           256\*512                           |                5\.46G                 |                       Cityscapes                        |                        Cityscapes                         |                           0\.5520                            |                           0\.5300                            |
|  28  | Object Detection                | dk\_yolov3\_voc\_416\_416\_65\.42G_2.0                  |  darknet   |                           416\*416                           |                65\.42G                |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.8240                            |                           0\.8130                            |
|  29  | Object Detection                | dk\_yolov2\_voc\_448\_448\_34G_2.0                      |  darknet   |                           448\*448                           |                  34G                  |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7845                            |                           0\.7740                            |
|  30  | Object Detection                | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G\_2.0          |  darknet   |                           448\*448                           |                11\.56G                |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7700                            |                           0\.7610                            |
|  31  | Object Detection                | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G_2.0            |  darknet   |                           448\*448                           |                9\.86G                 |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7670                            |                           0\.7540                            |
|  32  | Object Detection                | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G_2.0            |  darknet   |                           448\*448                           |                7\.82G                 |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7576                            |                           0\.7482                            |
|  33  | Face Recognition                | cf_facerec-resnet20_112_96_3.5G_2.0                     |   caffe    |                            112*96                            |                 3.5G                  |                      private data                       |                       private data                        |                        0.9610 (1e-06)                        |                        0.9480 (1e-06)                        |
|  34  | Face Recognition                | cf_facerec-resnet64_112_96_11G_2.0                      |   caffe    |                            112*96                            |                  11G                  |                      private data                       |                       private data                        |                        0.9830 (1e-06)                        |                        0.9820 (1e-06)                        |
|  35  | Medical Segmentation            | cf_FPN-resnet18_EDD_320_320_45.3G_2.0                   |   caffe    |                           320*320                            |                45.30G                 |                         EDD_seg                         |                          EDD_seg                          | mean dice=0.8203<br/>mean jaccard=0.7925<br>F2-score=0.8075  | mean dice=0.8055<br/>mean jaccard=0.7772<br/>F2- score=0.7925 |
|  36  | Plate Detection                 | cf_plate-detection_320_320_0.49G_2.0                    |   caffe    |                           320*320                            |                 0.49G                 |                      private data                       |                       private data                        |                       Recall1: 0.9660                        |                       Recall1: 0.9650                        |
|  37  | Plate Recognition               | cf_plate-recognition_96_288_1.75G_2.0                   |   caffe    |                            96*288                            |                 1.75G                 |                      private data                       |                       private data                        |             plate number:99.51% plate color:100%             |             plate number:99.51% plate color:100%             |
|  38  | Face Detection                  | cf_retinaface_wider_360_640_1.11G_2.0                   |   caffe    |                           360*640                            |                 1.11G                 |                    wideface and FDDB                    |                    widerface and FDDB                     |                        0.9140@fp=100                         |                        0.8940@fp=100                         |
|  39  | Face Quality                    | cf_face-quality_80_60_61.68M_2.0                        |   caffe    |                            80*60                             |                61.68M                 |                      private data                       |                       private data                        |                      12.5481 (L1 loss)                       |                      12.6863 (L1 loss)                       |
|  40  | Robot Instrument Segmentation   | cf_FPN-resnet18_Endov_240_320_13.75G_2.0                |   caffe    |                           240*320                            |                13.75G                 |                       EndoVis'15                        |                        EndoVis'15                         |                  Dice=0.8050 Jaccard=0.7270                  |                  Dice=0.8010 Jaccard=0.7230                  |
|  41  | Pose Estimation                 | cf_hourglass_mpii_256_256_10.2G_2.0                     |   caffe    |                           256*256                            |                10.20G                 |                     MPII Human Pose                     |                      MPII Human Pose                      |                     0.8718 \(PCKh0\.5\)                      |                     0.8661 \(PCKh0\.5\)                      |
|  42  | Commodity Detection             | dk_tiny-yolov3_416_416_5.46G_2.0                        |  darknet   |                           416*416                            |                 5.46G                 |                      private data                       |                       private data                        |                            0.9739                            |                            0.9650                            |
|  43  | Object Detection                | dk_yolov4_coco_416_416_60.1G_2.0                        |  darknet   |                           416*416                            |                 60.1G                 |                          coco                           |                        coco2014-5k                        |                            0.3950                            |                            0.3730                            |
|  44  | Object Detection                | dk_yolov4_coco_416_416_0.36_38.2G_2.0                   |  darknet   |                           416*416                            |                 38.2G                 |                          coco                           |                        coco2014-5k                        |                            0.3810                            |                            0.3590                            |
|  45  | Classifiction                   | tf\_inceptionresnetv2\_imagenet\_299\_299\_26\.35G_2.0  | tensorflow |                           299\*299                           |                26\.35G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.8037                            |                           0\.7946                            |
|  46  | Classifiction                   | tf\_inceptionv1\_imagenet\_224\_224\_3G_2.0             | tensorflow |                           224\*224                           |                  3G                   |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.6976                            |                           0\.6794                            |
|  47  | Classifiction                   | tf\_inceptionv3\_imagenet\_299\_299\_11\.45G_2.0        | tensorflow |                           299\*299                           |                11\.45G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7798                            |                           0\.7607                            |
|  48  | Classifiction                   | tf\_inceptionv4\_imagenet\_299\_299\_24\.55G_2.0        | tensorflow |                           299\*299                           |                24\.55G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.8018                            |                           0\.7928                            |
|  49  | Classifiction                   | tf\_mobilenetv1\_0\.25\_imagenet\_128\_128\_27M_2.0     | tensorflow |                           128\*128                           |                27\.15M                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.4144                            |                           0\.3464                            |
|  50  | Classifiction                   | tf\_mobilenetv1\_0\.5\_imagenet\_160\_160\_150M\_2.0    | tensorflow |                           160\*160                           |               150\.07M                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.5903                            |                           0\.5195                            |
|  51  | Classifiction                   | tf\_mobilenetv1\_1\.0\_imagenet\_224\_224\_1\.14G_2.0   | tensorflow |                           224\*224                           |                1\.14G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7102                            |                           0\.6779                            |
|  52  | Classifiction                   | tf\_mobilenetv2\_1\.0\_imagenet\_224\_224\_602M_2.0     | tensorflow |                           224\*224                           |                0\.59G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7013                            |                           0\.6767                            |
|  53  | Classifiction                   | tf\_mobilenetv2\_1\.4\_imagenet\_224\_224\_1\.16G_2.0   | tensorflow |                           224\*224                           |                1\.16G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7411                            |                           0\.7194                            |
|  54  | Classifiction                   | tf\_resnetv1\_50\_imagenet\_224\_224\_6\.97G_2.0        | tensorflow |                           224\*224                           |                6\.97G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7520                            |                           0\.7478                            |
|  55  | Classifiction                   | tf\_resnetv1\_101\_imagenet\_224\_224\_14\.4G_2.0       | tensorflow |                           224\*224                           |                14\.4G                 |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7640                            |                           0\.7560                            |
|  56  | Classifiction                   | tf\_resnetv1\_152\_imagenet\_224\_224\_21\.83G_2.0      | tensorflow |                           224\*224                           |                21\.83G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7681                            |                           0\.7463                            |
|  57  | Classifiction                   | tf\_vgg16\_imagenet\_224\_224\_30\.96G_2.0              | tensorflow |                           224\*224                           |                30\.96G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7089                            |                           0\.7069                            |
|  58  | Classifiction                   | tf\_vgg19\_imagenet\_224\_224\_39\.28G_2.0              | tensorflow |                           224\*224                           |                39\.28G                |                     ImageNet Train                      |                       ImageNet Val                        |                           0\.7100                            |                           0\.7026                            |
|  59  | Object Detection                | tf\_ssdmobilenetv1\_coco\_300\_300\_2\.47G_2.0          | tensorflow |                           300\*300                           |                2\.47G                 |                        coco2017                         |                     coco2014 minival                      |                           0\.2080                            |                           0\.2100                            |
|  60  | Object Detection                | tf\_ssdmobilenetv2\_coco\_300\_300\_3\.75G_2.0          | tensorflow |                           300\*300                           |                3\.75G                 |                        coco2017                         |                     coco2014 minival                      |                           0\.2150                            |                           0\.2110                            |
|  61  | Object Detection                | tf\_ssdresnet50v1\_fpn\_coco\_640\_640\_178\.4G_2.0     | tensorflow |                           640\*640                           |                178\.4G                |                        coco2017                         |                     coco2014 minival                      |                           0\.3010                            |                           0\.2900                            |
|  62  | Object Detection                | tf\_yolov3\_voc\_416\_416\_65\.63G_2.0                  | tensorflow |                           416\*416                           |                65\.63G                |                   voc07\+12\_trainval                   |                        voc07\_test                        |                           0\.7846                            |                           0\.7744                            |
|  63  | Object Detection                | tf\_mlperf_resnet34\_coco\_1200\_1200\_433G_2.0         | tensorflow |                          1200\*1200                          |                 433G                  |                        coco2017                         |                         coco2017                          |                           0\.2250                            |                           0\.2150                            |
|  64  | Classifiction                   | tf_inceptionv2_imagenet_224_224_3.88G_2.0               | tensorflow |                           224*224                            |                 3.88G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7399                            |                            07331                             |
|  65  | Classifiction                   | tf_resnetv2_50_imagenet_299_299_13.1G_2.0               | tensorflow |                           299*299                            |                 13.1G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7559                            |                            0.7445                            |
|  66  | Classifiction                   | tf_resnetv2_101_imagenet_299_299_26.78G_2.0             | tensorflow |                           299*299                            |                26.78G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7695                            |                            0.7506                            |
|  67  | Classifiction                   | tf_resnetv2_152_imagenet_299_299_40.47G_2.0             | tensorflow |                           299*299                            |                40.47G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7779                            |                            0.7432                            |
|  68  | Object Detection                | tf_ssdlite_mobilenetv2_coco_300_300_1.5G_2.0            | tensorflow |                           300*300                            |                 1.5G                  |                        coco2017                         |                     coco2014 minival                      |                            0.2170                            |                            0.2090                            |
|  69  | Object Detection                | tf_ssdinceptionv2_coco_300_300_9.62G_2.0                | tensorflow |                           300*300                            |                 9.62G                 |                        coco2017                         |                     coco2014 minival                      |                            0.2390                            |                            0.2360                            |
|  70  | Segmentation                    | tf_mobilenetv2_cityscapes_1024_2048_132.74G_2.0         | tensorflow |                          1024*2048                           |                132.74G                |                       Cityscapes                        |                        Cityscapes                         |                            0.6263                            |                            0.4578                            |
|  71  | Classification                  | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G_2.0    | tensorflow |                           224*224                            |                 4.72G                 |                     ImageNet Train                      |                       ImageNet Val                        |                        0.7702/0.9377                         |                        0.7660/0.9337                         |
|  72  | Classification                  | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G_2.0    | tensorflow |                           240*240                            |                 7.34G                 |                     ImageNet Train                      |                       ImageNet Val                        |                        0.7862/0.9440                         |                        0.7798/0.9406                         |
|  73  | Classification                  | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G_2.0   | tensorflow |                           300*300                            |                19.36G                 |                     ImageNet Train                      |                       ImageNet Val                        |                        0.8026/0.9514                         |                        0.7996/0.9491                         |
|  74  | Classifiction                   | tf_mlperf_resnet50_imagenet_224_224_8.19G_2.0           | tensorflow |                           224*224                            |                 8.19G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7652                            |                            0.7606                            |
|  75  | General Detection               | tf_refinedet_VOC_320_320_81.9G_2.0                      | tensorflow |                           320*320                            |                 81.9G                 |                   voc07\+12\_trainval                   |                        voc07\_test                        |                            0.8015                            |                            0.7999                            |
|  76  | Classifiction                   | tf_mobilenetEdge1.0_imagenet_224_224_990M_2.0           | tensorflow |                           224*224                            |                 990M                  |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7227                            |                            0.6775                            |
|  77  | Classifiction                   | tf_mobilenetEdge0.75_imagenet_224_224_624M_2.0          | tensorflow |                           224*224                            |                 624M                  |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7201                            |                            0.6489                            |
|  78  | Medical Detection               | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G_2.0         | tensorflow |                           320*320                            |                 9.83G                 |                           EDD                           |                            EDD                            |                            0.7839                            |                            0.8014                            |
|  79  | Super Resolution                | tf_rcan_DIV2K_360_640_0.98_86.95G_2.0                   | tensorflow |                           360*640                            |                86.95G                 |                          DIV2K                          |                           DIV2K                           |             Set5 Y_PSNR : 37.6402 SSIM : 0.9592              |             Set5 Y_PSNR : 37.2495 SSIM : 0.9556              |
|  80  | Classifiction                   | tf2_resnet50_imagenet_224_224_7.76G_2.0                 | tensorflow |                           224*224                            |                 7.76G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7513                            |                            0.7423                            |
|  81  | Classifiction                   | tf2_mobilenetv1_imagenet_224_224_1.15G_2.0              | tensorflow |                           224*224                            |                 1.15G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7005                            |                            0.5603                            |
|  82  | Classifiction                   | tf2_inceptionv3_imagenet_299_299_11.5G_2.0              | tensorflow |                           299*299                            |                 11.5G                 |                     ImageNet Train                      |                       ImageNet Val                        |                            0.7753                            |                            0.7694                            |
|  83  | Medical Segmentation            | tf2_2d-unet_nuclei_128_128_5.31G_2.0                    | tensorflow |                           128*128                            |                 5.31G                 |                       Nuclei Cell                       |                        Nuclei Cell                        |                            0.3968                            |                            0.3968                            |
|  84  | Semantic Segmentation           | tf2_erfnet_cityscapes_512_1024_54G_2.0                  | tensorflow |                           512*1024                           |                  54G                  |                       Cityscapes                        |                        Cityscapes                         |                            0.5298                            |                            0.5167                            |
|  85  | Classifiction                   | tf2_efficientnet-b0_imagenet_224_224_0.36G_2.0          | tensorflow |                           224*224                            |                 0.36G                 |                     ImageNet Train                      |                       ImageNet Val                        |                        0.7690/0.9320                         |                        0.7515/0.9273                         |
|  86  | Sentiment detection             | tf2_sentiment-detection_IMDB_500_32_53.3M_2.0           | tensorflow |                           500*32                             |                 53.3M                 |                          IMDB                           |                           IMDB                            |                            0.8708                            |                            0.8695                            |
|  87  | Customer satisfaction           | tf2_customer-satisfaction_Cars4U_25_32_2.7M_2.0         | tensorflow |                            25*32                             |                 2.7M                  |                         Cars 4U                         |                          Cars 4U                          |                            0.9565                            |                            0.9565                            |
|  88  | Cassification                   | tf2_mobilenetv3_imagenet_224_224_132M_2.0               | tensorflow |                           224*224                            |                 132M                  |                     ImageNet Train                      |                       ImageNet Val                        |                        0.6756/0.8728                         |                        0.6536/0.8544                         |
|  89  | Segmentation                    | pt_ENet_cityscapes_512_1024_8.6G_2.0                    |  pytorch   |                           512*1024                           |                 8.6G                  |                       Cityscapes                        |                        Cityscapes                         |                            0.6442                            |                            0.6315                            |
|  90  | Segmentation                    | pt_SemanticFPN-resnet18_cityscapes_256_512_10G_2.0      |  pytorch   |                           256*512                            |                  10G                  |                       Cityscapes                        |                        Cityscapes                         |                            0.6290                            |                            0.6230                            |
|  91  | Face Recognition                | pt_facerec-resnet20_mixed_112_96_3.5G_2.0               |  pytorch   |                            112*96                            |                 3.5G                  |                          mixed                          |                           mixed                           |                            0.9955                            |                            0.9947                            |
|  92  | Face Quality                    | pt_face-quality_80_60_61.68M_2.0                        |  pytorch   |                            80*60                             |                61.68M                 |                      private data                       |                       private data                        |                            0.1233                            |                            0.1258                            |
|  93  | Multi Task                      | pt_MT-resnet18_mixed_320_512_13.65G_2.0                 |  pytorch   |                           320*512                            |                13.65G                 |                          mixed                          |                           mixed                           |                   mAP:0.3951  mIOU:0.4403                    |                   mAP:0.3841  mIOU:0.4271                    |
|  94  | Face ReID                       | pt_facereid-large_96_96_515M_2.0                        |  pytorch   |                            96*96                             |                 515M                  |                      private data                       |                       private data                        |                   mAP:0.7940  Rank1:0.9550                   |                  mAP:0.7900   Rank1:0.9530                   |
|  95  | Face ReID                       | pt_facereid-small_80_80_90M_2.0                         |  pytorch   |                            80*80                             |                  90M                  |                      private data                       |                       private data                        |                  mAP:0.5600   Rank1:0.8650                   |                  mAP:0.5590   Rank1:0.8650                   |
|  96  | Re\-identification              | pt_personreid-res50_market1501_256_128_5.4G_2.0         |  pytorch   |                           256*128                            |                 4.2G                  |                       market1501                        |                        market1501                         |                  mAP:0.8540   Rank1:0.9410                   |                  mAP:0.8380   Rank1:0.9370                   |
|  97  | Re\-identification              | pt_personreid-res18_market1501_176_80_1.1G_2.0          |  pytorch   |                            176*80                            |                 1.1G                  |                       market1501                        |                        market1501                         |                  mAP:0.7530   Rank1:0.8980                   |                  mAP:0.7460   Rank1:0.8930                   |
|  98  | 3D Detection                    | pt_pointpillars_kitti_12000_100_10.8G_2.0               |  pytorch   |                        12000\*100\*4                         |                 10.8G                 |                          kitti                          |                           kitti                           |   Car 3D AP@0.5(easy, moderate, hard) 90.79, 89.66, 88.78    |   Car 3D AP@0.5(easy, moderate, hard) 90.75, 87.04, 83.44    |
|  99  | 3D point cloud Segmentation     | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G_2.0       |  pytorch   |                           64*2048                            |                 20.4G                 |                     semantic-kitti                      |                      semantic-kitti                       |                Acc avg 0.8860 IoU avg 0.5100                 |                Acc avg 0.8350 IoU avg 0.4540                 |
| 100  | Covid-19 Segmentation           | pt_FPN-resnet18_covid19-seg_352_352_22.7G_2.0           |  pytorch   |                           352*352                            |                 22.7G                 |                       COVID19-seg                       |                        COVID19-seg                        |       2-classes Dice:0.8588     3-classes mIoU:0.5989        |       2-classes Dice:0.8547     3-classes mIoU:0.5957        |
| 101  | CT lung Segmentation            | pt_unet_chaos-CT_512_512_23.3G_2.0                      |  pytorch   |                           512*512                            |                 23.3G                 |                        Chaos-CT                         |                         Chaos-CT                          |                         Dice:0.9758                          |                         Dice:0.9747                          |
| 102  | Surround-view 3D Detection      | pt_pointpillars_nuscenes_40000_64_108G_2.0              |  pytorch   |                         40000\*64\*5                         |                 108G                  |                        NuScenes                         |                         NuScenes                          |                    mAP: 42.2<br>NDS: 55.1                    |                    mAP: 40.5<br>NDS: 53.0                    |
| 103  | 3D Segmentation                 | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G_2.0      |  pytorch   |                         64\*2048\*5                          |                  32G                  |                     semantic-kitti                      |                      semantic-kitti                       |                         mIou: 54.2%                          |                       mIou: 54.2%(QAT)                       |
| 104  | 4D radar based 3D Detection     | pt_centerpoint_astyx_2560_40_54G_2.0                    |  pytorch   |                         2560\*40\*4                          |                  54G                  |                     Astyx 4D radar                      |                      Astyx 4D radar                       |              BEV AP@0.5: 32.843D AP@0.5: 28.27               |            BEV AP@0.5: 33.823D AP@0.5: 18.54(QAT)            |
| 105  | Image-lidar sensor fusion       | pt_pointpainting_nuscenes_126G_2.0                      |  pytorch   |      semanticfpn:320*576<br>pointpillars:40000\*64\*16       | semanticfpn:14G<br/>pointpillars:112G | semanticfpn:NuImages v1.0<br>pointpillars:NuScenes v1.0 | semanticfpn:NuImages v1.0 <br/>pointpillars:NuScenes v1.0 |            mIoU: 69.1%<br>mAP: 51.8<br>NDS: 58.7             |         mIoU: 68.6%<br/>mAP: 50.4<br/>NDS: 56.4(QAT)         |
| 106  | Multi Task                      | pt_multitaskv3_mixed_320_512_25.44G_2.0                 |  pytorch   |                           320*512                            |                25.44G                 |                          mixed                          |                           mixed                           | mAP:51.2<br>mIOU:58.14<br/>Drivable mIOU: 82.57<br/>Lane IOU:43.71<br/>Silog: 8.78 | mAP:50.9<br/>mIOU:57.52<br/>Drivable mIOU: 82.30<br/>Lane IOU:44.01<br/>Silog: 9.32 |
| 107  | Depth Estimation                | pt_fadnet_sceneflow_576_960_441G_2.0                    |  pytorch   |                           576*960                            |                 359G                  |                       Scene Flow                        |                        Scene Flow                         |                          EPE: 0.926                          |                          EPE: 1.169                          |
| 108  | RGB-D Segmentation              | pt_sa-gate_NYUv2_360_360_59.71G_2.0                     |  pytorch   |                           360*360                            |                59.71G                 |                          NYUv2                          |                           NYUv2                           |                         miou: 47.58%                         |                         miou: 46.80%                         |
| 109  | Crowd Counting                  | pt_BCC_shanghaitech_800_1000_268.9G_2.0                 |  pytorch   |                           800*1000                           |                268.9G                 |                     shanghaitech A                      |                      shanghaitech A                       |                  MAE: 65.83<br>MSE: 111.75                   |             MAE: 67.60<br>MSE: 117.36<br/>(QAT)              |
| 110  | Production Recognition          | pt_pmg_rp2k_224_224_2.28G_2.0                           |  pytorch   |                           224*224                            |                 2.28G                 |                          RP2K                           |                           RP2K                            |                            0.9640                            |                            0.9618                            |
| 111  | Segmentation                    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G_2.0 |  pytorch   |                           512*1024                           |                 5.4G                  |                       Cityscapes                        |                        Cityscapes                         |                            0.6870                            |                            0.6820                            |
| 112  | Lane detection                  | pt_ultrafast_CULane_288_800_8.4G_2.0                    |  pytorch   |                           288*800                            |                 8.4G                  |                         CULane                          |                          CULane                           |                            0.6988                            |                            0.6922                            |
| 113  | Image-lidar fusion 3D detection | pt_CLOCs_kitti_2.0                                      |  pytorch   | 2d detection (YOLOX): 384*1248<br/>3d detection (PointPillars): 12000\*100\*4<br/>fusionnet: 800\*1000\*4 |                  41G                  |                          kitti                          |                           kitti                           | 2d detection: Mod Car bbox AP@0.70: 89.40<br/>3d detection:   Mod Car bev@0.7 :85.50<br/>Mod Car 3d@0.7 :70.01<br/>Mod Car bev@0.5 :89.69<br/>Mod Car 3d@0.5 :89.48<br/>fusionnet: Mod Car bev@0.7 :87.58<br/>Mod Car 3d@0.7 :73.04<br/>Mod Car bev@0.5 :93.98<br/>Mod Car 3d@0.5 :93.56 | 2d detection: Mod Car bbox AP@0.70: 89.50<br/>3d detection:   Mod Car bev@0.7 :85.50<br/>Mod Car 3d@0.7 :70.01<br/>Mod Car bev@0.5 :89.69<br/>Mod Car 3d@0.5 :89.48<br/>fusionnet: Mod Car bev@0.7 :87.58<br/>Mod Car 3d@0.7 :73.04<br/>Mod Car bev@0.5 :93.98<br/>Mod Car 3d@0.5 :93.56<br/>(QAT) |
| 114  | Instance segmentation           | pt_SOLO_coco_640_640_107G_2.0                           |  pytorch   |                           640*640                            |                 107G                  |                          coco                           |                           coco                            |                            0.242                             |                            0.212                             |
| 115  | Traffic sign detection          | pt_yolox_TT100K_640_640_73G_2.0                         |  pytorch   |                           640*640                            |                  73G                  |                         TT100K                          |                          TT100K                           |                            0.623                             |                            0.621                             |
| 116  | Image Super-resolution          | pt_SESR-S_DIV2K_360_640_7.48G_2.0                       |  pytorch   |                           360*640                            |                 7.48G                 |                          DIV2K                          |                           DIV2K                           | (Set5) PSNR/SSIM= 37.309/0.958<br/>(Set14) PSNR/SSIM= 32.894/ 0.911<br/>(B100)  PSNR/SSIM= 31.663/ 0.893<br/>(Urban100)  PSNR/SSIM = 30.276/0.908<br/> | (Set5) PSNR/SSIM= 36.813/0.954<br/>(Set14) PSNR/SSIM= 32.607/ 0.906<br/>(B100)  PSNR/SSIM= 31.443/ 0.889<br/>(Urban100)  PSNR/SSIM = 29.901/0.899<br/> |
| 117  | Image Denoising                 | pt_DRUNet_Kvasir_528_608_0.4G_2.0                       |  pytorch   |                           528*608                            |                 0.4G                  |                         Kvasir                          |                          Kvasir                           |                         PSNR = 34.57                         |                      PSNR = 34.06(QAT)                       |
| 118  | Spectral remove                 | pt_SSR_CVC_256_256_39.72G_2.0                           |  pytorch   |                           256*256                            |                39.72G                 |                           CVC                           |                            CVC                            |                              -                               |                              -                               |
| 119  | Coverage prediction             | pt_C2D2lite_CC20_512_512_6.86G_2.0                      |  pytorch   |                           512*512                            |                 6.86G                 |                          CC20                           |                           CC20                            |                          MAE=7.566%                          |                         MAE=10.399%                          |
| 120  | Polyp segmentation              | pt_HardNet_mixed_352_352_22.78G_2.0                     |  pytorch   |                           352*352                            |                22.78G                 |                          mixed                          |                           mixed                           |                         mDice=0.9142                         |                         mDice=0.9136                         |
| 121  | Binocular depth estimation      | pt_fadnet_sceneflow_576_960_0.65_154G_2.0               |  pytorch   |                           576*960                            |                 154G                  |                        sceneflow                        |                         sceneflow                         |                          EPE: 0.823                          |                          EPE: 1.158                          |
| 122  | Binocular depth estimation      | pt_psmnet_sceneflow_576_960_0.68_696G_2.0               |  pytorch   |                           576*960                            |                 696G                  |                        sceneflow                        |                         sceneflow                         |                          EPE: 0.961                          |                          EPE: 1.022                          |
| 123  | Person orientation estimation   | pt_person-orientation_224_112_558M_2.0                  |  pytorch   |                           224*112                            |                 558M                  |                         private                         |                          private                          |                            0.930                             |                            0.929                             |
| 124  | Joint detection and Tracking    | pt_FairMOT_mixed_640_480_0.5_36G_2.0                    |  pytorch   |                           640*480                            |                  36G                  |                          mixed                          |                           mixed                           |                  MOTA 59.1%<br/>IDF1 62.5%                   |                  MOTA 58.1%<br/>IDF1 60.5%                   |
| 125  | NLP                             | pt_open-information-extraction_qasrl_100_200_1.5G_2.0   |  pytorch   |                           100*200                            |                 1.5G                  |                         QA-SRL                          |                          QA-SRL                           |               Acc/F1-score:<br/>58.70%/77.12%                |               Acc/F1-score:<br/>58.70%/77.19%                |
| 126  | OFA                             | pt_OFA-resnet50_imagenet_160_160_900M_2.0               |  pytorch   |                           160*160                            |                 900M                  |                     ImageNet Train                      |                       ImageNet Val                        |                         0.758/0.926                          |                         0.748/0.921                          |
| 127  | OFA                             | pt_OFA-depthwise-res50_imagenet_176_176_1.25G_2.0       |  pytorch   |                           176*176                            |                 1.25G                 |                     ImageNet Train                      |                       ImageNet Val                        |                        0.7633/0.9292                         |                        0.7629/0.9306                         |
| 128  | torchvison                      | pt_inceptionv3_imagenet_299_299_5.7G_2.0                |  pytorch   |                           299*299                            |                 5.7G                  |                     ImageNet Train                      |                       ImageNet Val                        |                         0.758/0.927                          |                         0.757/0.927                          |
| 129  | torchvision                     | pt_squeezenet_imagenet_224_224_351.7M_2.0               |  pytorch   |                           224*224                            |                351.7M                 |                     ImageNet Train                      |                       ImageNet Val                        |                         0.582/0.806                          |                         0.582/0.806                          |
| 130  | torchvision                     | pt_resnet50_imagenet_224_224_4.1G_2.0                   |  pytorch   |                           224*224                            |                 4.1G                  |                     ImageNet Train                      |                       ImageNet Val                        |                         0.761/0.929                          |                         0.760/0.928                          |

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


For example, `cf_refinedet_coco_360_480_0.8_25G_2.0` is a `RefineDet` model trained with `Caffe` using `COCO` dataset, input data size is `360*480`, `80%` pruned, the computation per image is `25Gops` and Vitis AI version is `2.0`.


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
Step1: You need input framework and model name keyword. Use space divide. If input `all` you will get list of all models.

tf: tensorflow1.x,  tf2: tensorflow2.x,  pt: pytorch,  cf: caffe,  dk: darknet, all: list all model

Step2: Select the specified model based on standard name.

Step3: Select the specified hardware platform for your slected model.

For example, after running downloader.py and input `cf  densebox` then you will see the alternatives such as:

```
0:  all
1:  cf_densebox_wider_320_320_0.49G_2.0
2:  cf_densebox_wider_360_640_1.11G_2.0 
```
After you input the num: 1, you will see the alternatives such as:

```
0:  all
1:  cf_densebox_wider_320_320_0.49G_2.0    GPU
2:  densebox_320_320    ZCU102 & ZCU104 & KV260
3:  densebox_320_320    VCK190
4:  densebox_320_320    U50
5:  densebox_320_320    U50lv
6:  densebox_320_320    U50-V3ME
......
```
Then you could choose and input the number, the specified version of model will be automatically downloaded to the current directory.

In addition, if you need download all models on all platforms at once, you just need enter number in the order indicated by the tips of Step 1/2/3 (select all -> 0 -> 0).

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
         └── trainval.prototxt                # Used for training and testing with caffe train/test command 
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
        
                                          
                                          
**Note:** For more information on `vai_q_caffe`, `vai_q_tensorflow` and `vai_q_pytorch`, please see the [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/2_0/ug1414-vitis-ai.pdf).



## Model Performance
All the models in the Model Zoo have been deployed on Xilinx hardware with [Vitis AI](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library](https://github.com/Xilinx/Vitis-AI/tree/master/tools/Vitis-AI-Library). The performance number including end-to-end throughput and latency for each model on various boards with different DPU configurations are listed in the following sections.

For more information about DPU, see [DPU IP Product Guide](https://www.xilinx.com/cgi-bin/docs/ipdoc?c=dpu;v=latest;d=pg338-dpu.pdf).

For RNN models such as NLP models, please refer to [DPU-for-RNN](https://github.com/Xilinx/Vitis-AI/blob/master/demo/DPU-for-RNN) for dpu specification information. 

**Note:** The model performance number listed in the following sections is generated with Vitis AI v2.0 and Vitis AI Lirary v2.0. For different platforms, the different DPU configurations are used. Vitis AI and Vitis AI Library can be downloaded for free from [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library Github](https://github.com/Xilinx/Vitis-AI/tree/master/tools/Vitis-AI-Library).
We will continue to improve the performance with Vitis AI. The performance number reported here is subject to change in the near future.


### Performance on ZCU102 (0432055-05)
Measured with Vitis AI 2.0 and Vitis AI Library 2.0  

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-05)` board with a `3 * B4096  @ 281MHz` DPU configuration:


| No\. | Model                      | Name                                                | E2E latency \(ms\) <br/>Thread Num =1 | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 12.44                                 | 80.3                                     | 188.2                                   |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 5.41                                  | 184.9                                    | 477.3                                   |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 5.50                                  | 181.8                                    | 470.2                                   |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 7.55                                  | 132.4                                    | 300.7                                   |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 16.92                                 | 59.1                                     | 135.9                                   |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 34.89                                 | 28.6                                     | 68.5                                    |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 3.93                                  | 254.3                                    | 733.3                                   |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 3.65                                  | 273.9                                    | 1078.9                                  |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 13.10                                 | 76.2                                     | 278.3                                   |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 117.33                                | 8.5                                      | 24.7                                    |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 31.03                                 | 32.2                                     | 97.9                                    |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 16.03                                 | 62.4                                     | 197.3                                   |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 11.62                                 | 86.0                                     | 278.2                                   |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 11.46                                 | 87.2                                     | 296.8                                   |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 18.19                                 | 54.9                                     | 200.1                                   |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 10.54                                 | 94.8                                     | 351.7                                   |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 25.64                                 | 39.0                                     | 117.4                                   |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 28.95                                 | 34.5                                     | 151.6                                   |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 1.79                                  | 557.0                                    | 1626.5                                  |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 263.50                                | 3.8                                      | 15.2                                    |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 2.31                                  | 432.2                                    | 1654.7                                  |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 4.72                                  | 211.7                                    | 818.9                                   |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 1.12                                  | 890.2                                    | 1552.9                                  |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 2.79                                  | 357.9                                    | 692.9                                   |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 26.29                                 | 38.0                                     | 127.0                                   |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 80.47                                 | 12.4                                     | 32.8                                    |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 10.94                                 | 91.4                                     | 263.6                                   |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 79.18                                 | 12.6                                     | 33.1                                    |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 38.14                                 | 26.2                                     | 69.3                                    |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 15.83                                 | 63.1                                     | 191.8                                   |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 13.77                                 | 72.6                                     | 224.1                                   |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 11.78                                 | 84.9                                     | 268.9                                   |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 6.06                                  | 164.9                                    | 334.0                                   |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 14.08                                 | 71.0                                     | 177.7                                   |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 79.72                                 | 12.5                                     | 47.1                                    |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 1.92                                  | 520.4                                    | 2080.5                                  |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 5.22                                  | 191.4                                    | 558.4                                   |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 7.92                                  | 126.2                                    | 541.9                                   |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 0.39                                  | 2584.7                                   | 8015.8                                  |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G                | 27.43                                 | 36.4                                     | 156.5                                   |
| 41   | Hourglass                  | cf_hourglass_mpii_256_256_10.2G                     | 54.79                                 | 18.2                                     | 73.8                                    |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 8.45                                  | 118.3                                    | 393.2                                   |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 75.63                                 | 13.2                                     | 33.6                                    |
| 44   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 55.49                                 | 18.0                                     | 45.3                                    |
| 45   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 44.61                                 | 22.4                                     | 50.8                                    |
| 46   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 5.41                                  | 184.7                                    | 467.7                                   |
| 47   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 16.95                                 | 59.0                                     | 135.1                                   |
| 48   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 34.87                                 | 28.7                                     | 68.6                                    |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M         | 0.82                                  | 1216.3                                   | 4500.6                                  |
| 50   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 1.33                                  | 749.9                                    | 2778.7                                  |
| 51   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 3.27                                  | 305.8                                    | 951.3                                   |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 4.03                                  | 248.1                                    | 687.8                                   |
| 53   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 5.50                                  | 181.6                                    | 467.7                                   |
| 54   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 12.49                                 | 80.0                                     | 185.6                                   |
| 55   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 23.41                                 | 42.7                                     | 106.4                                   |
| 56   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 34.22                                 | 29.2                                     | 74.1                                    |
| 57   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 49.71                                 | 20.1                                     | 41.1                                    |
| 58   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 57.65                                 | 17.3                                     | 36.5                                    |
| 59   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 9.25                                  | 108.0                                    | 337.0                                   |
| 60   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 12.59                                 | 79.4                                     | 212.2                                   |
| 61   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 344.14                                | 2.9                                      | 5.2                                     |
| 62   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 75.83                                 | 13.2                                     | 34.3                                    |
| 63   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 558.99                                | 1.8                                      | 7.0                                     |
| 64   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G               | 10.90                                 | 91.7                                     | 230.0                                   |
| 65   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G               | 28.15                                 | 35.5                                     | 95.7                                    |
| 66   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G             | 48.45                                 | 20.6                                     | 54.4                                    |
| 67   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G             | 68.70                                 | 14.5                                     | 37.5                                    |
| 68   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 10.08                                 | 99.1                                     | 307.8                                   |
| 69   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G                | 25.53                                 | 39.1                                     | 102.3                                   |
| 70   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G         | 583.35                                | 1.7                                      | 5.4                                     |
| 71   | efficientnet-edgetpu-S     | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G    | 8.90                                  | 112.3                                    | 308.2                                   |
| 72   | efficientnet-edgetpu-M     | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G    | 12.79                                 | 78.2                                     | 205.1                                   |
| 73   | efficientnet-edgetpu-L     | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G   | 31.86                                 | 31.4                                     | 87.8                                    |
| 74   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 13.97                                 | 71.6                                     | 168.5                                   |
| 75   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 88.77                                 | 11.3                                     | 34.4                                    |
| 76   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M           | 4.98                                  | 200.8                                    | 546.9                                   |
| 77   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M          | 4.14                                  | 241.2                                    | 706.4                                   |
| 78   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G         | 15.08                                 | 66.3                                     | 230.4                                   |
| 79   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 131.79                                | 7.6                                      | 17.1                                    |
| 80   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 12.69                                 | 78.8                                     | 186.2                                   |
| 81   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 3.32                                  | 301.2                                    | 940.6                                   |
| 82   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 17.07                                 | 58.6                                     | 136.7                                   |
| 83   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 6.42                                  | 155.8                                    | 394.8                                   |
| 84   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 141.07                                | 7.1                                      | 23.9                                    |
| 85   | Mobilenet\_v3              | tf2_mobilenetv3_imagenet_224_224_132M               | 600.45                                | 1.7                                      | 6.6                                     |
| 86   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 107.75                                | 9.3                                      | 36.8                                    |
| 87   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 29.31                                 | 34.1                                     | 162.1                                   |
| 88   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 6.00                                  | 166.7                                    | 336.3                                   |
| 89   | face quality               | pt_face-quality_80_60_61.68M                        | 0.39                                  | 2538.9                                   | 7962.3                                  |
| 90   | multi_task_v2              | pt_MT-resnet18_mixed_320_512_13.65G                 | 31.96                                 | 31.2                                     | 101.9                                   |
| 91   | face_reid_large            | pt_facereid-large_96_96_515M                        | 1.12                                  | 889.2                                    | 2232.6                                  |
| 92   | face_reid_small            | pt_facereid-small_80_80_90M                         | 0.48                                  | 2093.8                                   | 6299.2                                  |
| 93   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 10.27                                 | 97.3                                     | 228.1                                   |
| 94   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 2.80                                  | 356.5                                    | 683.8                                   |
| 95   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G               | 50.46                                 | 19.8                                     | 49.9                                    |
| 96   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 182.18                                | 5.5                                      | 21.1                                    |
| 97   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 27.80                                 | 36.0                                     | 106.1                                   |
| 98   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 44.89                                 | 22.3                                     | 69.5                                    |
| 99   | surround-view pointpillars | pt_pointpillars_nuscenes_40000_64_108G              | 440.31                                | 2.3                                      | 9.9                                     |
| 100  | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 247.47                                | 4.0                                      | 11.1                                    |
| 101  | centerpoint                | pt_centerpoint_astyx_2560_40_54G                    | 62.30                                 | 16.0                                     | 48.2                                    |
| 102  | pointpainting              | pt_pointpainting_nuscenes_126G                      | 778.55                                | 1.3                                      | 4.4                                     |
| 103  | multi_task_v3              | pt_multitaskv3_mixed_320_512_25.44G                 | 60.00                                 | 16.7                                     | 60.9                                    |
| 104  | FADnet                     | pt_fadnet_sceneflow_576_960_441G                    | 889.57                                | 1.1                                      | 1.5                                     |
| 105  | Bayesian Crowd Counting    | pt_BCC_shanghaitech_800_1000_268.9G                 | 316.62                                | 3.2                                      | 10.4                                    |
| 106  | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 6.86                                  | 145.6                                    | 362.9                                   |
| 107  | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 98.19                                 | 10.2                                     | 54.0                                    |
| 108  | Ultra-Fast                 | pt_ultrafast_CULane_288_800_8.4G                    | 30.02                                 | 33.3                                     | 93.9                                    |
| 109  | CLOCs                      | pt_CLOCs_kitti                                      | 347.10                                | 2.9                                      | 10.4                                    |
| 110  | Yolo-X                     | pt_yolox_TT100K_640_640_73G                         | 77.41                                 | 12.9                                     | 33.7                                    |
| 111  | SESR-S                     | pt_SESR-S_DIV2K_360_640_7.48G                       | 12.34                                 | 81.0                                     | 135.0                                   |
| 112  | DRUNet                     | pt_DRUNet_Kvasir_528_608_0.4G                       | 21.87                                 | 45.7                                     | 161.1                                   |
| 113  | SSR                        | pt_SSR_CVC_256_256_39.72G                           | 174.07                                | 5.7                                      | 14.2                                    |
| 114  | C2D2lite                   | pt_C2D2lite_CC20_512_512_6.86G                      | 340.09                                | 2.9                                      | 5.4                                     |
| 115  | HardNet_Mseg               | pt_HardNet_mixed_352_352_22.78G                     | 42.94                                 | 23.3                                     | 60.1                                    |
| 116  | FadNet                     | pt_fadnet_sceneflow_576_960_0.65_154G               | 601.25                                | 1.7                                      | 2.4                                     |
| 117  | Person orientation         | pt_person-orientation_224_112_558M                  | 1.63                                  | 613.3                                    | 1388.6                                  |
| 118  | FairMOT                    | pt_FairMOT_mixed_640_480_0.5_36G                    | 45.56                                 | 21.9                                     | 65.6                                    |
| 119  | OFA                        | pt_OFA-resnet50_imagenet_160_160_900M               | 5.90                                  | 169.5                                    | 348.1                                   |
| 120  | OFA                        | pt_OFA-depthwise-res50_imagenet_176_176_1.25G       | 9.72                                  | 102.8                                    | 376.3                                   |
| 121  | Inception_v3               | pt_inceptionv3_imagenet_299_299_5.7G                | 16.92                                 | 59.1                                     | 136.2                                   |
| 122  | SqueezeNet                 | pt_squeezenet_imagenet_224_224_351.7M               | 3.34                                  | 298.9                                    | 1161.2                                  |
| 123  | resnet50                   | pt_resnet50_imagenet_224_224_4.1G                   | 14.07                                 | 71.0                                     | 168.4                                   |

</details>


### Performance on ZCU104
Measured with Vitis AI 2.0 and Vitis AI Library 2.0 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 300MHz` DPU configuration:


| No\. | Model                      | Name                                                | E2E latency \(ms\) <br/>Thread Num =1 | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 11.68                                 | 85.6                                     | 166.2                                   |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 5.11                                  | 195.7                                    | 415.4                                   |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 5.19                                  | 192.5                                    | 405.9                                   |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 7.10                                  | 140.7                                    | 276.3                                   |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 15.95                                 | 62.7                                     | 122.0                                   |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 32.75                                 | 30.5                                     | 59.0                                    |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 3.74                                  | 267.3                                    | 611.5                                   |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 3.58                                  | 279.4                                    | 986.8                                   |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 12.52                                 | 79.8                                     | 215.7                                   |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 110.15                                | 9.1                                      | 18.4                                    |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 29.27                                 | 34.1                                     | 73.1                                    |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 15.18                                 | 65.9                                     | 152.1                                   |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 11.05                                 | 90.4                                     | 220.2                                   |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 10.94                                 | 91.4                                     | 238.1                                   |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 17.33                                 | 57.7                                     | 150.4                                   |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 10.15                                 | 98.5                                     | 300.6                                   |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 39.95                                 | 25.0                                     | 106.0                                   |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 28.25                                 | 35.4                                     | 146.0                                   |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 1.71                                  | 582.9                                    | 1355.0                                  |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 252.22                                | 4.0                                      | 11.0                                    |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 2.26                                  | 441.4                                    | 1596.4                                  |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 4.59                                  | 217.5                                    | 761.5                                   |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 1.06                                  | 940.7                                    | 1595.2                                  |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 2.62                                  | 380.9                                    | 704.0                                   |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 25.24                                 | 39.6                                     | 108.9                                   |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 75.55                                 | 13.2                                     | 26.6                                    |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 10.47                                 | 95.5                                     | 233.6                                   |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 74.37                                 | 13.4                                     | 26.9                                    |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 35.86                                 | 27.9                                     | 57.2                                    |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 14.98                                 | 66.7                                     | 153.6                                   |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 13.04                                 | 76.6                                     | 180.9                                   |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 11.18                                 | 89.5                                     | 217.3                                   |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 5.68                                  | 175.9                                    | 310.0                                   |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 13.21                                 | 75.6                                     | 145.2                                   |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 75.96                                 | 13.2                                     | 34.5                                    |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 1.89                                  | 529.3                                    | 1971.3                                  |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 4.71                                  | 212.1                                    | 492.4                                   |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 7.69                                  | 130.0                                    | 483.0                                   |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 0.37                                  | 2663.6                                   | 7365.3                                  |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G                | 26.47                                 | 37.8                                     | 129.6                                   |
| 41   | Hourglass                  | cf_hourglass_mpii_256_256_10.2G                     | 53.81                                 | 18.6                                     | 73.7                                    |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 8.14                                  | 122.8                                    | 329.2                                   |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 71.20                                 | 14.0                                     | 28.3                                    |
| 44   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 52.17                                 | 19.2                                     | 38.8                                    |
| 45   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 41.74                                 | 23.9                                     | 45.0                                    |
| 46   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 5.12                                  | 195.4                                    | 410.3                                   |
| 47   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 15.97                                 | 62.6                                     | 121.7                                   |
| 48   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 32.73                                 | 30.5                                     | 59.1                                    |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M         | 0.80                                  | 1247.5                                   | 4084.2                                  |
| 50   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 1.28                                  | 777.8                                    | 2249.5                                  |
| 51   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 3.11                                  | 321.3                                    | 774.8                                   |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 3.83                                  | 261.2                                    | 588.8                                   |
| 53   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 5.20                                  | 192.2                                    | 406.3                                   |
| 54   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 11.74                                 | 85.2                                     | 165.0                                   |
| 55   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 21.98                                 | 45.5                                     | 88.8                                    |
| 56   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 32.12                                 | 31.1                                     | 60.7                                    |
| 57   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 46.62                                 | 21.4                                     | 37.1                                    |
| 58   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 54.01                                 | 18.5                                     | 32.6                                    |
| 59   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 9.08                                  | 110.0                                    | 308.0                                   |
| 60   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 12.62                                 | 79.2                                     | 194.4                                   |
| 61   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 343.64                                | 2.9                                      | 5.2                                     |
| 62   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 71.15                                 | 14.0                                     | 28.1                                    |
| 63   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 537.93                                | 1.9                                      | 5.2                                     |
| 64   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G               | 10.24                                 | 97.6                                     | 195.1                                   |
| 65   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G               | 26.84                                 | 37.2                                     | 84.2                                    |
| 66   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G             | 45.79                                 | 21.8                                     | 46.0                                    |
| 67   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G             | 64.70                                 | 15.4                                     | 31.6                                    |
| 68   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 9.41                                  | 106.2                                    | 276.4                                   |
| 69   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G                | 24.53                                 | 40.7                                     | 87.9                                    |
| 70   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G         | 563.23                                | 1.8                                      | 5.6                                     |
| 71   | efficientnet-edgetpu-S     | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G    | 8.37                                  | 119.4                                    | 247.2                                   |
| 72   | efficientnet-edgetpu-M     | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G    | 12.02                                 | 83.2                                     | 167.6                                   |
| 73   | efficientnet-edgetpu-L     | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G   | 30.06                                 | 33.3                                     | 71.1                                    |
| 74   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 13.12                                 | 76.2                                     | 148.0                                   |
| 75   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 93.58                                 | 10.7                                     | 25.9                                    |
| 76   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M           | 4.71                                  | 212.2                                    | 461.8                                   |
| 77   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M          | 3.93                                  | 254.5                                    | 578.7                                   |
| 78   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G         | 14.35                                 | 69.7                                     | 169.7                                   |
| 79   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 123.90                                | 8.1                                      | 15.3                                    |
| 80   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 11.92                                 | 83.9                                     | 163.2                                   |
| 81   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 3.16                                  | 316.7                                    | 763.7                                   |
| 82   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 16.08                                 | 62.2                                     | 122.0                                   |
| 83   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 6.06                                  | 165.0                                    | 342.2                                   |
| 84   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 136.88                                | 7.3                                      | 25.4                                    |
| 85   | Mobilenet\_v3              | tf2_mobilenetv3_imagenet_224_224_132M               | 679.64                                | 1.5                                      | 5.8                                     |
| 86   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 105.26                                | 9.5                                      | 39.0                                    |
| 87   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 28.58                                 | 35.0                                     | 159.1                                   |
| 88   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 5.62                                  | 177.8                                    | 313.0                                   |
| 89   | face quality               | pt_face-quality_80_60_61.68M                        | 0.38                                  | 2623.5                                   | 7321.1                                  |
| 90   | multi_task_v2              | pt_MT-resnet18_mixed_320_512_13.65G                 | 30.54                                 | 32.7                                     | 92.0                                    |
| 91   | face_reid_large            | pt_facereid-large_96_96_515M                        | 1.07                                  | 936.0                                    | 2050.3                                  |
| 92   | face_reid_small            | pt_facereid-small_80_80_90M                         | 0.46                                  | 2162.8                                   | 5659.8                                  |
| 93   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 9.64                                  | 103.6                                    | 201.2                                   |
| 94   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 2.63                                  | 379.5                                    | 688.6                                   |
| 95   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G               | 49.40                                 | 20.2                                     | 49.7                                    |
| 96   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 180.34                                | 5.5                                      | 21.3                                    |
| 97   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 26.21                                 | 38.1                                     | 79.8                                    |
| 98   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 43.44                                 | 23.0                                     | 59.3                                    |
| 99   | surround-view pointpillars | pt_pointpillars_nuscenes_40000_64_108G              | 429.70                                | 2.3                                      | 9.4                                     |
| 100  | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 239.02                                | 4.2                                      | 11.7                                    |
| 101  | centerpoint                | pt_centerpoint_astyx_2560_40_54G                    | 59.29                                 | 16.9                                     | 20.7                                    |
| 102  | pointpainting              | pt_pointpainting_nuscenes_126G                      | 768.16                                | 1.3                                      | 4.7                                     |
| 103  | multi_task_v3              | pt_multitaskv3_mixed_320_512_25.44G                 | 57.72                                 | 17.3                                     | 54.4                                    |
| 104  | FADnet                     | pt_fadnet_sceneflow_576_960_441G                    | 656.06                                | 1.5                                      | 3.8                                     |
| 105  | Bayesian Crowd Counting    | pt_BCC_shanghaitech_800_1000_268.9G                 | 299.03                                | 3.3                                      | 7.6                                     |
| 106  | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 6.46                                  | 154.7                                    | 313.6                                   |
| 107  | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 96.30                                 | 10.4                                     | 53.6                                    |
| 108  | Ultra-Fast                 | pt_ultrafast_CULane_288_800_8.4G                    | 28.29                                 | 35.3                                     | 76.2                                    |
| 109  | CLOCs                      | pt_CLOCs_kitti                                      | 339.55                                | 2.9                                      | 10.5                                    |
| 110  | Yolo-X                     | pt_yolox_TT100K_640_640_73G                         | 72.64                                 | 13.8                                     | 27.7                                    |
| 111  | SESR-S                     | pt_SESR-S_DIV2K_360_640_7.48G                       | 11.71                                 | 85.4                                     | 138.5                                   |
| 112  | DRUNet                     | pt_DRUNet_Kvasir_528_608_0.4G                       | 20.84                                 | 48.0                                     | 117.9                                   |
| 113  | SSR                        | pt_SSR_CVC_256_256_39.72G                           | 162.99                                | 6.1                                      | 11.9                                    |
| 114  | C2D2lite                   | pt_C2D2lite_CC20_512_512_6.86G                      | 311.13                                | 3.2                                      | 3.6                                     |
| 115  | HardNet_Mseg               | pt_HardNet_mixed_352_352_22.78G                     | 40.44                                 | 24.7                                     | 53.1                                    |
| 116  | FadNet                     | pt_fadnet_sceneflow_576_960_0.65_154G               | 427.27                                | 2.3                                      | 6.4                                     |
| 117  | Person orientation         | pt_person-orientation_224_112_558M                  | 1.54                                  | 647.1                                    | 1310.8                                  |
| 118  | FairMOT                    | pt_FairMOT_mixed_640_480_0.5_36G                    | 43.22                                 | 23.1                                     | 51.9                                    |
| 119  | OFA                        | pt_OFA-resnet50_imagenet_160_160_900M               | 5.49                                  | 182.0                                    | 339.2                                   |
| 120  | OFA                        | pt_OFA-depthwise-res50_imagenet_176_176_1.25G       | 9.48                                  | 105.4                                    | 348.4                                   |
| 121  | Inception_v3               | pt_inceptionv3_imagenet_299_299_5.7G                | 15.95                                 | 62.7                                     | 122.2                                   |
| 122  | SqueezeNet                 | pt_squeezenet_imagenet_224_224_351.7M               | 3.28                                  | 305.0                                    | 1068.2                                  |
| 123  | resnet50                   | pt_resnet50_imagenet_224_224_4.1G                   | 13.21                                 | 75.7                                     | 147.3                                   |

</details>


### Performance on VCK190
Measured with Vitis AI 2.0 and Vitis AI Library 2.0 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Versal` board with 96 AIEs running at 1250 MHz:
  

| No\. | Model                      | Name                                                | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 898.4                                    | 1419.4                                  |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 1310.6                                   | 2691.0                                  |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 1015.0                                   | 1721.5                                  |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 833.3                                    | 1273.5                                  |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 422.6                                    | 589.1                                   |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 240.0                                    | 285.8                                   |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 1348.4                                   | 2911.4                                  |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 2584.8                                   | 4111.4                                  |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 316.6                                    | 708.2                                   |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 115.5                                    | 142.2                                   |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 263.4                                    | 443.2                                   |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 334.6                                    | 673.7                                   |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 403.0                                    | 775.6                                   |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 386.3                                    | 830.9                                   |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 283.8                                    | 677.7                                   |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 272.1                                    | 678.8                                   |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 77.8                                     | 172.6                                   |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 94.8                                     | 219.9                                   |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 2161.0                                   | 4399.8                                  |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 20.9                                     | 39.9                                    |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 1054.5                                   | 2311.2                                  |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 515.7                                    | 1094.0                                  |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 6436.9                                   | 12013.9                                 |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 2750.4                                   | 4856.9                                  |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 153.3                                    | 317.5                                   |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 148.8                                    | 191.1                                   |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 425.4                                    | 944.6                                   |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 151.6                                    | 189.9                                   |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 273.7                                    | 432.0                                   |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 384.7                                    | 793.9                                   |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 406.0                                    | 893.1                                   |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 425.5                                    | 992.4                                   |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 2067.3                                   | 2598.3                                  |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 1080.7                                   | 1208.5                                  |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 68.0                                     | 188.2                                   |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 1243.6                                   | 2591.6                                  |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 916.1                                    | 1557.1                                  |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 344.1                                    | 790.8                                   |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 10145.5                                  | 25104.5                                 |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G                | 105.7                                    | 241.3                                   |
| 41   | Hourglass                  | cf_hourglass_mpii_256_256_10.2G                     | 73.8                                     | 139.1                                   |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 536.1                                    | 1339.3                                  |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 115.1                                    | 158.9                                   |
| 44   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 127.8                                    | 182.8                                   |
| 45   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 243.2                                    | 290.9                                   |
| 46   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 1022.7                                   | 1754.0                                  |
| 47   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 422.8                                    | 590.4                                   |
| 48   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 241.0                                    | 287.4                                   |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M         | 4124.1                                   | 9165.5                                  |
| 50   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 2827.4                                   | 7082.6                                  |
| 51   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 1463.6                                   | 3245.8                                  |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 1336.2                                   | 2914.7                                  |
| 53   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 1104.2                                   | 2021.3                                  |
| 54   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 895.9                                    | 1421.1                                  |
| 55   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 591.4                                    | 781.6                                   |
| 56   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 440.6                                    | 538.7                                   |
| 57   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 308.5                                    | 353.3                                   |
| 58   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 276.1                                    | 311.6                                   |
| 59   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 370.9                                    | 528.3                                   |
| 60   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 323.3                                    | 526.0                                   |
| 61   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 10.8                                     | 12.5                                    |
| 62   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 151.9                                    | 189.7                                   |
| 63   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 10.8                                     | 23.4                                    |
| 64   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G               | 566.9                                    | 739.8                                   |
| 65   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G               | 393.3                                    | 530.8                                   |
| 66   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G             | 267.0                                    | 324.1                                   |
| 67   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G             | 201.9                                    | 232.8                                   |
| 68   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 347.6                                    | 513.8                                   |
| 69   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G                | 208.2                                    | 379.0                                   |
| 70   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G         | 5.1                                      | 13.2                                    |
| 71   | efficientnet-edgetpu-S     | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G    | 805.7                                    | 1205.1                                  |
| 72   | efficientnet-edgetpu-M     | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G    | 587.0                                    | 801.6                                   |
| 73   | efficientnet-edgetpu-L     | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G   | 257.6                                    | 311.2                                   |
| 74   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 827.8                                    | 1251.0                                  |
| 75   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 77.2                                     | 175.7                                   |
| 76   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M           | 1276.0                                   | 2597.2                                  |
| 77   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M          | 1361.2                                   | 2911.7                                  |
| 78   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G         | 413.7                                    | 887.0                                   |
| 79   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 46.4                                     | 59.5                                    |
| 80   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 899.8                                    | 1422.3                                  |
| 81   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 1463.1                                   | 3166.2                                  |
| 82   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 439.0                                    | 623.6                                   |
| 83   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 1164.9                                   | 1958.4                                  |
| 84   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 20.1                                     | 52.2                                    |
| 85   | efficientnet-b0            | tf2_efficientnet-b0_imagenet_224_224_0.36G          | 441.3                                    | 540.4                                   |
| 86   | Mobilenet\_v3              | tf2_mobilenetv3_imagenet_224_224_132M               | 1271.7                                   | 2595.7                                  |
| 87   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 24.1                                     | 55.3                                    |
| 88   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 102.1                                    | 221.5                                   |
| 89   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 2073.6                                   | 2600.1                                  |
| 90   | face quality               | pt_face-quality_80_60_61.68M                        | 10135.7                                  | 25144.8                                 |
| 91   | multi_task_v2              | pt_MT-resnet18_mixed_320_512_13.65G                 | 126.4                                    | 276.9                                   |
| 92   | face_reid_large            | pt_facereid-large_96_96_515M                        | 5529.3                                   | 11537.5                                 |
| 93   | face_reid_small            | pt_facereid-small_80_80_90M                         | 8575.4                                   | 21872.0                                 |
| 94   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 1119.1                                   | 1712.0                                  |
| 95   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 2826.4                                   | 4673.2                                  |
| 96   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G               | 36.4                                     | 57.5                                    |
| 97   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 12.3                                     | 24.8                                    |
| 98   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 365.6                                    | 548.3                                   |
| 99   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 86.5                                     | 217.3                                   |
| 100  | surround-view pointpillars | pt_pointpillars_nuscenes_40000_64_108G              | 7.5                                      | 17.6                                    |
| 101  | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 10.3                                     | 23.9                                    |
| 102  | centerpoint                | pt_centerpoint_astyx_2560_40_54G                    | 113.5                                    | 218.8                                   |
| 103  | pointpainting              | pt_pointpainting_nuscenes_126G                      | 3.7                                      | 6.8                                     |
| 104  | multi_task_v3              | pt_multitaskv3_mixed_320_512_25.44G                 | 73.6                                     | 190.1                                   |
| 105  | FADnet                     | pt_fadnet_sceneflow_576_960_441G                    | 7.0                                      | 11.5                                    |
| 106  | SA-gate                    | pt_sa-gate_NYUv2_360_360_59.71G                     | 14.5                                     | 24.2                                    |
| 107  | Bayesian Crowd Counting    | pt_BCC_shanghaitech_800_1000_268.9G                 | 31.2                                     | 53.5                                    |
| 108  | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 1176.0                                   | 1990.6                                  |
| 109  | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 25.2                                     | 56.0                                    |
| 110  | Ultra-Fast                 | pt_ultrafast_CULane_288_800_8.4G                    | 278.6                                    | 581.0                                   |
| 111  | CLOCs                      | pt_CLOCs_kitti                                      | 8.4                                      | 15.7                                    |
| 112  | SOLO                       | pt_SOLO_coco_640_640_107G                           | 2.0                                      | 3.6                                     |
| 113  | Yolo-X                     | pt_yolox_TT100K_640_640_73G                         | 106.3                                    | 147.4                                   |
| 114  | SESR-S                     | pt_SESR-S_DIV2K_360_640_7.48G                       | 287.6                                    | 471.8                                   |
| 115  | DRUNet                     | pt_DRUNet_Kvasir_528_608_0.4G                       | 163.9                                    | 268.5                                   |
| 116  | SSR                        | pt_SSR_CVC_256_256_39.72G                           | 61.6                                     | 64.4                                    |
| 117  | C2D2lite                   | pt_C2D2lite_CC20_512_512_6.86G                      | 14.6                                     | 17.7                                    |
| 118  | HardNet_Mseg               | pt_HardNet_mixed_352_352_22.78G                     | 172.7                                    | 211.4                                   |
| 119  | FadNet                     | pt_fadnet_sceneflow_576_960_0.65_154G               | 8.1                                      | 13.9                                    |
| 120  | PSMNet                     | pt_psmnet_sceneflow_576_960_0.68_696G               | 0.3                                      | 0.7                                     |
| 121  | Person orientation         | pt_person-orientation_224_112_558M                  | 3771.3                                   | 6992.2                                  |
| 122  | FairMOT                    | pt_FairMOT_mixed_640_480_0.5_36G                    | 158.0                                    | 298.2                                   |
| 123  | OFA                        | pt_OFA-resnet50_imagenet_160_160_900M               | 1460.5                                   | 2329.7                                  |
| 124  | OFA                        | pt_OFA-depthwise-res50_imagenet_176_176_1.25G       | 279.9                                    | 456.3                                   |
| 125  | Inception_v3               | pt_inceptionv3_imagenet_299_299_5.7G                | 421.4                                    | 587.1                                   |
| 126  | SqueezeNet                 | pt_squeezenet_imagenet_224_224_351.7M               | 2605.4                                   | 4256.9                                  |
| 127  | resnet50                   | pt_resnet50_imagenet_224_224_4.1G                   | 827.5                                    | 1252.6                                  |

</details>


### Performance on Kria KV260 SOM
Measured with Vitis AI 2.0 and Vitis AI Library 2.0 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Kria KV260` board with a `1 * B4096F  @ 300MHz` DPU configuration:
  

| No\. | Model                      | Name                                                | E2E latency \(ms\) <br/>Thread Num =1 | E2E throughput \(fps) <br/>Single Thread | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 12.45                                 | 80.3                                     | 84.8                                    |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 5.27                                  | 189.8                                    | 214.2                                   |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 5.32                                  | 187.9                                    | 211.4                                   |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 7.44                                  | 134.4                                    | 146.8                                   |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 16.70                                 | 59.9                                     | 63.7                                    |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 34.16                                 | 29.2                                     | 30.1                                    |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G               | 3.82                                  | 261.9                                    | 310.7                                   |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 3.50                                  | 285.9                                    | 654.9                                   |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 12.43                                 | 80.4                                     | 107.2                                   |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 110.77                                | 9.0                                      | 9.2                                     |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 29.58                                 | 33.8                                     | 36.5                                    |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 15.48                                 | 64.6                                     | 75.6                                    |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 11.19                                 | 89.3                                     | 109.7                                   |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 10.95                                 | 91.3                                     | 118.8                                   |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 17.25                                 | 58.0                                     | 74.4                                    |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 10.16                                 | 98.4                                     | 149.6                                   |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G                 | 39.55                                 | 25.3                                     | 61.2                                    |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 28.24                                 | 35.4                                     | 76.6                                    |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 1.76                                  | 567.5                                    | 739.8                                   |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 258.52                                | 3.9                                      | 5.5                                     |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 2.19                                  | 455.7                                    | 920.7                                   |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 4.55                                  | 219.5                                    | 434.8                                   |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 1.18                                  | 848.9                                    | 945.3                                   |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 2.94                                  | 340.2                                    | 379.8                                   |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 25.75                                 | 38.8                                     | 53.6                                    |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 76.63                                 | 13.0                                     | 13.5                                    |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 10.46                                 | 95.5                                     | 123.1                                   |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 75.44                                 | 13.2                                     | 13.6                                    |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 36.81                                 | 27.2                                     | 28.9                                    |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 14.92                                 | 67.0                                     | 76.8                                    |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 13.05                                 | 76.6                                     | 90.5                                    |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 11.09                                 | 90.2                                     | 108.7                                   |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 6.14                                  | 162.7                                    | 167.5                                   |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 13.65                                 | 73.2                                     | 74.2                                    |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 74.48                                 | 13.4                                     | 17.3                                    |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 1.75                                  | 570.3                                    | 1225.5                                  |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 4.66                                  | 214.4                                    | 279.7                                   |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 7.45                                  | 134.3                                    | 283.6                                   |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 0.36                                  | 2742.2                                   | 4046.0                                  |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G                | 25.94                                 | 38.5                                     | 64.1                                    |
| 41   | Hourglass                  | cf_hourglass_mpii_256_256_10.2G                     | 52.70                                 | 19.0                                     | 57.2                                    |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 7.92                                  | 126.2                                    | 165.8                                   |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 72.43                                 | 13.8                                     | 14.7                                    |
| 44   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 53.72                                 | 18.6                                     | 20.4                                    |
| 45   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 44.01                                 | 22.7                                     | 23.2                                    |
| 46   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 5.28                                  | 189.3                                    | 214.2                                   |
| 47   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 16.74                                 | 59.7                                     | 63.6                                    |
| 48   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 34.15                                 | 29.3                                     | 30.2                                    |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M         | 0.78                                  | 1277.2                                   | 2088.7                                  |
| 50   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M         | 1.27                                  | 784.0                                    | 1135.0                                  |
| 51   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 3.15                                  | 316.9                                    | 393.3                                   |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G           | 3.93                                  | 254.2                                    | 299.9                                   |
| 53   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 5.41                                  | 184.7                                    | 208.4                                   |
| 54   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 12.54                                 | 79.7                                     | 84.2                                    |
| 55   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 23.02                                 | 43.4                                     | 44.6                                    |
| 56   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 23.02                                 | 43.4                                     | 44.6                                    |
| 57   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 51.98                                 | 19.2                                     | 19.5                                    |
| 58   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 59.38                                 | 16.8                                     | 17.0                                    |
| 59   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 9.02                                  | 110.8                                    | 165.7                                   |
| 60   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 12.59                                 | 79.4                                     | 103.7                                   |
| 61   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 346.87                                | 2.9                                      | 5.3                                     |
| 62   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 72.10                                 | 13.9                                     | 14.3                                    |
| 63   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 526.76                                | 1.9                                      | 2.6                                     |
| 64   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G               | 10.56                                 | 94.7                                     | 100.7                                   |
| 65   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G               | 27.19                                 | 36.8                                     | 44.7                                    |
| 66   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G             | 47.00                                 | 21.3                                     | 23.7                                    |
| 67   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G             | 66.86                                 | 14.9                                     | 16.1                                    |
| 68   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 9.39                                  | 106.4                                    | 148.6                                   |
| 69   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G                | 24.97                                 | 40.0                                     | 45.5                                    |
| 70   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G         | 550.57                                | 1.8                                      | 3.1                                     |
| 71   | efficientnet-edgetpu-S     | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G    | 8.64                                  | 115.7                                    | 124.4                                   |
| 72   | efficientnet-edgetpu-M     | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G    | 12.47                                 | 80.2                                     | 84.8                                    |
| 73   | efficientnet-edgetpu-L     | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G   | 30.57                                 | 32.7                                     | 36.4                                    |
| 74   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 13.78                                 | 72.6                                     | 75.9                                    |
| 75   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 92.23                                 | 10.8                                     | 13.0                                    |
| 76   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M           | 4.86                                  | 205.6                                    | 234.2                                   |
| 77   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M          | 4.02                                  | 248.8                                    | 292.4                                   |
| 78   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G         | 14.25                                 | 70.1                                     | 84.2                                    |
| 79   | pruned_rcan                | tf_rcan_DIV2K_360_640_0.98_86.95G                   | 131.87                                | 7.6                                      | 7.8                                     |
| 80   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 12.65                                 | 79.0                                     | 83.2                                    |
| 81   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G              | 3.19                                  | 313.4                                    | 386.3                                   |
| 82   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 16.85                                 | 59.3                                     | 63.1                                    |
| 83   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 6.82                                  | 146.6                                    | 158.4                                   |
| 84   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 138.31                                | 7.2                                      | 12.8                                    |
| 85   | Mobilenet\_v3              | tf2_mobilenetv3_imagenet_224_224_132M               | 611.60                                | 1.6                                      | 6.4                                     |
| 86   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 105.21                                | 9.5                                      | 21.1                                    |
| 87   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 27.70                                 | 36.1                                     | 76.9                                    |
| 88   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 6.08                                  | 164.3                                    | 168.8                                   |
| 89   | face quality               | pt_face-quality_80_60_61.68M                        | 0.39                                  | 2565.0                                   | 4057.2                                  |
| 90   | multi_task_v2              | pt_MT-resnet18_mixed_320_512_13.65G                 | 31.86                                 | 31.3                                     | 43.6                                    |
| 91   | face_reid_large            | pt_facereid-large_96_96_515M                        | 1.11                                  | 899.4                                    | 1072.3                                  |
| 92   | face_reid_small            | pt_facereid-small_80_80_90M                         | 0.46                                  | 2180.2                                   | 3110.2                                  |
| 93   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 10.34                                 | 96.7                                     | 102.4                                   |
| 94   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 2.95                                  | 338.7                                    | 369.6                                   |
| 95   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G               | 47.82                                 | 20.9                                     | 29.0                                    |
| 96   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 168.02                                | 6.0                                      | 19.7                                    |
| 97   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 26.49                                 | 37.7                                     | 39.9                                    |
| 98   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 53.39                                 | 18.7                                     | 22.8                                    |
| 99   | surround-view pointpillars | pt_pointpillars_nuscenes_40000_64_108G              | 466.63                                | 2.1                                      | 5.1                                     |
| 100  | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 235.52                                | 4.2                                      | 9.9                                     |
| 101  | centerpoint                | pt_centerpoint_astyx_2560_40_54G                    | 58.12                                 | 17.2                                     | 20.0                                    |
| 102  | pointpainting              | pt_pointpainting_nuscenes_126G                      | 816.12                                | 1.2                                      | 2.6                                     |
| 103  | multi_task_v3              | pt_multitaskv3_mixed_320_512_25.44G                 | 59.59                                 | 16.8                                     | 25.9                                    |
| 104  | FADnet                     | pt_fadnet_sceneflow_576_960_441G                    | 739.20                                | 1.3                                      | 1.7                                     |
| 105  | Bayesian Crowd Counting    | pt_BCC_shanghaitech_800_1000_268.9G                 | 297.72                                | 3.4                                      | 3.8                                     |
| 106  | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 6.73                                  | 148.5                                    | 160.4                                   |
| 107  | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 93.41                                 | 10.7                                     | 28.1                                    |
| 108  | Ultra-Fast                 | pt_ultrafast_CULane_288_800_8.4G                    | 28.78                                 | 34.7                                     | 38.9                                    |
| 109  | CLOCs                      | pt_CLOCs_kitti                                      | 326.06                                | 3.1                                      | 9.1                                     |
| 110  | Yolo-X                     | pt_yolox_TT100K_640_640_73G                         | 74.80                                 | 13.4                                     | 14.2                                    |
| 111  | SESR-S                     | pt_SESR-S_DIV2K_360_640_7.48G                       | 13.26                                 | 75.4                                     | 82.6                                    |
| 112  | DRUNet                     | pt_DRUNet_Kvasir_528_608_0.4G                       | 20.74                                 | 48.2                                     | 57.9                                    |
| 113  | SSR                        | pt_SSR_CVC_256_256_39.72G                           | 168.61                                | 5.9                                      | 6.0                                     |
| 114  | C2D2lite                   | pt_C2D2lite_CC20_512_512_6.86G                      | 343.40                                | 2.9                                      | 3.1                                     |
| 115  | HardNet_Mseg               | pt_HardNet_mixed_352_352_22.78G                     | 44.02                                 | 22.7                                     | 26.2                                    |
| 116  | FadNet                     | pt_fadnet_sceneflow_576_960_0.65_154G               | 514.94                                | 1.9                                      | 2.6                                     |
| 117  | Person orientation         | pt_person-orientation_224_112_558M                  | 1.60                                  | 623.3                                    | 707.7                                   |
| 118  | FairMOT                    | pt_FairMOT_mixed_640_480_0.5_36G                    | 43.58                                 | 22.9                                     | 26.0                                    |
| 119  | OFA                        | pt_OFA-resnet50_imagenet_160_160_900M               | 5.96                                  | 167.7                                    | 179.8                                   |
| 120  | OFA                        | pt_OFA-depthwise-res50_imagenet_176_176_1.25G       | 9.25                                  | 108.1                                    | 262.5                                   |
| 121  | Inception_v3               | pt_inceptionv3_imagenet_299_299_5.7G                | 16.68                                 | 59.9                                     | 63.8                                    |
| 122  | SqueezeNet                 | pt_squeezenet_imagenet_224_224_351.7M               | 3.21                                  | 311.8                                    | 675.5                                   |
| 123  | resnet50                   | pt_resnet50_imagenet_224_224_4.1G                   | 13.94                                 | 71.7                                     | 75.0                                    |

</details>


### Performance on VCK5000
Measured with Vitis AI 2.0 and Vitis AI Library 2.0 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H running at 8PE@350
MHz in Gen3x16:

| No\. | Model                      | Name                                           | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :--------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G              | 350                | 4941.68                                 |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G             | 350                | 6621.65                                 |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G          | 350                | 3967.01                                 |
| 4    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G           | 350                | 8768.06                                 |
| 5    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G        | 350                | 751.79                                  |
| 6    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                 | 350                | 285.42                                  |
| 7    | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G              | 350                | 667.24                                  |
| 8    | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G          | 350                | 854.08                                  |
| 9    | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G           | 350                | 880.72                                  |
| 10   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G               | 350                | 925.38                                  |
| 11   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                | 350                | 823.66                                  |
| 12   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G        | 350                | 619.57                                  |
| 13   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                 | 350                | 1034.06                                 |
| 14   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G            | 350                | 8444.81                                 |
| 15   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G    | 350                | 170.05                                  |
| 16   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                | 350                | 5972.10                                 |
| 17   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                | 350                | 3054.66                                 |
| 18   | face_landmark              | cf_landmark_celeba_96_72_0.14G                 | 350                | 20320.80                                |
| 19   | reid                       | cf_reid_market1501_160_80_0.95G                | 350                | 10771.90                                |
| 20   | multi_task                 | cf_multitask_bdd_288_512_14.8G                 | 350                | 715.77                                  |
| 21   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                    | 350                | 399.01                                  |
| 22   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G         | 350                | 1441.31                                 |
| 23   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                   | 350                | 475.68                                  |
| 24   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                      | 350                | 961.64                                  |
| 25   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G              | 350                | 1531.11                                 |
| 26   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G               | 350                | 1764.26                                 |
| 27   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G               | 350                | 1535.86                                 |
| 28   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G              | 350                | 554.92                                  |
| 29   | plate detection            | cf_plate-detection_320_320_0.49G               | 350                | 8169.45                                 |
| 30   | plate recognition          | cf_plate-recognition_96_288_1.75G              | 350                | 2995.55                                 |
| 31   | face_quality               | cf_face-quality_80_60_61.68M                   | 350                | 31443.70                                |
| 32   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                   | 350                | 2590.73                                 |
| 33   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G             | 350                | 4204.56                                 |
| 34   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G          | 350                | 4939.12                                 |
| 35   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G         | 350                | 2975.00                                 |
| 36   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G        | 350                | 2120.41                                 |
| 37   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G       | 350                | 114.26                                  |
| 38   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                   | 350                | 477.08                                  |
| 39   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G         | 350                | 75.94 (6 thread)                        |
| 40   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G      | 350                | 4505.04                                 |
| 41   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                 | 350                | 398.16                                  |
| 42   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G    | 350                | 1272.02                                 |
| 43   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G            | 350                | 4941.54                                 |
| 44   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G               | 350                | 1511.86                                 |
| 45   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G             | 350                | 118.02 (6 thread)                       |
| 46   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G               | 350                | 147.57                                  |
| 47   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G          | 350                | 1089.61                                 |
| 48   | face quality               | pt_face-quality_80_60_61.68M                   | 350                | 31639.60                                |
| 49   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G  | 350                | 158.53                                  |
| 50   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G      | 350                | 1176.98                                 |
| 51   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                 | 350                | 246.54                                  |
| 52   | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G | 350                | 87.66                                   |
| 53   | SqueezeNet                 | pt_squeezenet_imagenet_224_224_351.7M          | 350                | 7579.40                                 |
| 54   | resnet50                   | pt_resnet50_imagenet_224_224_4.1G_2.0          | 350                | 4529.60                                 |
| 55   | UltraFast                  | pt_ultrafast_CULane_288_800_8.4G               | 350                | 1373.24                                 |
| 56   | DRUNet                     | pt_DRUNet_Kvasir_528_608_0.4G                  | 350                | 204.26                                  |
| 57   | SESR_S                     | pt_SESR-S_DIV2K_360_640_7.48G                  | 350                | 232.84                                  |
| 58   | FairMOT                    | pt_FairMOT_mixed_640_480_0.5_36G               | 350                | 500.23                                  |


The following table lists the performance number including end-to-end throughput for models on the `Versal ACAP VCK5000` board with DPUCVDX8H DWC running at 6PE@350
MHz in Gen3x16:


| No\. | Model                      | Name                                                | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 350                | 3738.35                                 |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 350                | 5185.26                                 |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 350                | 3293.69                                 |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 350                | 2613.47                                 |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 350                | 912.40                                  |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 350                | 506.92                                  |
| 7    | mobilenetv2                | cf_mobilenetv2_imagenet_224_224_0.59G               | 350                | 3752.71                                 |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 350                | 7672.00                                 |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 350                | 605.79                                  |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 350                | 234.10                                  |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 350                | 513.16                                  |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 350                | 649.21                                  |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 350                | 687.19                                  |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 350                | 714.24                                  |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 350                | 697.63                                  |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 350                | 443.39                                  |
| 17   | ssd_mobilenetv2            | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 350                | 590.97                                  |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 350                | 933.44                                  |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 350                | 6753.90                                 |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 350                | 132.81                                  |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 350                | 4581.78                                 |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 350                | 2339.43                                 |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 350                | 22446.80                                |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 350                | 8422.35                                 |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 350                | 660.98                                  |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 350                | 320.06                                  |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 350                | 1120.72                                 |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 350                | 391.59                                  |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 350                | 825.49                                  |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 350                | 1208.89                                 |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 350                | 1351.28                                 |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 350                | 1327.28                                 |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 350                | 4498.68                                 |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 350                | 2280.95                                 |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 350                | 426.93                                  |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 350                | 6612.75                                 |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 350                | 2759.17                                 |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 350                | 1627.01                                 |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 350                | 31512.70                                |
| 40   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 350                | 1971.40                                 |
| 41   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 350                | 326.98                                  |
| 42   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 350                | 313.13                                  |
| 43   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 350                | 492.18                                  |
| 44   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 350                | 3532.76                                 |
| 45   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G               | 350                | 483.27                                  |
| 46   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 350                | 913.67                                  |
| 47   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 350                | 506.77                                  |
| 48   | mobilenetv1_0.25           | tf_mobilenetv1_0.25_imagenet_128_128_27M            | 350                | 22781.30                                |
| 49   | mobilenetv1_0.5            | tf_mobilenetv1_0.5_imagenet_160_160_150M            | 350                | 11945.00                                |
| 50   | mobilenetv1_1.0            | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 350                | 5224.42                                 |
| 51   | mobilenetv2_1.0            | tf_mobilenetv2_1.0_imagenet_224_224_602M            | 350                | 3638.07                                 |
| 52   | mobilenetv2_1.4            | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 350                | 2842.98                                 |
| 53   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 350                | 3739.72                                 |
| 54   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 350                | 2244.88                                 |
| 55   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 350                | 1596.44                                 |
| 56   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 350                | 505.57                                  |
| 57   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 350                | 450.82                                  |
| 58   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 350                | 108.19                                  |
| 59   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 350                | 2338.31                                 |
| 60   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 350                | 1287.66                                 |
| 61   | ssdlite_mobilenet_v2       | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 350                | 1482.77                                 |
| 62   | ssd_inception_v2           | tf_ssdinceptionv2_coco_300_300_9.62G                | 350                | 240.00                                  |
| 63   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 350                | 392.26                                  |
| 64   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 350                | 72.21                                   |
| 65   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 350                | 3406.26                                 |
| 66   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 350                | 307.83                                  |
| 67   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G         | 350                | 1000.00                                 |
| 68   | efficientNet-edgetpu-S     | tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G    | 350                | 1858.11                                 |
| 69   | efficientNet-edgetpu-M     | tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G    | 350                | 1100.45                                 |
| 70   | efficientNet-edgetpu-L     | tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G   | 350                | 427.57                                  |
| 71   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M           | 350                | 4298.57                                 |
| 72   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M          | 350                | 4813.85                                 |
| 73   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 350                | 3737.98                                 |
| 74   | mobilenetv1                | tf2_mobilenetv1_imagenet_224_224_1.15G              | 350                | 5222.53                                 |
| 75   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 350                | 962.69                                  |
| 76   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 350                | 1358.26                                 |
| 77   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 350                | 115.01                                  |
| 78   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 350                | 141.94                                  |
| 79   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 350                | 1094.60                                 |
| 80   | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 350                | 216.55                                  |
| 81   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 350                | 4496.60                                 |
| 82   | face quality               | pt_face-quality_80_60_61.68M                        | 350                | 32034.80                                |
| 83   | face_reid_large            | pt_facereid-large_96_96_515M                        | 350                | 21110.00                                |
| 84   | face_reid_small            | pt_facereid-small_80_80_90M                         | 350                | 33214.20                                |
| 85   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 350                | 3750.77                                 |
| 86   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 350                | 8172.30                                 |
| 87   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 350                | 154.52                                  |
| 88   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 350                | 960.40                                  |
| 89   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 350                | 201.87                                  |
| 90   | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 350                | 90.88                                   |
| 91   | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 350                | 3477.35                                 |
| 92   | Inception_v3               | pt_inceptionv3_imagenet_299_299_5.7G                | 350                | 910.16                                  |
| 93   | SqueezeNet                 | pt_squeezenet_imagenet_224_224_351.7M               | 350                | 7132.38                                 |
| 94   | resnet50                   | pt_resnet50_imagenet_224_224_4.1G_2.0               | 350                | 3429.55                                 |
| 95   | UltraFast                  | pt_ultrafast_CULane_288_800_8.4G                    | 350                | 1061.32                                 |
| 96   | DRUNet                     | pt_DRUNet_Kvasir_528_608_0.4G                       | 350                | 153.40                                  |
| 97   | SESR_S                     | pt_SESR-S_DIV2K_360_640_7.48G                       | 350                | 185.58                                  |
| 98   | FairMOT                    | pt_FairMOT_mixed_640_480_0.5_36G                    | 350                | 427.84                                  |
| 99   | Person-orientation         | pt_person-orientation_224_112_558M                  | 350                | 9541.32                                 |
| 100  | TSD_YoloX                  | pt_yolox_TT100K_640_640_73G                         | 350                | 263.49                                  |
| 101  | ofa_resnet50               | pt_OFA-resnet50_imagenet_160_160_900M               | 350                | 4050.59                                 |
| 102  | ofa_resnet50_depthwise     | pt_OFA-depthwise-res50_imagenet_176_176_1.246G      | 350                | 3378.06                                 |

</details>


### Performance on U50lv
Measured with Vitis AI 2.0 and Vitis AI Library 2.0 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for each model on the `Alveo U50` board with 10 DPUCAHX8H kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                      | Name                                           | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :--------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G              | 192.5              | 691.01                                  |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G             | 192.5              | 1514.25                                 |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G          | 192.5              | 1340.78                                 |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G             | 192.5              | 1056.64                                 |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G          | 192.5              | 436.96                                  |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G          | 192.5              | 198.69                                  |
| 7    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G           | 192.5              | 3932.12                                 |
| 8    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G        | 192.5              | 678.48                                  |
| 9    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                 | 192.5              | 59.07                                   |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G              | 192.5              | 235.88                                  |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G          | 192.5              | 480.04                                  |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G           | 192.5              | 703.73                                  |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G               | 192.5              | 731.10                                  |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                | 192.5              | 461.08                                  |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G        | 192.5              | 646.59                                  |
| 16   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                 | 192.5              | 474.75                                  |
| 17   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G            | 192.5              | 3594.50                                 |
| 18   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G    | 192.5              | 34.78                                   |
| 19   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                | 192.5              | 3040.52                                 |
| 20   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                | 192.5              | 1316.35                                 |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                 | 192.5              | 11396.4                                 |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G                | 192.5              | 4270.81                                 |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                 | 192.5              | 361.93                                  |
| 24   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                    | 165                | 89.10                                   |
| 25   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G         | 165                | 742.51                                  |
| 26   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                   | 165                | 78.56                                   |
| 27   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                      | 165                | 193.58                                  |
| 28   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G              | 192.5              | 489.60                                  |
| 29   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G               | 192.5              | 575.05                                  |
| 30   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G               | 192.5              | 696.65                                  |
| 31   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                | 192.5              | 1453.98                                 |
| 32   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                 | 192.5              | 530.59                                  |
| 33   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G              | 192.5              | 110.01                                  |
| 34   | plate detection            | cf_plate-detection_320_320_0.49G               | 192.5              | 6369.09                                 |
| 35   | plate recognition          | cf_plate-recognition_96_288_1.75G              | 192.5              | 1396.99                                 |
| 36   | face_quality               | cf_face-quality_80_60_61.68M                   | 192.5              | 21514.9                                 |
| 37   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                   | 192.5              | 1035.44                                 |
| 38   | yolov4                     | dk_yolov4_coco_416_416_60.1G                   | 165                | 82.96                                   |
| 39   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G              | 192.5              | 91.36                                   |
| 40   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G   | 192.5              | 184.68                                  |
| 41   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G             | 192.5              | 1359.83                                 |
| 42   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G         | 192.5              | 438.49                                  |
| 43   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G         | 192.5              | 198.88                                  |
| 44   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G          | 192.5              | 690.78                                  |
| 45   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G         | 192.5              | 358.87                                  |
| 46   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G        | 192.5              | 239.28                                  |
| 47   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G               | 192.5              | 176.15                                  |
| 48   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G               | 192.5              | 146.51                                  |
| 49   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G       | 192.5              | 37.78                                   |
| 50   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                   | 165                | 78.70                                   |
| 51   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G         | 192.5              | 16.44                                   |
| 52   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G      | 192.5              | 596.07                                  |
| 53   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                 | 192.5              | 84.72                                   |
| 54   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G    | 192.5              | 502.22                                  |
| 55   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G            | 192.5              | 691.62                                  |
| 56   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G         | 192.5              | 444.04                                  |
| 57   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G               | 192.5              | 1250.55                                 |
| 58   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G             | 192.5              | 63.02                                   |
| 59   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G               | 192.5              | 95.44                                   |
| 60   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G          | 192.5              | 505.79                                  |
| 61   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G          | 192.5              | 1453.13                                 |
| 62   | face quality               | pt_face-quality_80_60_61.68M                   | 192.5              | 21385.4                                 |
| 63   | face_reid_large            | pt_facereid-large_96_96_515M                   | 192.5              | 8417.59                                 |
| 64   | face_reid_small            | pt_facereid-small_80_80_90M                    | 192.5              | 23548.7                                 |
| 65   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G    | 192.5              | 960.45                                  |
| 66   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G     | 192.5              | 4052.47                                 |
| 67   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G  | 192.5              | 145.11                                  |
| 68   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G      | 192.5              | 247.6                                   |
| 69   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                 | 192.5              | 85.58                                   |
| 70   | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G | 192.5              | 44.21                                   |
| 71   | PMG                        | pt_pmg_rp2k_224_224_2.28G                      | 192.5              | 1170.22                                 |
| 72   | Inception_v3               | pt_inceptionv3_imagenet_299_299_5.7G           | 192.5              | 437.18                                  |
| 73   | SqueezeNet                 | pt_squeezenet_imagenet_224_224_351.7M          | 192.5              | 4186.64                                 |
| 74   | resnet50                   | pt_resnet50_imagenet_224_224_4.1G_2.0          | 192.5              | 596.10                                  |
| 75   | UltraFast                  | pt_ultrafast_CULane_288_800_8.4G               | 192.5              | 288.60                                  |
| 76   | DRUNet                     | pt_DRUNet_Kvasir_528_608_0.4G                  | 192.5              | 372.39                                  |
| 77   | SESR_S                     | pt_SESR-S_DIV2K_360_640_7.48G                  | 192.5              | 184.14                                  |
| 78   | FairMOT                    | pt_FairMOT_mixed_640_480_0.5_36G               | 165                | 160.70                                  |
| 79   | Person-orientation         | pt_person-orientation_224_112_558M             | 192.5              | 6733.67                                 |
| 80   | TSD_YoloX                  | pt_yolox_TT100K_640_640_73G                    | 165                | 72.22                                   |
| 81   | ofa_resnet50               | pt_OFA-resnet50_imagenet_160_160_900M          | 192.5              | 1767.30                                 |


The following table lists the performance number including end-to-end throughput for each model on the `Alveo U50` board with 8 DPUCAHX8H DWC kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                      | Name                                                | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 137.5              | 396.71                                  |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 137.5              | 871.54                                  |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 137.5              | 776.98                                  |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 137.5              | 608.44                                  |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 137.5              | 250.90                                  |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 137.5              | 114.06                                  |
| 7    | mobilenetv2                | cf_mobilenetv2_imagenet_224_224_0.59G               | 137.5              | 1883.93                                 |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 137.5              | 2299.17                                 |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 137.5              | 389.98                                  |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 137.5              | 33.86                                   |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 137.5              | 135.29                                  |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 137.5              | 275.63                                  |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 137.5              | 403.98                                  |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 137.5              | 430.43                                  |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 137.5              | 264.78                                  |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 137.5              | 424.78                                  |
| 17   | ssd_mobilenetv2            | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 137.5              | 255.39                                  |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 137.5              | 275.09                                  |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 137.5              | 2117.40                                 |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 137.5              | 20.05                                   |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 137.5              | 2180.70                                 |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 137.5              | 954.51                                  |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 137.5              | 6770.42                                 |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 137.5              | 2488.00                                 |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 137.5              | 207.78                                  |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 137.5              | 50.93                                   |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 137.5              | 435.65                                  |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 137.5              | 52.35                                   |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 137.5              | 110.82                                  |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 137.5              | 281.58                                  |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 137.5              | 330.35                                  |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 137.5              | 399.89                                  |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 137.5              | 837.13                                  |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 137.5              | 304.40                                  |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 137.5              | 63.27                                   |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 137.5              | 4798.49                                 |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 137.5              | 884.38                                  |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 137.5              | 1171.76                                 |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 137.5              | 17806.60                                |
| 40   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 137.5              | 594.69                                  |
| 41   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 137.5              | 55.46                                   |
| 42   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 137.5              | 63.78                                   |
| 43   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 137.5              | 106.01                                  |
| 44   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 137.5              | 788.10                                  |
| 45   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 137.5              | 251.22                                  |
| 46   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 137.5              | 113.95                                  |
| 47   | mobilenetv1_0.25           | tf_mobilenetv1_0.25_imagenet_128_128_27M            | 137.5              | 13994.40                                |
| 48   | mobilenetv1_0.5            | tf_mobilenetv1_0.5_imagenet_160_160_150M            | 137.5              | 7508.79                                 |
| 49   | mobilenetv1_1.0            | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 137.5              | 2021.15                                 |
| 50   | mobilenetv2_1.0            | tf_mobilenetv2_1.0_imagenet_224_224_602M            | 137.5              | 1862.71                                 |
| 51   | mobilenetv2_1.4            | tf_mobilenetv2_1.4_imagenet_224_224_1.16G           | 137.5              | 1252.61                                 |
| 52   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 137.5              | 396.44                                  |
| 53   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 137.5              | 205.59                                  |
| 54   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 137.5              | 136.93                                  |
| 55   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 137.5              | 101.28                                  |
| 56   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 137.5              | 84.05                                   |
| 57   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 137.5              | 21.61                                   |
| 58   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 137.5              | 943.25                                  |
| 59   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 137.5              | 530.97                                  |
| 60   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 137.5              | 873.85                                  |
| 61   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 137.5              | 52.53                                   |
| 62   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 137.5              | 9.41                                    |
| 63   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 137.5              | 342.16                                  |
| 64   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 137.5              | 48.37                                   |
| 65   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G         | 137.5              | 289.07                                  |
| 66   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 137.5              | 397.0                                   |
| 67   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 137.5              | 254.60                                  |
| 68   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 137.5              | 719.06                                  |
| 69   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 137.5              | 39.18                                   |
| 70   | mobilenetv1                | tf2_mobilenetv1_imagenet_224_224_1.15G              | 137.5              | 2021.57                                 |
| 71   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 137.5              | 59.20                                   |
| 72   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 137.5              | 290.10                                  |
| 73   | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 137.5              | 120.07                                  |
| 74   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 137.5              | 838.06                                  |
| 75   | face quality               | pt_face-quality_80_60_61.68M                        | 137.5              | 17811.40                                |
| 76   | face_reid_large            | pt_facereid-large_96_96_515M                        | 137.5              | 4997.13                                 |
| 77   | face_reid_small            | pt_facereid-small_80_80_90M                         | 137.5              | 15164.50                                |
| 78   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 137.5              | 552.22                                  |
| 79   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 137.5              | 2361.76                                 |
| 80   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 137.5              | 113.08                                  |
| 81   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 137.5              | 142.42                                  |
| 82   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 137.5              | 61.21                                   |
| 83   | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 137.5              | 24.91                                   |
| 84   | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 137.5              | 672.85                                  |
| 85   | Inception_v3               | pt_inceptionv3_imagenet_299_299_5.7G                | 137.5              | 251.21                                  |
| 86   | SqueezeNet                 | pt_squeezenet_imagenet_224_224_351.7M               | 137.5              | 2447.79                                 |
| 87   | resnet50                   | pt_resnet50_imagenet_224_224_4.1G_2.0               | 137.5              | 341.97                                  |
| 88   | UltraFast                  | pt_ultrafast_CULane_288_800_8.4G                    | 137.5              | 164.96                                  |
| 89   | DRUNet                     | pt_DRUNet_Kvasir_528_608_0.4G                       | 137.5              | 214.20                                  |
| 90   | SESR_S                     | pt_SESR-S_DIV2K_360_640_7.48G                       | 137.5              | 135.92                                  |
| 91   | FairMOT                    | pt_FairMOT_mixed_640_480_0.5_36G                    | 137.5              | 92.15                                   |
| 92   | Person-orientation         | pt_person-orientation_224_112_558M                  | 137.5              | 3981.36                                 |
| 93   | TSD_YoloX                  | pt_yolox_TT100K_640_640_73G                         | 137.5              | 48.44                                   |
| 94   | ofa_resnet50               | pt_OFA-resnet50_imagenet_160_160_900M               | 137.5              | 1044.55                                 |

</details>


### Performance on U55C DWC
Measured with Vitis AI 2.0 and Vitis AI Library 2.0 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput for each model on the `Alveo U55C` board with 11 DPUCAHX8H DWC kernels running at 300Mhz in Gen3x4:
  

| No\. | Model                      | Name                                                | DPU Frequency(MHz) | E2E throughput \(fps) <br/>Multi Thread |
| ---- | :------------------------- | :-------------------------------------------------- | ------------------ | --------------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G                   | 300                | 1178.18                                 |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G                  | 300                | 2609.77                                 |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G               | 300                | 2135.29                                 |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G                  | 300                | 1731.11                                 |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G               | 300                | 697.38                                  |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G               | 300                | 326.77                                  |
| 7    | mobilenetv2                | cf_mobilenetv2_imagenet_224_224_0.59G               | 350                | 5370.48                                 |
| 8    | SqueezeNet                 | cf_squeezenet_imagenet_227_227_0.76G                | 300                | 6173.84                                 |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G             | 300                | 904.75                                  |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                      | 300                | 94.32                                   |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G                   | 300                | 331.45                                  |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G               | 300                | 717.55                                  |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G                | 300                | 1011.15                                 |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G                    | 300                | 994.51                                  |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G                     | 300                | 672.57                                  |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G             | 300                | 946.99                                  |
| 17   | ssd_mobilenetv2            | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 350                | 720.55                                  |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                      | 300                | 725.33                                  |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G                 | 300                | 5669.09                                 |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G         | 300                | 57.89                                   |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G                     | 300                | 4922.83                                 |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G                     | 300                | 2205.46                                 |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                      | 300                | 19902.40                                |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G                     | 300                | 7476.71                                 |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                      | 300                | 536.58                                  |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                         | 300                | 147.58                                  |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G              | 300                | 1229.14                                 |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                        | 300                | 152.95                                  |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                           | 300                | 318.21                                  |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G                   | 300                | 779.56                                  |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G                    | 300                | 906.92                                  |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G                    | 300                | 1091.51                                 |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G                     | 300                | 2500.45                                 |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                      | 300                | 907.31                                  |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G                   | 300                | 182.27                                  |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G                    | 300                | 7895.80                                 |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G                   | 300                | 2312.90                                 |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G                   | 350                | 1764.23                                 |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                        | 300                | 30667.30                                |
| 40   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                        | 300                | 1634.88                                 |
| 41   | yolov4                     | dk_yolov4_coco_416_416_60.1G                        | 300                | 156.64                                  |
| 42   | pruned_yolov4              | dk_yolov4_coco_416_416_0.36_38.2G                   | 300                | 166.85                                  |
| 43   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G        | 300                | 301.68                                  |
| 44   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G                  | 300                | 2214.22                                 |
| 45   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G              | 300                | 698.57                                  |
| 46   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G              | 300                | 327.53                                  |
| 47   | mobilenetv1_0.25           | tf_mobilenetv1_0.25_imagenet_128_128_27M            | 350                | 18874.90                                |
| 48   | mobilenetv1_0.5            | tf_mobilenetv1_0.5_imagenet_160_160_150M            | 350                | 12898.20                                |
| 49   | mobilenetv1_1.0            | tf_mobilenetv1_1.0_imagenet_224_224_1.14G           | 350                | 5325.89                                 |
| 50   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G               | 300                | 1178.59                                 |
| 51   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G              | 300                | 611.32                                  |
| 52   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G             | 300                | 407.69                                  |
| 53   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G                    | 300                | 295.64                                  |
| 54   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G                    | 300                | 246.94                                  |
| 55   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G            | 300                | 59.37                                   |
| 56   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G                | 350                | 2180.88                                 |
| 57   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G                | 350                | 1472.47                                 |
| 58   | ssdlite_mobilenet_v2       | tf_ssdlite_mobilenetv2_coco_300_300_1.5G            | 350                | 2135.17                                 |
| 59   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                        | 300                | 152.84                                  |
| 60   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G              | 300                | 1015.18                                 |
| 61   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G           | 300                | 25.83                                   |
| 62   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                      | 300                | 138.97                                  |
| 63   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_0.88_9.83G         | 300                | 797.82                                  |
| 64   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G                 | 300                | 1178.36                                 |
| 65   | mobilenetv1                | tf2_mobilenetv1_imagenet_224_224_1.15G              | 350                | 5340.33                                 |
| 66   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G              | 300                | 714.44                                  |
| 67   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G                    | 300                | 2021.76                                 |
| 68   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G                  | 300                | 90.59                                   |
| 69   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G                    | 300                | 144.02                                  |
| 70   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G               | 300                | 782.05                                  |
| 71   | SemanticFPN-mobilenetv2    | pt_SemanticFPN-mobilenetv2_cityscapes_512_1024_5.4G | 350                | 230.77                                  |
| 72   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G               | 300                | 2499.11                                 |
| 73   | face quality               | pt_face-quality_80_60_61.68M                        | 300                | 30562.20                                |
| 74   | face_reid_large            | pt_facereid-large_96_96_515M                        | 300                | 15187.80                                |
| 75   | face_reid_small            | pt_facereid-small_80_80_90M                         | 300                | 33132.60                                |
| 76   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G         | 300                | 1637.65                                 |
| 77   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G          | 300                | 7096.63                                 |
| 78   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G       | 300                | 152.34                                  |
| 79   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G           | 300                | 408.05                                  |
| 80   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                      | 300                | 138.36                                  |
| 81   | pointpillars_nuscenes      | pt_pointpillars_nuscenes_40000_64_108G              | 300                | 42.50                                   |
| 82   | salsanext_v2               | pt_salsanextv2_semantic-kitti_64_2048_0.75_32G      | 300                | 58.46                                   |
| 83   | pointpainting              | pt_pointpainting_nuscenes_126G                      | 300                | 21.28                                   |
| 84   | PMG                        | pt_pmg_rp2k_224_224_2.28G                           | 300                | 1995.08                                 |
| 85   | Inception_v3               | pt_inceptionv3_imagenet_299_299_5.7G                | 300                | 697.54                                  |
| 86   | SqueezeNet                 | pt_squeezenet_imagenet_224_224_351.7M               | 300                | 6560.42                                 |
| 87   | resnet50                   | pt_resnet50_imagenet_224_224_4.1G_2.0               | 300                | 1015.14                                 |
| 88   | UltraFast                  | pt_ultrafast_CULane_288_800_8.4G                    | 300                | 481.47                                  |
| 89   | DRUNet                     | pt_DRUNet_Kvasir_528_608_0.4G                       | 300                | 479.90                                  |
| 90   | SESR_S                     | pt_SESR-S_DIV2K_360_640_7.48G                       | 300                | 290.95                                  |
| 91   | FairMOT                    | pt_FairMOT_mixed_640_480_0.5_36G                    | 300                | 239.92                                  |
| 92   | Person-orientation         | pt_person-orientation_224_112_558M                  | 300                | 11906.10                                |
| 93   | TSD_YoloX                  | pt_yolox_TT100K_640_640_73G                         | 300                | 132.59                                  |
| 94   | ofa_resnet50               | pt_OFA-resnet50_imagenet_160_160_900M               | 300                | 2942.50                                 |


</details>


### Performance on U200
Measured with Vitis AI 2.0 and Vitis AI Library 2.0 

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
Measured with Vitis AI 2.0 and Vitis AI Library 2.0 

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
