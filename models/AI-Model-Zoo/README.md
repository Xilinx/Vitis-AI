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

## Vitis AI 1.3 Model Zoo New Features！
1.New added 28 models. Total 92 models from different deep learning frameworks, Caffe, TensorFlow, TensorFlow 2 and PyTorch are supported in Vitis AI Model Zoo. </br>

2.The variety of Model Zoo has been significantly improved for a wider range of applications.</br> 
For medical applications, we added CT image segmentation, medical robot instrument segmentation, Covid-19 chest radiograph segmentation and other reference models.</br> 
For autonomous driving and ADAS applications, we added 3D point cloud detection and point cloud segmentation models. </br>

3.Provided accuracy evaluation and quantization scripts for all the released models.</br>

4.Restructured model list show all versions of models that running on all supported Xilinx platforms in a clearer and more explicit way, and users could download the specific version they need for every model.</br>

## Model Details
The following table includes comprehensive information about each model, including application, framework, training and validation dataset, backbone, input size, computation as well as float and quantized precision.

<details>
 <summary><b>Click here to view details</b></summary>

| No\. |          Application          |             Model              | Name                                                      | Framework  |     Backbone      | Input Size | OPS per image |              Training Set               |         Val Set         |               Float Accuracy     (Top1/ Top5\)               |               Quantized Accuracy (Top1/Top5\)                |
| :--: | :---------------------------: | :----------------------------: | :-------------------------------------------------------- | :--------: | :---------------: | :--------: | :-----------: | :-------------------------------------: | :---------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  1   |     Image Classification      |            resnet50            | cf\_resnet50\_imagenet\_224\_224\_7\.7G\_1\.3             |   caffe    |     resnet50      |  224\*224  |     7\.7G     |             ImageNet Train              |      ImageNet Val       |                       0\.7444/0\.9185                        |                       0\.7335/0\.9130                        |
|  2   |      Image Classifiction      |            resnet18            | cf\_resnet18\_imagenet\_224\_224\_3\.65G\_1\.3            |   caffe    |     resnet18      |  224\*224  |    3\.65G     |             ImageNet Train              |      ImageNet Val       |                       0\.6844/0\.8864                        |                        0.6699/0.8826                         |
|  3   |     Image Classification      |         Inception\_v1          | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G\_1\.3         |   caffe    |   inception\_v1   |  224\*224  |    3\.16G     |             ImageNet Train              |      ImageNet Val       |                       0\.7030/0\.8971                        |                       0\.6964/0\.8942                        |
|  4   |     Image Classification      |         Inception\_v2          | cf\_inceptionv2\_imagenet\_224\_224\_4G\_1\.3             |   caffe    |   bn\-inception   |  224\*224  |      4G       |             ImageNet Train              |      ImageNet Val       |                       0\.7275/0\.9111                        |                       0\.7166/0\.9029                        |
|  5   |     Image Classification      |         Inception\_v3          | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G\_1\.3         |   caffe    |   inception\_v3   |  299\*299  |    11\.4G     |             ImageNet Train              |      ImageNet Val       |                       0\.7701/0\.9329                        |                       0\.7622/0\.9302                        |
|  6   |     Image Classification      |         Inception\_v4          | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G\_1\.3         |   caffe    |   inception\_v3   |  299\*299  |    24\.5G     |             ImageNet Train              |      ImageNet Val       |                       0\.7958/0\.9470                        |                       0\.7899/0\.9444                        |
|  7   |     Image Classification      |         mobileNet\_v2          | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G\_1\.3         |   caffe    |   MobileNet\_v2   |  224\*224  |     608M      |             ImageNet Train              |      ImageNet Val       |                       0\.6527/0\.8643                        |                       0\.6399/0\.8540                        |
|  8   |      Image Classifiction      |           SqueezeNet           | cf\_squeeze\_imagenet\_227\_227\_0\.76G\_1\.3             |   caffe    |    squeezenet     |  227\*227  |    0\.76G     |             ImageNet Train              |      ImageNet Val       |                       0\.5432/0\.7824                        |                       0\.5270/0\.7816                        |
|  9   |   ADAS Pedstrain Detection    | ssd\_pedestrian\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G\_1\.3     |   caffe    |    VGG\-bn\-16    |  360\*640  |     5\.9G     | coco2014\_train\_person and crowndhuman |  coco2014\_val\_person  |                           0\.5903                            |                           0\.5857                            |
|  10  |       Object Detection        |      refinedet\_baseline       | cf_refinedet_coco_360_480_123G_1.3                        |   caffe    |    VGG\-bn\-16    |  360\*480  |     123G      |         coco2014\_train\_person         |  coco2014\_val\_person  |                            0.6928                            |                            0.7042                            |
|  11  |       Object Detection        |    refinedet\_pruned\_0\.8     | cf\_refinedet\_coco\_360\_480\_0\.8\_25G\_1\.3            |   caffe    |    VGG\-bn\-16    |  360\*480  |      25G      |         coco2014\_train\_person         |  coco2014\_val\_person  |                           0\.6794                            |                           0\.6790                            |
|  12  |       Object Detection        |    refinedet\_pruned\_0\.92    | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G\_1\.3       |   caffe    |    VGG\-bn\-16    |  360\*480  |    10\.10G    |         coco2014\_train\_person         |  coco2014\_val\_person  |                           0\.6489                            |                           0\.6487                            |
|  13  |       Object Detection        |    refinedet\_pruned\_0\.96    | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G\_1\.3        |   caffe    |    VGG\-bn\-16    |  360\*480  |    5\.08G     |         coco2014\_train\_person         |  coco2014\_val\_person  |                           0\.6120                            |                           0\.6112                            |
|  14  |    ADAS Vehicle Detection     |    ssd\_adas\_pruned\_0\.95    | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G\_1\.3            |   caffe    |      VGG\-16      |  360\*480  |     6\.3G     |         bdd100k \+ private data         | bdd100k \+ private data |                           0\.4207                            |                           0\.4200                            |
|  15  |       Traffic Detection       |   ssd\_traffic\_pruned\_0\.9   | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G\_1\.3              |   caffe    |      VGG\-16      |  360\*480  |    11\.6G     |              private data               |      private data       |                           0\.5982                            |                           0\.5921                            |
|  16  |      ADAS Lane Detection      |     VPGnet\_pruned\_0\.99      | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G\_1\.3     |   caffe    |        VGG        |  480\*640  |     2\.5G     |              caltech lanes              |      caltech lanes      |                    0\.8864 \(F1\-score\)                     |                    0\.8882 \(F1\-score\)                     |
|  17  |       Object Detection        |       ssd\_mobilnet\_v2        | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G\_1\.3           |   caffe    |   MobileNet\_v2   |  360\*480  |    6\.57G     |              bdd100k train              |       bdd100k val       |                           0\.3052                            |                           0\.2752                            |
|  18  |       ADAS Segmentation       |              FPN               | cf\_fpn\_cityscapes\_256\_512\_8\.9G\_1\.3                |   caffe    |  Google\_v1\_BN   |  256\*512  |     8\.9G     |         Cityscapes gtFineTrain          |     Cityscapes Val      |                           0\.5669                            |                           0\.5607                            |
|  19  |        Pose Estimation        |            SP\-net             | cf\_SPnet\_aichallenger\_224\_128\_0\.54G\_1\.3           |   caffe    |  Google\_v1\_BN   |  224\*128  |    548\.6M    |             ai\_challenger              |     ai\_challenger      |                     0\.9000 \(PCKh0\.5\)                     |                     0\.8964 \(PCKh0\.5\)                     |
|  20  |        Pose Estimation        |     Openpose\_pruned\_0\.3     | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G\_1\.3 |   caffe    |        VGG        |  368\*368  |    189.7G     |             ai\_challenger              |     ai\_challenger      |                       0\.4507 \(OKs\)                        |                       0\.4429 \(Oks\)                        |
|  21  |        Face Detection         |       densebox\_320\_320       | cf\_densebox\_wider\_320\_320\_0\.49G\_1\.3               |   caffe    |      VGG\-16      |  320\*320  |    0\.49G     |               wider\_face               |          FDDB           |                           0\.8833                            |                           0\.8783                            |
|  22  |        Face Detection         |       densebox\_360\_640       | cf\_densebox\_wider\_360\_640\_1\.11G\_1\.3               |   caffe    |      VGG\-16      |  360\*640  |    1\.11G     |               wider\_face               |          FDDB           |                           0\.8931                            |                           0\.8922                            |
|  23  |       Face Recognition        |         face\_landmark         | cf\_landmark\_celeba\_96\_72\_0\.14G\_1\.3                |   caffe    |       lenet       |   96\*72   |    0\.14G     |                 celebA                  |     processed helen     |                      0\.1952 (L2 loss\)                      |                      0\.1971 (L2 loss\)                      |
|  24  |      Re\-identification       |              reid              | cf\_reid\_market1501\_160\_80\_0\.95G\_1\.3               |   caffe    |     resnet18      |  160\*80   |    0\.95G     |           Market1501\+CUHK03            |       Market1501        |                mAP:0\.5660     Rank-1:0\.7800                |                mAP:0\.5590     Rank-1:0\.7760                |
|  25  |    Detection+Segmentation     |           multi-task           | cf\_multitask\_bdd\_288\_512\_14\.8G\_1\.3                |   caffe    |        ssd        |  288\*512  |    14\.8G     |           BDD100K+Cityscapes            |   BDD100K+Cityscapes    |                   mAP:0\.2228 mIOU:0\.4088                   |                   mAP:0\.2140 mIOU:0\.4058                   |
|  26  |       Object Detection        |          yolov3\_bdd           | dk\_yolov3\_bdd\_288\_512\_53\.7G\_1\.3                   |  darknet   |    darknet\-53    |  288\*512  |    53\.7G     |                 bdd100k                 |         bdd100k         |                           0\.5058                            |                           0\.4910                            |
|  27  |       Object Detection        |   yolov3\_adas\_prune\_0\.9    | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G\_1\.3      |  darknet   |    darknet\-53    |  256\*512  |    5\.46G     |               Cityscapes                |       Cityscapes        |                           0\.5520                            |                           0\.5300                            |
|  28  |       Object Detection        |          yolov3\_voc           | dk\_yolov3\_voc\_416\_416\_65\.42G\_1\.3                  |  darknet   |    darknet\-53    |  416\*416  |    65\.42G    |           voc07\+12\_trainval           |       voc07\_test       |                           0\.8240                            |                           0\.8130                            |
|  29  |       Object Detection        |          yolov2\_voc           | dk\_yolov2\_voc\_448\_448\_34G\_1\.3                      |  darknet   |    darknet\-19    |  448\*448  |      34G      |           voc07\+12\_trainval           |       voc07\_test       |                           0\.7845                            |                           0\.7740                            |
|  30  |       Object Detection        |   yolov2\_voc\_pruned\_0\.66   | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G\_1\.3           |  darknet   |    darknet\-19    |  448\*448  |    11\.56G    |           voc07\+12\_trainval           |       voc07\_test       |                           0\.7700                            |                           0\.7610                            |
|  31  |       Object Detection        |   yolov2\_voc\_pruned\_0\.71   | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G\_1\.3            |  darknet   |    darknet\-19    |  448\*448  |    9\.86G     |           voc07\+12\_trainval           |       voc07\_test       |                           0\.7670                            |                           0\.7540                            |
|  32  |       Object Detection        |   yolov2\_voc\_pruned\_0\.77   | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G\_1\.3            |  darknet   |    darknet\-19    |  448\*448  |    7\.82G     |           voc07\+12\_trainval           |       voc07\_test       |                           0\.7576                            |                           0\.7482                            |
|  33  |       Face Recognition        |         ResNet20-face          | cf_facerec-resnet20_112_96_3.5G_1.3                       |   caffe    |     resnet20      |   112*96   |     3.5G      |              private data               |      private data       |                        0.9610 (1e-06)                        |                        0.9480 (1e-06)                        |
|  34  |       Face Recognition        |         ResNet64-face          | cf_facerec-resnet64_112_96_11G_1.3                        |   caffe    |     resnet64      |   112*96   |      11G      |              private data               |      private data       |                        0.9830 (1e-06)                        |                        0.9820 (1e-06)                        |
|  35  |     Medical Segmentation      | FPN_Res18                      | cf_FPN-resnet18_EDD_320_320_45.3G_1.3                     |   caffe    |     resnet18      |  320*320   |    45.30G     |                 EDD_seg                 |         EDD_seg         | mean dice=0.8203                mean jaccard=0.7925                                                                       F2-score=0.8075 | mean dice=0.8055                     mean jaccard=0.7772                                                                              F2- score=0.7925 |
|  36  |        Plate Detection        |        plate_detection         | cf_plate-detection_320_320_0.49G_1.3                      |   caffe    |    modify_vgg     |  320*320   |     0.49G     |              private data               |      private data       |                       Recall1: 0.9660                        |                       Recall1: 0.9650                        |
|  37  |       Plate Recognition       |       plate_recognition        | cf_plate-recognition_96_288_1.75G_1.3                     |   caffe    |    Google\_v1     |   96*288   |     1.75G     |              private data               |      private data       |             plate number:99.51% plate color:100%             |             plate number:99.51% plate color:100%             |
|  38  |        Face Detection         |           retinaface           | cf_retinaface_wider_360_640_1.11G_1.3                     |   caffe    | MobileNet_v1 0.25 |  360*640   |     1.11G     |            wideface and FDDB            |   widerface and FDDB    |                        0.9140@fp=100                         |                        0.8940@fp=100                         |
|  39  |         Face Quality          |          face_quality          | cf_face-quality_80_60_61.68M_1.3                          |   caffe    |       lenet       |   80*60    |    61.68M     |              private data               |      private data       |                      12.5481 (L1 loss)                       |                      12.6863 (L1 loss)                       |
|  40  | Robot Instrument Segmentation |     FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G_1.3                  |   caffe    |     resnet18      |  240*320   |    13.75G     |               EndoVis'15                |       EndoVis'15        |                  Dice=0.8050 Jaccard=0.7270                  |                  Dice=0.8010 Jaccard=0.7230                  |
|  41  |        Pose Estimation        |           Hourglass            | cf_hourglass-pe_mpii_256_256_10.2G_1.3                    |   caffe    |      resnet       |  256*256   |    10.20G     |             MPII Human Pose             |     MPII Human Pose     |                     0.8718 \(PCKh0\.5\)                      |                     0.8661 \(PCKh0\.5\)                      |
|  42  |         2D Detection          |          tiny-yolov3           | dk_tiny-yolov3_416_416_5.46G_1.3                          |  darknet   |    darknet-19     |  416*416   |     5.46G     |              private data               |      private data       |                            0.9739                            |                            0.9650                            |
|  43  |         2D Detection          |             yolov4             | dk_yolov4_coco_416_416_60.1G_1.3                          |  darknet   |      yolov4       |  416*416   |    60.10G     |                  coco                   |       coco2014-5k       |                            0.3950                            |                            0.3730                            |
|  44  |      Image Classifiction      |     Inception\_resnet\_v2      | tf\_inceptionresnetv2\_imagenet\_299\_299\_26\.35G\_1\.3  | tensorflow |     inception     |  299\*299  |    26\.35G    |             ImageNet Train              |      ImageNet Val       |                           0\.8037                            |                           0\.7946                            |
|  45  |      Image Classifiction      |         Inception\_v1          | tf\_inceptionv1\_imagenet\_224\_224\_3G\_1\.3             | tensorflow |     inception     |  224\*224  |      3G       |             ImageNet Train              |      ImageNet Val       |                           0\.6976                            |                           0\.6794                            |
|  46  |      Image Classifiction      |         Inception\_v3          | tf\_inceptionv3\_imagenet\_299\_299\_11\.45G\_1\.3        | tensorflow |     inception     |  299\*299  |    11\.45G    |             ImageNet Train              |      ImageNet Val       |                           0\.7798                            |                           0\.7607                            |
|  47  |      Image Classifiction      |         Inception\_v4          | tf\_inceptionv4\_imagenet\_299\_299\_24\.55G\_1\.3        | tensorflow |     inception     |  299\*299  |    24\.55G    |             ImageNet Train              |      ImageNet Val       |                           0\.8018                            |                           0\.7928                            |
|  48  |      Image Classifiction      |         Mobilenet\_v1          | tf\_mobilenetv1\_0\.25\_imagenet\_128\_128\_27M\_1\.3     | tensorflow |     mobilenet     |  128\*128  |    27\.15M    |             ImageNet Train              |      ImageNet Val       |                           0\.4144                            |                           0\.3464                            |
|  49  |      Image Classifiction      |         Mobilenet\_v1          | tf\_mobilenetv1\_0\.5\_imagenet\_160\_160\_150M\_1\.3     | tensorflow |     mobilenet     |  160\*160  |   150\.07M    |             ImageNet Train              |      ImageNet Val       |                           0\.5903                            |                           0\.5195                            |
|  50  |      Image Classifiction      |         Mobilenet\_v1          | tf\_mobilenetv1\_1\.0\_imagenet\_224\_224\_1\.14G\_1\.3   | tensorflow |     mobilenet     |  224\*224  |    1\.14G     |             ImageNet Train              |      ImageNet Val       |                           0\.7102                            |                           0\.6779                            |
|  51  |      Image Classifiction      |         Mobilenet\_v2          | tf\_mobilenetv2\_1\.0\_imagenet\_224\_224\_602M_1\.3      | tensorflow |     mobilenet     |  224\*224  |    0\.59G     |             ImageNet Train              |      ImageNet Val       |                           0\.7013                            |                           0\.6767                            |
|  52  |      Image Classifiction      |         Mobilenet\_v2          | tf\_mobilenetv2\_1\.4\_imagenet\_224\_224\_1\.16G\_1\.3   | tensorflow |     mobilenet     |  224\*224  |    1\.16G     |             ImageNet Train              |      ImageNet Val       |                           0\.7411                            |                           0\.7194                            |
|  53  |      Image Classifiction      |         resnet\_v1\_50         | tf\_resnetv1\_50\_imagenet\_224\_224\_6\.97G\_1\.3        | tensorflow |     resnetv1      |  224\*224  |    6\.97G     |             ImageNet Train              |      ImageNet Val       |                           0\.7520                            |                           0\.7478                            |
|  54  |      Image Classifiction      |        resnet\_v1\_101         | tf\_resnetv1\_101\_imagenet\_224\_224\_14\.4G\_1\.3       | tensorflow |     resnetv1      |  224\*224  |    14\.4G     |             ImageNet Train              |      ImageNet Val       |                           0\.7640                            |                           0\.7560                            |
|  55  |      Image Classifiction      |        resnet\_v1\_152         | tf\_resnetv1\_152\_imagenet\_224\_224\_21\.83G\_1\.3      | tensorflow |     resnetv1      |  224\*224  |    21\.83G    |             ImageNet Train              |      ImageNet Val       |                           0\.7681                            |                           0\.7463                            |
|  56  |      Image Classifiction      |            vgg\_16             | tf\_vgg16\_imagenet\_224\_224\_30\.96G\_1\.3              | tensorflow |        vgg        |  224\*224  |    30\.96G    |             ImageNet Train              |      ImageNet Val       |                           0\.7089                            |                           0\.7069                            |
|  57  |      Image Classifiction      |            vgg\_19             | tf\_vgg19\_imagenet\_224\_224\_39\.28G\_1\.3              | tensorflow |        vgg        |  224\*224  |    39\.28G    |             ImageNet Train              |      ImageNet Val       |                           0\.7100                            |                           0\.7026                            |
|  58  |       Object Detection        |       ssd\_mobilenet\_v1       | tf\_ssdmobilenetv1\_coco\_300\_300\_2\.47G\_1\.3          | tensorflow |     mobilenet     |  300\*300  |    2\.47G     |                coco2017                 |    coco2014 minival     |                           0\.2080                            |                           0\.2100                            |
|  59  |       Object Detection        |       ssd\_mobilenet\_v2       | tf\_ssdmobilenetv2\_coco\_300\_300\_3\.75G\_1\.3          | tensorflow |     mobilenet     |  300\*300  |    3\.75G     |                coco2017                 |    coco2014 minival     |                           0\.2150                            |                           0\.2110                            |
|  60  |       Object Detection        |     ssd\_resnet50\_v1\_fpn     | tf\_ssdresnet50v1\_fpn\_coco\_640\_640\_178\.4G\_1\.3     | tensorflow |     resnet50      |  300\*300  |    178\.4G    |                coco2017                 |    coco2014 minival     |                           0\.3010                            |                           0\.2900                            |
|  61  |       Object Detection        |          yolov3\_voc           | tf\_yolov3\_voc\_416\_416\_65\.63G\_1\.3                  | tensorflow |    darknet\-53    |  416\*416  |    65\.63G    |           voc07\+12\_trainval           |       voc07\_test       |                           0\.7846                            |                           0\.7744                            |
|  62  |       Object Detection        |     mlperf\_ssd\_resnet34      | tf\_mlperf_resnet34\_coco\_1200\_1200\_433G\_1\.3         | tensorflow |     resnet34      | 1200\*1200 |     433G      |                coco2017                 |        coco2017         |                           0\.2250                            |                           0\.2150                            |
|  63  |      Image Classifiction      |          Inception_v2          | tf_inceptionv2_imagenet_224_224_3.88G_1.3                 | tensorflow |     inception     |  224*224   |     3.88G     |             ImageNet Train              |      ImageNet Val       |                            0.7399                            |                            07331                             |
|  64  |      Image Classifiction      |         resnet\_v2\_50         | tf_resnetv2_50_imagenet_299_299_13.1G_1.3                 | tensorflow |     resnetv2      |  299*299   |     13.1G     |             ImageNet Train              |      ImageNet Val       |                            0.7559                            |                            0.7445                            |
|  65  |      Image Classifiction      |        resnet\_v2\_101         | tf_resnetv2_101_imagenet_299_299_26.78G_1.3               | tensorflow |     resnetv2      |  299*299   |    26.78G     |             ImageNet Train              |      ImageNet Val       |                            0.7695                            |                            0.7506                            |
|  66  |      Image Classifiction      |        resnet\_v2\_152         | tf_resnetv2_152_imagenet_299_299_40.47G_1.3               | tensorflow |     resnetv2      |  299*299   |    40.47G     |             ImageNet Train              |      ImageNet Val       |                            0.7779                            |                            0.7432                            |
|  67  |       Object Detection        |      ssdlite_mobilenetv2       | tf_ssdlite_mobilenetv2_coco_300_300_1.5G_1.3              | tensorflow |     mobilenet     |  300*300   |     1.5G      |                coco2017                 |    coco2014 minival     |                            0.2170                            |                            0.2090                            |
|  68  |       Object Detection        |        ssd_inceptionv2         | tf_ssdinceptionv2_coco_300_300_9.62G_1.3                  | tensorflow |     inception     |  300*300   |     9.62G     |                coco2017                 |    coco2014 minival     |                            0.2390                            |                            0.2360                            |
|  69  |         Segmentation          |         Mobilenet\_v2          | tf_mobilenetv2_cityscapes_1024_2048_132.74G_1.3           | tensorflow |     mobilenet     | 1024*2048  |    132.74G    |               Cityscapes                |       Cityscapes        |                            0.6263                            |                            0.4578                            |
|  70  |      Image Classifiction      |        mlperf_resnet50         | tf_mlperf_resnet50_imagenet_224_224_8.19G_1.3             | tensorflow |     resnet50      |  224*224   |     8.19G     |             ImageNet Train              |      ImageNet Val       |                            0.7652                            |                            0.7606                            |
|  71  |       General Detection       |           refinedet            | tf_refinedet_VOC_320_320_81.9G_1.3                        | tensorflow |      vgg_bn       |  320*320   |     81.9G     |           voc07\+12\_trainval           |       voc07\_test       |                            0.8015                            |                            0.7999                            |
|  72  |      Image Classifiction      |       mobilenet_edge_1.0       | tf_mobilenetEdge1.0_imagenet_224_224_990M_1.3             | tensorflow | mobilenetEdge1.0  |  224*224   |     990M      |             ImageNet Train              |      ImageNet Val       |                            0.7227                            |                            0.6775                            |
|  73  |      Image Classifiction      |      mobilenet_edge_0.75       | tf_mobilenetEdge0.75_imagenet_224_224_624M_1.3            | tensorflow | mobilenetEdge0.75 |  224*224   |     624M      |             ImageNet Train              |      ImageNet Val       |                            0.7201                            |                            0.6489                            |
|  74  |       Medical Detection       |       refinedet_medical        | tf_RefineDet-Medical_EDD_320_320_9.83G_1.3                | tensorflow |   pruned_vgg16    |  320*320   |     9.83G     |                   EDD                   |           EDD           |                            0.7839                            |                            0.8014                            |
|  75  |      Image Classifiction      |            resnet50            | tf2_resnet50_imagenet_224_224_7.76G_1.3                   | tensorflow |     resnet50      |  224*224   |     7.76G     |             ImageNet Train              |      ImageNet Val       |                            0.7513                            |                            0.7423                            |
|  76  |      Image Classifiction      |         Mobilenet\_v1          | tf2_mobilenetv1_imagenet_224_224_1.15G_1.3                | tensorflow |     mobilenet     |  224*224   |     1.15G     |             ImageNet Train              |      ImageNet Val       |                            0.7005                            |                            0.5603                            |
|  77  |      Image Classifiction      |          Inception_v3          | tf2_inceptionv3_imagenet_299_299_11.5G_1.3                | tensorflow |     inception     |  299*299   |     11.5G     |             ImageNet Train              |      ImageNet Val       |                            0.7753                            |                            0.7694                            |
|  78  |     Medical Segmentation      |            2d-unet             | tf2_2d-unet_nuclei_128_128_5.31G_1.3                      | tensorflow |       unet        |  128*128   |     5.31G     |               Nuclei Cell               |       Nuclei Cell       |                            0.3968                            |                            0.3968                            |
|  79  |     Semantic Segmentation     |             ERFNet             | tf2_erfnet_cityscapes_512_1024_54G_1.3                    | tensorflow |      ERFNet       |  512*1024  |      54G      |               Cityscapes                |       Cityscapes        |                            0.5298                            |                            0.5167                            |
|  80  |         Segmentation          |              ENet              | pt_ENet_cityscapes_512_1024_8.6G_1.3                      |  pytorch   |       ENet        |  512*1024  |     8.6G      |               Cityscapes                |       Cityscapes        |                            0.6442                            |                            0.6315                            |
|  81  |         Segmentation          |          SemanticFPN           | pt_SemanticFPN_cityscapes_256_512_10G_1.3                 |  pytorch   |   FPN-Resnet18    |  256*512   |      10G      |               Cityscapes                |       Cityscapes        |                            0.6290                            |                            0.6230                            |
|  82  |       Face Recognition        |         ResNet20-face          | pt_facerec-resnet20_mixed_112_96_3.5G_1.3                 |  pytorch   |     resnet20      |   112*96   |     3.5G      |                  mixed                  |          mixed          |                            0.9955                            |                            0.9947                            |
|  83  |         Face Quality          |          face quality          | pt_face-quality_80_60_61.68M_1.3                          |  pytorch   |       lenet       |   80*60    |    61.68M     |              private data               |      private data       |                            0.1233                            |                            0.1258                            |
|  84  |          Multi Task           |          MT-resnet18           | pt_MT-resnet18_mixed_320_512_13.65G_1.3                   |  pytorch   |     resnet18      |  320*512   |    13.65G     |                  mixed                  |          mixed          |                   mAP:0.3951  mIOU:0.4403                    |                   mAP:0.3841  mIOU:0.4271                    |
|  85  |           Face ReID           |        face_reid_large         | pt_facereid-large_96_96_515M_1.3                          |  pytorch   |     resnet18      |   96*96    |     515M      |              private data               |      private data       |                   mAP:0.7940  Rank1:0.9550                   |                  mAP:0.7900   Rank1:0.9530                   |
|  86  |           Face ReID           |        face_reid_small         | pt_facereid-small_80_80_90M_1.3                           |  pytorch   |   resnet_small    |   80*80    |      90M      |              private data               |      private data       |                  mAP:0.5600   Rank1:0.8650                   |                  mAP:0.5590   Rank1:0.8650                   |
|  87  |      Re\-identification       |          person_reid           | pt_personreid-res50_market1501_256_128_5.4G_1.3           |  pytorch   |     resnet50      |  256*128   |     4.2G      |               market1501                |       market1501        |                  mAP:0.8540   Rank1:0.9410                   |                  mAP:0.8380   Rank1:0.9370                   |
|  88  |      Re\-identification       |          person_reid           | pt_personreid-res18_market1501_176_80_1.1G_1.3            |  pytorch   |     resnet18      |   176*80   |     1.1G      |               market1501                |       market1501        |                  mAP:0.7530   Rank1:0.8980                   |                  mAP:0.7460   Rank1:0.8930                   |
|  89  |         3D Detection          |          pointpillars          | pt_pointpillars_kitti_12000_100_10.8G_1.3                 |  pytorch   |     PointNet      | 12000*100  |     10.8G     |                  kitti                  |          kitti          |   Car 3D AP@0.5(easy, moderate, hard) 90.79, 89.66, 88.78    |   Car 3D AP@0.5(easy, moderate, hard) 90.75, 87.04, 83.44    |
|  90  |  3D point cloud segmentation  |           salsanext            | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G_1.3         |  pytorch   |  pruned resnet50  |  64*2048   |     20.4G     |             semantic-kitti              |     semantic-kitti      |                Acc avg 0.8860 IoU avg 0.5100                 |                Acc avg 0.8350 IoU avg 0.4540                 |
|  91  |     Covid-19 segmentation     |     FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G_1.3             |  pytorch   |     resnet18      |  352*352   |     22.7G     |               COVID19-seg               |       COVID19-seg       |       2-classes Dice:0.8588     3-classes mIoU:0.5989        |       2-classes Dice:0.8547     3-classes mIoU:0.5957        |
|  92  |     CT lung segmentation      |            2d-unet             | pt_unet_chaos-CT_512_512_23.3G_1.3                        |  pytorch   |       unet        |  512*512   |     23.3G     |                Chaos-CT                 |        Chaos-CT         |                         Dice:0.9758                          |                         Dice:0.9747                          |

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


For example, `cf_refinedet_coco_360_480_0.8_25G_1.3` is a `RefineDet` model trained with `Caffe` using `COCO` dataset, input data size is `360*480`, `80%` pruned, the computation per image is `25Gops` and Vitis AI version is `1.3`.


### caffe-xilinx 
This is a custom distribution of caffe. Please use [caffe-xilinx](/models/AI-Model-Zoo/caffe-xilinx) to test and finetune the caffe models listed in this page.

## Model Download
Please visit [model-list](/models/AI-Model-Zoo/model-list) in this page. You will get downloadlink and MD5 of all the released models, including pre-compiled models running on different platforms.                                 

</details>

### Model Directory Structure
Download and extract the model archive to your working area on the local hard disk. For details on the various models, their download link and MD5 checksum for the zip file of each model, see [model-list](/models/AI-Model-Zoo/model-list).


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
    │    ├── quantize_train_test.prototxt     # Used for quantized-point training and testing with quantize_train_test.caffemodel 
    │    │                                      on GPU when datalayer modified to user's data path.         
    │    └── DNNC
    │         └──deploy.prototxt              # Quantized prototxt for dnnc. It's the deploy.prototxt without option 'keep fixed neuron'.
    │                                           
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
    │   ├── deploy.model.pb             # Quantized model for the compiler (extended Tensorflow format).
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
    ├── quantized                          
    │   ├── bias_corr.pth               # Quantized model.
    │   ├── quant_info.json             # Quantization steps of tensors got. Please keep it for evaluation of quantized model.
    │   ├── _int.py                     # Converted vai_q_pytorch format model.
    │   └── _int.xmodel                 # Deployed model. The name of different models may be different.
    │
    │
    └── float                           
        └── _int.pth                    # Trained float-point model. The pth name of different models may be different.
                                          Path and name in test scripts could be modified according to actual situation.
        
                                          
                                          
**Note:** For more information on `vai_q_caffe` and `vai_q_tensorflow`, see the [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_3/ug1414-vitis-ai.pdf).



## Model Performance
All the models in the Model Zoo have been deployed on Xilinx hardware with [Vitis AI](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library](https://github.com/Xilinx/Vitis-AI/tree/master/tools/Vitis-AI-Library). The performance number including end-to-end throughput and latency for each model on various boards with different DPU configurations are listed in the following sections.

For more information about DPU, see [DPU IP Product Guide](https://www.xilinx.com/cgi-bin/docs/ipdoc?c=dpu;v=latest;d=pg338-dpu.pdf).


**Note:** The model performance number listed in the following sections is generated with Vitis AI v1.3 and Vitis AI Lirary v1.3. For different boards, different DPU configurations are used. Vitis AI and Vitis AI Library can be downloaded for free from [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library Github](https://github.com/Xilinx/Vitis-AI/tree/master/tools/Vitis-AI-Library).
We will continue to improve the performance with Vitis AI. The performance number reported here is subject to change in the near future.

### Performance on ZCU102 (0432055-04)  
This version of ZCU102 is out of stock. The performance number shown below was measured with the previous AI SDK v2.0.4. Now this form has stopped updating.

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


### Performance on ZCU102 (0432055-05)
Measured with Vitis AI 1.3 and Vitis AI Library 1.3  

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-05)` board with a `3 * B4096  @ 281MHz   V1.4.1` DPU configuration:


| No\. | Model                      | Name                                          | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread) |
| ---- | :------------------------- | :-------------------------------------------- | -------------------------------- | ------------------------------------- | ----------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G             | 14.1                             | 70.8                                  | 164.7                               |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G            | 5.55                             | 180.2                                 | 469.1                               |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G         | 6.12                             | 163.2                                 | 428.1                               |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G            | 8.13                             | 122.9                                 | 286.7                               |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G         | 17.47                            | 57.2                                  | 132.3                               |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G         | 35.26                            | 28.3                                  | 68.1                                |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G         | 4.08                             | 245.1                                 | 712                                 |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G             | 3.9                              | 256.1                                 | 1035.4                              |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G       | 13.37                            | 74.7                                  | 275                                 |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                | 118.09                           | 8.5                                   | 24.4                                |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G             | 31.15                            | 32.1                                  | 98.7                                |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G         | 16.27                            | 61.4                                  | 194                                 |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G          | 11.78                            | 84.9                                  | 272.4                               |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G              | 11.98                            | 83.4                                  | 285.5                               |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G               | 18.76                            | 53.3                                  | 195                                 |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G       | 11.75                            | 85                                    | 250.4                               |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G           | 25.7                             | 38.9                                  | 109.2                               |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                | 30.62                            | 32.6                                  | 144.7                               |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G           | 2.24                             | 445.4                                 | 1227.1                              |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G   | 287.88                           | 3.5                                   | 14.9                                |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G               | 2.73                             | 365.9                                 | 1126.9                              |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G               | 5.58                             | 179.2                                 | 503.9                               |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                | 1.21                             | 823.9                                 | 1553.6                              |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G               | 2.88                             | 346.6                                 | 690.1                               |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                | 28.93                            | 34.5                                  | 117.1                               |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                   | 82.2                             | 12.1                                  | 32.3                                |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G        | 11.73                            | 85.2                                  | 251.3                               |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                  | 80.13                            | 12.5                                  | 32.6                                |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                     | 38.94                            | 25.7                                  | 68                                  |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G             | 16                               | 62.5                                  | 189                                 |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G              | 13.9                             | 71.9                                  | 221                                 |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G              | 11.92                            | 83.9                                  | 265.2                               |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G               | 6.11                             | 163.5                                 | 329.1                               |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                | 14.19                            | 70.4                                  | 176.4                               |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G             | 85.83                            | 11.6                                  | 42.1                                |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G              | 2.02                             | 493.8                                 | 1996                                |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G             | 8.14                             | 122.8                                 | 239.9                               |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G             | 8.01                             | 124.8                                 | 569.2                               |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                  | 0.5                              | 2000.4                                | 6721.9                              |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G          | 30.26                            | 33                                    | 152.4                               |
| 41   | Hourglass                  | cf_hourglass-pe_mpii_256_256_10.2G            | 20.69                            | 48.3                                  | 113.4                               |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                  | 8.65                             | 115.5                                 | 383.6                               |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                  | 77.69                            | 12.8                                  | 32.8                                |
| 44   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G  | 45.64                            | 21.9                                  | 49.2                                |
| 45   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G            | 6                                | 166.7                                 | 428.6                               |
| 46   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G        | 17.6                             | 56.8                                  | 130.8                               |
| 47   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G        | 35.27                            | 28.3                                  | 68                                  |
| 48   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M   | 0.94                             | 1062.6                                | 3991.2                              |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M   | 1.45                             | 689.5                                 | 2624.4                              |
| 50   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G     | 3.48                             | 287.4                                 | 892.8                               |
| 51   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G     | 4.15                             | 240.7                                 | 675.7                               |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G     | 5.67                             | 176.2                                 | 455.5                               |
| 53   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G         | 12.89                            | 77.5                                  | 178                                 |
| 54   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G        | 23.95                            | 41.7                                  | 103.4                               |
| 55   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G       | 34.89                            | 28.6                                  | 72.1                                |
| 56   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G              | 50.06                            | 19.9                                  | 40.5                                |
| 57   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G              | 58.01                            | 17.2                                  | 36                                  |
| 58   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G          | 10.51                            | 95.1                                  | 305.5                               |
| 59   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G          | 12.69                            | 78.7                                  | 206.2                               |
| 60   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G      | 351.33                           | 2.8                                   | 5                                   |
| 61   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                  | 77.44                            | 12.9                                  | 33.4                                |
| 62   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G        | 530.72                           | 1.9                                   | 6.9                                 |
| 63   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G         | 11.51                            | 86.8                                  | 220.8                               |
| 64   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G_1.3     | 29.38                            | 34                                    | 83.3                                |
| 65   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G       | 50.45                            | 19.8                                  | 48.5                                |
| 66   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G       | 71.38                            | 14                                    | 33.7                                |
| 67   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G      | 10                               | 99.9                                  | 292.8                               |
| 68   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G_1.3      | 26.4                             | 37.8                                  | 97.5                                |
| 69   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G   | 636.19                           | 1.6                                   | 4.4                                 |
| 70   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G     | 14.34                            | 69.6                                  | 160.6                               |
| 71   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                | 90.67                            | 11                                    | 33.6                                |
| 72   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M_1.3 | 5.17                             | 193.4                                 | 525.2                               |
| 73   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M    | 4.31                             | 231.6                                 | 675.8                               |
| 74   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G        | 15.51                            | 64.4                                  | 220.3                               |
| 75   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G           | 14.52                            | 68.8                                  | 165.2                               |
| 76   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G        | 3.52                             | 284.1                                 | 879.6                               |
| 77   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G        | 17.94                            | 55.7                                  | 130.4                               |
| 78   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G              | 6.6                              | 151.4                                 | 381.2                               |
| 79   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G            | 149.67                           | 6.7                                   | 23                                  |
| 80   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G              | 113.44                           | 8.8                                   | 37.9                                |
| 81   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G         | 30.38                            | 32.9                                  | 160.1                               |
| 82   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G         | 6.12                             | 163.2                                 | 328.2                               |
| 83   | face quality               | pt_face-quality_80_60_61.68M                  | 0.47                             | 2140.1                                | 6887.7                              |
| 84   | MT-resnet18                | pt_MT-resnet18_mixed_320_512_13.65G           | NA                               | NA                                    | NA                                  |
| 85   | face_reid_large            | pt_facereid-large_96_96_515M                  | 1.19                             | 837.2                                 | 2164.7                              |
| 86   | face_reid_small            | pt_facereid-small_80_80_90M                   | 0.54                             | 1833.4                                | 5948.5                              |
| 87   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G   | 10.36                            | 96.4                                  | 223.5                               |
| 88   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G    | 2.86                             | 349.2                                 | 681.6                               |
| 89   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G         | 53.7                             | 18.6                                  | 49.8                                |
| 90   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G | 188.48                           | 5.3                                   | 19.7                                |
| 91   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G     | 28.32                            | 35.3                                  | 104.1                               |
| 92   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                | 48.35                            | 20.7                                  | 64.8                                |
| 93   | Inception_v3               | torchvision_inception_v3                      | 17.42                            | 57.4                                  | 132.6                               |
| 94   | SqueezeNet                 | torchvision_squeezenet                        | 4.87                             | 205.2                                 | 766.9                               |
| 95   | resnet50                   | torchvision_resnet50                          | 14.92                            | 67                                    | 158.6                               |

</details>


### Performance on ZCU104
Measured with Vitis AI 1.3 and Vitis AI Library 1.3 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 300MHz   V1.4.1` DPU configuration:


| No\. | Model                      | Name                                          | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread) |
| ---- | :------------------------- | :-------------------------------------------- | -------------------------------- | ------------------------------------- | ----------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G             | 13.29                            | 75.1                                  | 144.7                               |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G            | 5.27                             | 189.7                                 | 409.7                               |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G         | 5.81                             | 172                                   | 368                                 |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G            | 7.67                             | 130.2                                 | 260.7                               |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G         | 16.46                            | 60.7                                  | 119                                 |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G         | 33.09                            | 30.2                                  | 58.7                                |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G         | 3.92                             | 255.2                                 | 589.3                               |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G             | 3.84                             | 260.2                                 | 925.7                               |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G       | 12.87                            | 77.6                                  | 209.5                               |
| 10   | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                | 110.92                           | 9                                     | 18.3                                |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G             | 29.46                            | 33.9                                  | 73.4                                |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G         | 15.5                             | 64.5                                  | 149.7                               |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G          | 11.26                            | 88.7                                  | 217.5                               |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G              | 11.49                            | 87                                    | 227.7                               |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G               | 17.97                            | 55.6                                  | 145.3                               |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G       | 11.36                            | 88                                    | 242.5                               |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G           | 39.53                            | 25.3                                  | 96.3                                |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                | 29.91                            | 33.4                                  | 126.8                               |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G           | 2.16                             | 462.8                                 | 708.4                               |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G   | 274.42                           | 3.6                                   | 10.9                                |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G               | 2.67                             | 374.4                                 | 1132.2                              |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G               | 5.4                              | 185                                   | 535.5                               |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G                | 1.15                             | 866.6                                 | 1593.6                              |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G               | 2.72                             | 367.4                                 | 696.6                               |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G                | 27.89                            | 35.8                                  | 104.7                               |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                   | 77.18                            | 12.9                                  | 26.3                                |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G        | 11.27                            | 88.7                                  | 224.8                               |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                  | 75.25                            | 13.3                                  | 26.6                                |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                     | 36.7                             | 27.2                                  | 56.1                                |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G             | 15.2                             | 65.8                                  | 153.6                               |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G              | 13.23                            | 75.6                                  | 181.1                               |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G              | 11.4                             | 87.7                                  | 217.5                               |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G               | 5.74                             | 174.2                                 | 308.3                               |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                | 13.32                            | 75                                    | 144.7                               |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G             | 81.71                            | 12.2                                  | 31.5                                |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G              | 2                                | 500                                   | 1862.4                              |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G             | 7.73                             | 129.2                                 | 376.7                               |
| 38   | retinaface                 | cf_retinaface_wider_360_640_1.11G             | 7.89                             | 126.7                                 | 474.8                               |
| 39   | face_quality               | cf_face-quality_80_60_61.68M                  | 0.48                             | 2060                                  | 6608.3                              |
| 40   | FPN-R18(light-weight)      | cf_FPN-resnet18_Endov_240_320_13.75G          | 29.38                            | 34                                    | 129.3                               |
| 41   | Hourglass                  | cf_hourglass-pe_mpii_256_256_10.2G            | 19.59                            | 51                                    | 106                                 |
| 42   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                  | 8.35                             | 119.7                                 | 324.9                               |
| 43   | yolov4                     | dk_yolov4_coco_416_416_60.1G                  | 73.22                            | 13.6                                  | 28.1                                |
| 44   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G  | 42.86                            | 23.3                                  | 43.7                                |
| 45   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G            | 5.69                             | 175.8                                 | 374.6                               |
| 46   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G        | 16.6                             | 60.2                                  | 117.9                               |
| 47   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G        | 33.09                            | 30.2                                  | 58.6                                |
| 48   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M   | 0.92                             | 1082.9                                | 3781                                |
| 49   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M   | 1.41                             | 708.8                                 | 2135                                |
| 50   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G     | 3.34                             | 299                                   | 724.4                               |
| 51   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G     | 3.98                             | 251.1                                 | 573.5                               |
| 52   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G     | 5.41                             | 184.7                                 | 392.7                               |
| 53   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G         | 12.15                            | 82.2                                  | 158.1                               |
| 54   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G        | 22.56                            | 44.3                                  | 85.9                                |
| 55   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G       | 32.84                            | 30.4                                  | 59                                  |
| 56   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G              | 47                               | 21.2                                  | 36.8                                |
| 57   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G              | 54.4                             | 18.3                                  | 32.4                                |
| 58   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G          | 10.08                            | 99.1                                  | 258.1                               |
| 59   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G          | 12.1                             | 82.6                                  | 186.9                               |
| 60   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G      | 353.32                           | 2.8                                   | 5.1                                 |
| 61   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                  | 72.74                            | 13.7                                  | 27.5                                |
| 62   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G        | 561.31                           | 1.8                                   | 5.1                                 |
| 63   | Inception_v2               | tf_inceptionv2_imagenet_224_224_3.88G         | 10.84                            | 92.2                                  | 187                                 |
| 64   | resnet\_v2\_50             | tf_resnetv2_50_imagenet_299_299_13.1G_1.3     | 27.89                            | 35.8                                  | 46.4                                |
| 65   | resnet\_v2\_101            | tf_resnetv2_101_imagenet_299_299_26.78G       | 47.67                            | 20.9                                  | 24.2                                |
| 66   | resnet\_v2\_152            | tf_resnetv2_152_imagenet_299_299_40.47G       | 67.31                            | 14.8                                  | 16.4                                |
| 67   | ssdlite_mobilenetv2        | tf_ssdlite_mobilenetv2_coco_300_300_1.5G      | 9.75                             | 102.5                                 | 258.9                               |
| 68   | ssd_inceptionv2            | tf_ssdinceptionv2_coco_300_300_9.62G_1.3      | 28.91                            | 34.6                                  | 83.9                                |
| 69   | Mobilenet\_v2              | tf_mobilenetv2_cityscapes_1024_2048_132.74G   | 615.68                           | 1.6                                   | 4.5                                 |
| 70   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G     | 13.51                            | 73.9                                  | 142.1                               |
| 71   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                | 93.44                            | 10.7                                  | 25.4                                |
| 72   | mobilenet_edge_1.0         | tf_mobilenetEdge1.0_imagenet_224_224_990M_1.3 | 4.94                             | 202.4                                 | 445.1                               |
| 73   | mobilenet_edge_0.75        | tf_mobilenetEdge0.75_imagenet_224_224_624M    | 4.13                             | 242                                   | 556.5                               |
| 74   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G        | 14.77                            | 67.7                                  | 164.9                               |
| 75   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G           | 13.68                            | 73.1                                  | 142.6                               |
| 76   | Mobilenet\_v1              | tf2_mobilenetv1_imagenet_224_224_1.15G        | 3.36                             | 297.4                                 | 719.6                               |
| 77   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G        | 16.9                             | 59.1                                  | 116.4                               |
| 78   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G              | 6.24                             | 160.2                                 | 334.7                               |
| 79   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G            | 145.2                            | 6.9                                   | 23.4                                |
| 80   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G              | 110.99                           | 9                                     | 38.3                                |
| 81   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G         | 29.72                            | 33.6                                  | 131.9                               |
| 82   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G         | 5.75                             | 173.7                                 | 306.8                               |
| 83   | face quality               | pt_face-quality_80_60_61.68M                  | 0.45                             | 2202.5                                | 6861                                |
| 84   | MT-resnet18                | pt_MT-resnet18_mixed_320_512_13.65G           | NA                               | NA                                    | NA                                  |
| 85   | face_reid_large            | pt_facereid-large_96_96_515M                  | 1.14                             | 877.7                                 | 1966.5                              |
| 86   | face_reid_small            | pt_facereid-small_80_80_90M                   | 0.53                             | 1882.8                                | 5511.7                              |
| 87   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G   | 9.78                             | 102.2                                 | 196.2                               |
| 88   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G    | 2.69                             | 370.9                                 | 682.9                               |
| 89   | pointpillars               | pt_pointpillars_kitti_12000_100_10.8G         | 52.48                            | 19                                    | 47.9                                |
| 90   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G | 189.89                           | 5.3                                   | 19.9                                |
| 91   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G     | 26.72                            | 37.4                                  | 78.8                                |
| 92   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                | 46.78                            | 21.4                                  | 57.4                                |
| 93   | Inception_v3               | torchvision_inception_v3                      | 16.41                            | 60.9                                  | 119.4                               |
| 94   | SqueezeNet                 | torchvision_squeezenet                        | 4.73                             | 211.4                                 | 669.9                               |
| 95   | resnet50                   | torchvision_resnet50                          | 14.04                            | 71.2                                  | 138                                 |

</details>


### Performance on U50
Measured with Vitis AI 1.3 and Vitis AI Library 1.3 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 6 DPUv3E kernels running at 300Mhz in Gen3x4:
  

| No\. | Model                      | Name                                          | Frequency(MHz) | E2E throughput \-fps\(Multi Thread) |
| ---- | :------------------------- | :-------------------------------------------- | -------------- | ----------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G             | 300            | 647.1                               |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G            | 300            | 1484.2                              |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G         | 300            | 1203.4                              |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G            | 300            | 976.2                               |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G         | 300            | 413.2                               |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G         | 300            | 189.5                               |
| 7    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G             | 300            | 3092.7                              |
| 8    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G       | 300            | 457.4                               |
| 9    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                | 300            | 47.2                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G             | 300            | 168.8                               |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G         | 300            | 405.2                               |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G          | 300            | 571.9                               |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G              | 300            | 579.7                               |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G               | 300            | 354.5                               |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G       | 300            | 395.8                               |
| 16   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                | 300            | 440.7                               |
| 17   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G           | 300            | 1339.0                              |
| 18   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G   | 300            | 28.4                                |
| 19   | densebox_320_320           | cf_densebox_wider_320_320_0.49G               | 300            | 1874.1                              |
| 20   | densebox_640_360           | cf_densebox_wider_360_640_1.11G               | 300            | 821.9                               |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                | 300            | 10057.4                             |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G               | 300            | 4010.7                              |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                | 300            | 321.1                               |
| 24   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                   | 300            | 78.1                                |
| 25   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G        | 300            | 675.8                               |
| 26   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                  | 300            | 80.6                                |
| 27   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                     | 300            | 163.9                               |
| 28   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G             | 300            | 413.1                               |
| 29   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G              | 300            | 484.1                               |
| 30   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G              | 300            | 589.7                               |
| 31   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G               | 300            | 1356.2                              |
| 32   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                | 300            | 507.9                               |
| 33   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G             | 300            | 103.2                               |
| 34   | plate detection            | cf_plate-detection_320_320_0.49G              | 300            | 4914.4                              |
| 35   | plate recognition          | cf_plate-recognition_96_288_1.75G             | 300            | 834.0                               |
| 36   | face_quality               | cf_face-quality_80_60_61.68M                  | 300            | 16238.6                             |
| 37   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                  | 300            | 874.7                               |
| 38   | yolov4                     | dk_yolov4_coco_416_416_60.1G                  | 300            | 84.1                                |
| 39   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G  | 300            | 186.4                               |
| 40   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G            | 300            | 1240.5                              |
| 41   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G        | 300            | 413.9                               |
| 42   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G        | 300            | 189.4                               |
| 43   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G         | 300            | 712.2                               |
| 44   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G        | 300            | 367.6                               |
| 45   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G       | 300            | 246.0                               |
| 46   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G              | 300            | 161.5                               |
| 47   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G              | 300            | 134.8                               |
| 48   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G      | 300            | 32.2                                |
| 49   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                  | 300            | 80.1                                |
| 50   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G        | 300            | 13.5                                |
| 51   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G     | 300            | 563.4                               |
| 52   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                | 300            | 69.5                                |
| 53   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G        | 300            | 422.0                               |
| 54   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G           | 300            | 583.6                               |
| 55   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G        | 300            | 372.5                               |
| 56   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G              | 300            | 1064.7                              |
| 57   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G            | 300            | 49.3                                |
| 58   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G              | 300            | 74.9                                |
| 59   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G         | 300            | 456.0                               |
| 60   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G         | 300            | 1235.9                              |
| 61   | face quality               | pt_face-quality_80_60_61.68M                  | 300            | 16458.3                             |
| 62   | face_reid_large            | pt_facereid-large_96_96_515M                  | 300            | 7468.9                              |
| 63   | face_reid_small            | pt_facereid-small_80_80_90M                   | 300            | 17815.5                             |
| 64   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G   | 300            | 916.4                               |
| 65   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G    | 300            | 3541.5                              |
| 66   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G | 300            | 129.8                               |
| 67   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G     | 300            | 212.6                               |
| 68   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                | 300            | 58.3                                |
| 69   | Inception_v3               | torchvision_inception_v3                      | 300            | 411.3                               |
| 70   | SqueezeNet                 | torchvision_squeezenet                        | 300            | 1987.7                              |
| 71   | resnet50                   | torchvision_resnet50                          | 300            | 624.4                               |


</details>


### Performance on U50 lv9e
Measured with Vitis AI 1.3 and Vitis AI Library 1.3 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 9 DPUv3E kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                      | Name                                          | Frequency(MHz) | E2E throughput \-fps\(Multi Thread) |
| ---- | :------------------------- | :-------------------------------------------- | -------------- | ----------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G             | 275            | 789.1                               |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G            | 275            | 1944.9                              |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G         | 275            | 1643.2                              |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G            | 275            | 1337.0                              |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G         | 275            | 565.0                               |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G         | 275            | 259.6                               |
| 7    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G             | 275            | 3926.6                              |
| 8    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G       | 275            | 728.0                               |
| 9    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                | 275            | 75.8                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G             | 275            | 284.6                               |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G         | 275            | 606.9                               |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G          | 275            | 875.3                               |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G              | 275            | 896.1                               |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G               | 275            | 587.3                               |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G       | 275            | 582.5                               |
| 16   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                | 275            | 544.1                               |
| 17   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G           | 275            | 2802.2                              |
| 18   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G   | 275            | 44.0                                |
| 19   | densebox_320_320           | cf_densebox_wider_320_320_0.49G               | 275            | 2317.5                              |
| 20   | densebox_640_360           | cf_densebox_wider_360_640_1.11G               | 275            | 1020.0                              |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                | 275            | 13350.3                             |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G               | 275            | 5250.9                              |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                | 275            | 391.1                               |
| 24   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                   | 275            | 104.2                               |
| 25   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G        | 275            | 851.2                               |
| 26   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                  | 275            | 106.3                               |
| 27   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                     | 275            | 225.0                               |
| 28   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G             | 275            | 566.9                               |
| 29   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G              | 275            | 664.3                               |
| 30   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G              | 275            | 807.3                               |
| 31   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G               | 275            | 1830.6                              |
| 32   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                | 275            | 676.4                               |
| 33   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G             | 275            | 140.0                               |
| 34   | plate detection            | cf_plate-detection_320_320_0.49G              | 275            | 5225.8                              |
| 35   | plate recognition          | cf_plate-recognition_96_288_1.75G             | 275            | 1281.5                              |
| 36   | face_quality               | cf_face-quality_80_60_61.68M                  | 275            | 18211.6                             |
| 37   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                  | 275            | 1193.5                              |
| 38   | yolov4                     | dk_yolov4_coco_416_416_60.1G                  | 275            | 111.4                               |
| 39   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G  | 275            | 238.0                               |
| 40   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G            | 275            | 1705.9                              |
| 41   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G        | 275            | 564.7                               |
| 42   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G        | 275            | 259.7                               |
| 43   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G         | 275            | 885.8                               |
| 44   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G        | 275            | 459.8                               |
| 45   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G       | 275            | 306.6                               |
| 46   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G              | 275            | 222.0                               |
| 47   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G              | 275            | 185.3                               |
| 48   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G      | 275            | 43.1                                |
| 49   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                  | 275            | 106.4                               |
| 50   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G        | 275            | 18.8                                |
| 51   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G     | 275            | 689.5                               |
| 52   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                | 275            | 97.8                                |
| 53   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G        | 275            | 579.6                               |
| 54   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G           | 275            | 711.6                               |
| 55   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G        | 275            | 510.1                               |
| 56   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G              | 275            | 1453.6                              |
| 57   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G            | 275            | 65.4                                |
| 58   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G              | 275            | 88.6                                |
| 59   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G         | 275            | 563.2                               |
| 60   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G         | 275            | 1668.6                              |
| 61   | face quality               | pt_face-quality_80_60_61.68M                  | 275            | 17832.6                             |
| 62   | face_reid_large            | pt_facereid-large_96_96_515M                  | 275            | 9523.9                              |
| 63   | face_reid_small            | pt_facereid-small_80_80_90M                   | 275            | 19805.9                             |
| 64   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G   | 275            | 1108.4                              |
| 65   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G    | 275            | 4645.1                              |
| 66   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G | 275            | 134.9                               |
| 67   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G     | 275            | 286.5                               |
| 68   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                | 275            | 79.8                                |
| 69   | Inception_v3               | torchvision_inception_v3                      | 275            | 564.4                               |
| 70   | SqueezeNet                 | torchvision_squeezenet                        | 275            | 2543.0                              |
| 71   | resnet50                   | torchvision_resnet50                          | 275            | 764.1                               |


</details>


### Performance on U50 lv10e
Measured with Vitis AI 1.3 and Vitis AI Library 1.3 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 10 DPUv3E kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                      | Name                                          | Frequency(MHz) | E2E throughput \-fps\(Multi Thread) |
| ---- | :------------------------- | :-------------------------------------------- | -------------- | ----------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G             | 275            | 790.3                               |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G            | 275            | 1936.8                              |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G         | 275            | 1579.3                              |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G            | 275            | 1277.6                              |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G         | 275            | 488.3                               |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G         | 275            | 223.3                               |
| 7    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G             | 275            | 3501.0                              |
| 8    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G       | 275            | 620.5                               |
| 9    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                | 275            | 26.0                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G             | 275            | 249.4                               |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G         | 275            | 572.3                               |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G          | 275            | 783.0                               |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G              | 275            | 727.9                               |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G               | 275            | 507.7                               |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G       | 275            | 573.7                               |
| 16   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                | 275            | 567.9                               |
| 17   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G           | 275            | 1929.1                              |
| 18   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G   | 275            | 33.3                                |
| 19   | densebox_320_320           | cf_densebox_wider_320_320_0.49G               | 275            | 2394.8                              |
| 20   | densebox_640_360           | cf_densebox_wider_360_640_1.11G               | 275            | 1041.3                              |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                | 275            | 14694.9                             |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G               | 275            | 5797.1                              |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                | 275            | 368.1                               |
| 24   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                   | 275            | 100.9                               |
| 25   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G        | 275            | 787.1                               |
| 26   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                  | 275            | 102.6                               |
| 27   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                     | 275            | 220.8                               |
| 28   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G             | 275            | 555.2                               |
| 29   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G              | 275            | 651.6                               |
| 30   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G              | 275            | 790.6                               |
| 31   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G               | 275            | 2030.8                              |
| 32   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                | 275            | 751.0                               |
| 33   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G             | 275            | 139.5                               |
| 34   | plate detection            | cf_plate-detection_320_320_0.49G              | 275            | 5160.8                              |
| 35   | plate recognition          | cf_plate-recognition_96_288_1.75G             | 275            | 1070.1                              |
| 36   | face_quality               | cf_face-quality_80_60_61.68M                  | 275            | 19352.6                             |
| 37   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                  | 275            | 1317.2                              |
| 38   | yolov4                     | dk_yolov4_coco_416_416_60.1G                  | 275            | 119.8                               |
| 39   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G  | 275            | 210.1                               |
| 40   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G            | 275            | 1643.1                              |
| 41   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G        | 275            | 488.5                               |
| 42   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G        | 275            | 223.5                               |
| 43   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G         | 275            | 887.3                               |
| 44   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G        | 275            | 460.6                               |
| 45   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G       | 275            | 307.0                               |
| 46   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G              | 275            | 225.1                               |
| 47   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G              | 275            | 187.3                               |
| 48   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G      | 275            | 41.2                                |
| 49   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                  | 275            | 104.3                               |
| 50   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G        | 275            | 18.5                                |
| 51   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G     | 275            | 765.7                               |
| 52   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                | 275            | 105.1                               |
| 53   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G        | 275            | 630.8                               |
| 54   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G           | 275            | 790.3                               |
| 55   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G        | 275            | 554.9                               |
| 56   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G              | 275            | 1577.9                              |
| 57   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G            | 275            | 64.5                                |
| 58   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G              | 275            | 85.5                                |
| 59   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G         | 275            | 591.9                               |
| 60   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G         | 275            | 1850.8                              |
| 61   | face quality               | pt_face-quality_80_60_61.68M                  | 275            | 19209.5                             |
| 62   | face_reid_large            | pt_facereid-large_96_96_515M                  | 275            | 10486.7                             |
| 63   | face_reid_small            | pt_facereid-small_80_80_90M                   | 275            | 20809.8                             |
| 64   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G   | 275            | 1230.2                              |
| 65   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G    | 275            | 5090.4                              |
| 66   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G | 275            | 137.6                               |
| 67   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G     | 275            | 313.2                               |
| 68   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                | 275            | 70.6                                |
| 69   | Inception_v3               | torchvision_inception_v3                      | 275            | 488.5                               |
| 70   | SqueezeNet                 | torchvision_squeezenet                        | 275            | 2334.1                              |
| 71   | resnet50                   | torchvision_resnet50                          | 275            | 765.6                               |


</details>


### Performance on U200
Measured with Vitis AI 1.3 and Vitis AI Library 1.3 

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
Measured with Vitis AI 1.3 and Vitis AI Library 1.3 

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

</details>

### Performance on U280
Measured with Vitis AI 1.3 and Vitis AI Library 1.3 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U280` board with 14 DPUv3E kernels running at 300Mhz in Gen3x4:
  

| No\. | Model                      | Name                                          | Frequency(MHz) | E2E throughput \-fps\(Multi Thread) |
| ---- | :------------------------- | :-------------------------------------------- | -------------- | ----------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G             | 300            | 801.2                               |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G            | 300            | 1649.2                              |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G         | 300            | 1235.5                              |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G            | 300            | 1055.4                              |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G         | 300            | 416.3                               |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G         | 300            | 189.9                               |
| 7    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G             | 300            | 2899.2                              |
| 8    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G       | 300            | 434.9                               |
| 9    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                | 300            | 46.4                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G             | 300            | 171.2                               |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G         | 300            | 427.2                               |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G          | 300            | 591                                 |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G              | 300            | 570                                 |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G               | 300            | 310.2                               |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G       | 300            | 581.5                               |
| 16   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                | 300            | 374.9                               |
| 17   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G           | 300            | 2449.3                              |
| 18   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G   | 300            | 35.3                                |
| 19   | densebox_320_320           | cf_densebox_wider_320_320_0.49G               | 300            | 2924.8                              |
| 20   | densebox_640_360           | cf_densebox_wider_360_640_1.11G               | 300            | 1300.1                              |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                | 300            | 12371                               |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G               | 300            | 4765.6                              |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                | 300            | 226.6                               |
| 24   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                   | 300            | 92                                  |
| 25   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G        | 300            | 790.3                               |
| 26   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                  | 300            | 96.5                                |
| 27   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                     | 300            | 196.9                               |
| 28   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G             | 300            | 495.7                               |
| 29   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G              | 300            | 579.7                               |
| 30   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G              | 300            | 691.7                               |
| 31   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G               | 300            | 1598.4                              |
| 32   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                | 300            | 580.6                               |
| 33   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G             | 300            | 107.4                               |
| 34   | plate detection            | cf_plate-detection_320_320_0.49G              | 300            | 4288.6                              |
| 35   | plate recognition          | cf_plate-recognition_96_288_1.75G             | 300            | 1100.8                              |
| 36   | face_quality               | cf_face-quality_80_60_61.68M                  | 300            | 23507                               |
| 37   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                  | 300            | 920.2                               |
| 38   | yolov4                     | dk_yolov4_coco_416_416_60.1G                  | 300            | 95.3                                |
| 39   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G  | 300            | 186.4                               |
| 40   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G            | 300            | 1293.2                              |
| 41   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G        | 300            | 414.8                               |
| 42   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G        | 300            | 189.6                               |
| 43   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G         | 300            | 751.7                               |
| 44   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G        | 300            | 390.1                               |
| 45   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G       | 300            | 259.9                               |
| 46   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G              | 300            | 183.7                               |
| 47   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G              | 300            | 153.9                               |
| 48   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G      | 300            | 30.7                                |
| 49   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                  | 300            | 95.7                                |
| 50   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G        | 300            | 11.2                                |
| 51   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G     | 300            | 647.3                               |
| 52   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                | 300            | 85.2                                |
| 53   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G        | 300            | 498.7                               |
| 54   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G           | 300            | 668.2                               |
| 55   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G        | 300            | 437.5                               |
| 56   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G              | 300            | 102.5                               |
| 57   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G            | 300            | 54.1                                |
| 58   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G              | 300            | 88.5                                |
| 59   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G         | 300            | 408.9                               |
| 60   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G         | 300            | 1598.4                              |
| 61   | face quality               | pt_face-quality_80_60_61.68M                  | 300            | 23286                               |
| 62   | face_reid_large            | pt_facereid-large_96_96_515M                  | 300            | 9164.1                              |
| 63   | face_reid_small            | pt_facereid-small_80_80_90M                   | 300            | 20088.5                             |
| 64   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G   | 300            | 1044.9                              |
| 65   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G    | 300            | 4521.9                              |
| 66   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G | 300            | 105                                 |
| 67   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G     | 300            | 221.2                               |
| 68   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                | 300            | 102.5                               |
| 69   | Inception_v3               | torchvision_inception_v3                      | 300            | 415.5                               |
| 70   | SqueezeNet                 | torchvision_squeezenet                        | 300            | 1780.7                              |
| 71   | resnet50                   | torchvision_resnet50                          | 300            | 776.2                               |

</details>

### Performance on VCK190
Measured with Vitis AI 1.3 and Vitis AI Library 1.3 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Versal VCK190` board with 96 AIEs running at 1333Mhz:
  

| No\. | Model                      | Name                                          | Frequency(MHz) | E2E throughput \-fps\(Multi Thread) |
| ---- | :------------------------- | :-------------------------------------------- | -------------- | ----------------------------------- |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G             | 300            | 1295.9                              |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G            | 300            | 2831.2                              |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G         | 300            | 1501.7                              |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G            | 300            | 1089.9                              |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G         | 300            | 572.6                               |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G         | 300            | 278.8                               |
| 7    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G             | 300            | 1794.7                              |
| 8    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G       | 300            | 673.6                               |
| 9    | refinedet\_baseline        | cf_refinedet_coco_360_480_123G                | 300            | 138.8                               |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G             | 300            | 385.9                               |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G         | 300            | 593.6                               |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G          | 300            | 796.9                               |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G              | 300            | 720.4                               |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G               | 300            | 622.8                               |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G       | 300            | 463.8                               |
| 16   | FPN                        | cf_fpn_cityscapes_256_512_8.9G                | 300            | 196.6                               |
| 17   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G           | 300            | 3408.7                              |
| 18   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G   | 300            | 43.6                                |
| 19   | densebox_320_320           | cf_densebox_wider_320_320_0.49G               | 300            | 2146.6                              |
| 20   | densebox_640_360           | cf_densebox_wider_360_640_1.11G               | 300            | 1040                                |
| 21   | face_landmark              | cf_landmark_celeba_96_72_0.14G                | 300            | 10031.2                             |
| 22   | reid                       | cf_reid_market1501_160_80_0.95G               | 300            | 4825.8                              |
| 23   | multi_task                 | cf_multitask_bdd_288_512_14.8G                | 300            | 197.2                               |
| 24   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                   | 300            | 185                                 |
| 25   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G        | 300            | 866.8                               |
| 26   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                  | 300            | 185                                 |
| 27   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                     | 300            | 439.2                               |
| 28   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G             | 300            | 879.8                               |
| 29   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G              | 300            | 978                                 |
| 30   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G              | 300            | 1093.9                              |
| 31   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G               | 300            | 2561.6                              |
| 32   | ResNet64-face              | cf_facerec-resnet64_112_96_11G                | 300            | 1260.4                              |
| 33   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G             | 300            | 75.5                                |
| 34   | plate detection            | cf_plate-detection_320_320_0.49G              | 300            | 2518.1                              |
| 35   | plate recognition          | cf_plate-recognition_96_288_1.75G             | 300            | 716.3                               |
| 36   | face_quality               | cf_face-quality_80_60_61.68M                  | 300            | 18695.5                             |
| 37   | tiny-yolov3                | dk_tiny-yolov3_416_416_5.46G                  | 300            | 1322.7                              |
| 38   | yolov4                     | dk_yolov4_coco_416_416_60.1G                  | 300            | 147.4                               |
| 39   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G  | 300            | 288.3                               |
| 40   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G            | 300            | 1523.5                              |
| 41   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G        | 300            | 572.7                               |
| 42   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G        | 300            | 278.7                               |
| 43   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G         | 300            | 1385.5                              |
| 44   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G        | 300            | 779                                 |
| 45   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G       | 300            | 546.5                               |
| 46   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G              | 300            | 336                                 |
| 47   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G              | 300            | 302                                 |
| 48   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G      | 300            | 11.8                                |
| 49   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                  | 300            | 184                                 |
| 50   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G        | 300            | 22.8                                |
| 51   | mlperf_resnet50            | tf_mlperf_resnet50_imagenet_224_224_8.19G     | 300            | 1288.3                              |
| 52   | refinedet                  | tf_refinedet_VOC_320_320_81.9G                | 300            | 175.7                               |
| 53   | refinedet_medical          | tf_RefineDet-Medical_EDD_320_320_9.83G        | 300            | 747.3                               |
| 54   | resnet50                   | tf2_resnet50_imagenet_224_224_7.76G           | 300            | 1275.1                              |
| 55   | Inception_v3               | tf2_inceptionv3_imagenet_299_299_11.5G        | 300            | 603.6                               |
| 56   | 2d-unet                    | tf2_2d-unet_nuclei_128_128_5.31G              | 300            | 1746.5                              |
| 57   | ERFNet                     | tf2_erfnet_cityscapes_512_1024_54G            | 300            | 48.9                                |
| 58   | ENet                       | pt_ENet_cityscapes_512_1024_8.6G              | 300            | 49.7                                |
| 59   | SemanticFPN                | pt_SemanticFPN_cityscapes_256_512_10G         | 300            | 196.9                               |
| 60   | ResNet20-face              | pt_facerec-resnet20_mixed_112_96_3.5G         | 300            | 2561.4                              |
| 61   | face quality               | pt_face-quality_80_60_61.68M                  | 300            | 19593.9                             |
| 62   | face_reid_large            | pt_facereid-large_96_96_515M                  | 300            | 11841.2                             |
| 63   | face_reid_small            | pt_facereid-small_80_80_90M                   | 300            | 16828.4                             |
| 64   | person_reid                | pt_personreid-res50_market1501_256_128_5.4G   | 300            | 1804.7                              |
| 65   | person_reid                | pt_personreid-res18_market1501_176_80_1.1G    | 300            | 4666.3                              |
| 66   | salsanext                  | pt_salsanext_semantic-kitti_64_2048_0.6_20.4G | 300            | 21.7                                |
| 67   | FPN-R18 (light-weight)     | pt_FPN-resnet18_covid19-seg_352_352_22.7G     | 300            | 330.4                               |
| 68   | 2d-unet                    | pt_unet_chaos-CT_512_512_23.3G                | 300            | 148.7                               |
| 69   | Inception_v3               | torchvision_inception_v3                      | 300            | 567.9                               |
| 70   | SqueezeNet                 | torchvision_squeezenet                        | 300            | 1677.4                              |
| 71   | resnet50                   | torchvision_resnet50                          | 300            | 1267.9                              |

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
