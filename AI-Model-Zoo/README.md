<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Model Zoo</h1>
    </td>
 </tr>
 </table>

# Introduction
This repository includes optimized deep learning models to speed up the deployment of deep learning inference on Xilinx&trade; platforms. These models cover different applications, including but not limited to ADAS/AD, video surveillance, robotics, data center, etc. You can get started with these free pre-trained models to enjoy the benefits of deep learning acceleration.

<p align="left">
  <img width="1264" height="420" src="images/vitis_ai_model_zoo.png">
</p>

## Vitis AI 1.2 Model Zoo New Features！
1.Newly added 8 Pytorch models and provide test & training code for them.</br>
2.The variety and quantity of Caffe models have been improved, newly added such as vehicle license plate recognition and medical segmentation models.</br>
3.Updated information about more supported edge and cloud hardware platforms and performance.

## Model Information
The following table includes comprehensive information about each model, including application, framework, training and validation dataset, backbone, input size, computation as well as float and quantized precision.<br>
At present, most of the python scripts we provided are compatible with python2/3, except a few models(No.21~23 and 25)need python2 environment.

<details>
 <summary><b>Click here to view details</b></summary>

| No\. | Application              | Model                          | Name                                                      | Framework  | Backbone       | Input Size | OPS per image | Training Set                            | Val Set                 | Float \(Top1, Top5\)/ mAP/mIoU                               | Quantized \(Top1, Top5\)/mAP/mIoU                            |
| :--- | :----------------------- | :----------------------------- | :-------------------------------------------------------- | :--------- | -------------- | ---------- | ------------- | --------------------------------------- | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Image Classification     | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G\_1\.2             | caffe      | resnet50       | 224\*224   | 7\.7G         | ImageNet Train                          | ImageNet Val            | 0\.7444/0\.9185                                              | 0\.7334/0\.9131                                              |
| 2    | Image Classifiction      | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G\_1\.2            | caffe      | resnet18       | 224\*224   | 3\.65G        | ImageNet Train                          | ImageNet Val            | 0\.6832/0\.8848                                              | 0.6699/0.8826                                                |
| 3    | Image Classification     | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G\_1\.2         | caffe      | inception\_v1  | 224\*224   | 3\.16G        | ImageNet Train                          | ImageNet Val            | 0\.7030/0\.8971                                              | 0\.6984/0\.8942                                              |
| 4    | Image Classification     | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G\_1\.2             | caffe      | bn\-inception  | 224\*224   | 4G            | ImageNet Train                          | ImageNet Val            | 0\.7275/0\.9111                                              | 0\.7168/0\.9029                                              |
| 5    | Image Classification     | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G\_1\.2         | caffe      | inception\_v3  | 299\*299   | 11\.4G        | ImageNet Train                          | ImageNet Val            | 0\.7701/0\.9329                                              | 0\.7626/0\.9303                                              |
| 6    | Image Classification     | Inception\_v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G\_1\.2         | caffe      | inception\_v3  | 299\*299   | 24\.5G        | ImageNet Train                          | ImageNet Val            | 0\.7958/0\.9470                                              | 0\.7898/0\.9445                                              |
| 7    | Image Classification     | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G\_1\.2         | caffe      | MobileNet\_v2  | 224\*224   | 608M          | ImageNet Train                          | ImageNet Val            | 0\.6475/0\.8609                                              | 0\.6354/0\.8506                                              |
| 8    | Image Classifiction      | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G\_1\.2             | caffe      | squeezenet     | 227\*227   | 0\.76G        | ImageNet Train                          | ImageNet Val            | 0\.5438/0\.7813                                              | 0\.5026/0\.7658                                              |
| 9    | ADAS Pedstrain Detection | ssd\_pedestrian\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G\_1\.2     | caffe      | VGG\-bn\-16    | 360\*640   | 5\.9G         | coco2014\_train\_person and crowndhuman | coco2014\_val\_person   | 0\.5903                                                      | 0\.5876                                                      |
| 10   | Object Detection         | refinedet\_baseline            | cf_refinedet_coco_480_360_123G_1.2                        | caffe      | VGG\-bn\-16    | 360\*480   | 123G          | coco2014\_train\_person                 | coco2014\_val\_person   | 0.6928                                                       | 0.7042                                                       |
| 11   | Object Detection         | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G\_1\.2            | caffe      | VGG\-bn\-16    | 360\*480   | 25G           | coco2014\_train\_person                 | coco2014\_val\_person   | 0\.6794                                                      | 0\.6780                                                      |
| 12   | Object Detection         | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G\_1\.2       | caffe      | VGG\-bn\-16    | 360\*480   | 10\.10G       | coco2014\_train\_person                 | coco2014\_val\_person   | 0\.6489                                                      | 0\.6486                                                      |
| 13   | Object Detection         | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G\_1\.2        | caffe      | VGG\-bn\-16    | 360\*480   | 5\.08G        | coco2014\_train\_person                 | coco2014\_val\_person   | 0\.6120                                                      | 0\.6113                                                      |
| 14   | ADAS Vehicle Detection   | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G\_1\.2            | caffe      | VGG\-16        | 360\*480   | 6\.3G         | bdd100k \+ private data                 | bdd100k \+ private data | 0\.4207                                                      | 0\.4200                                                      |
| 15   | Traffic Detection        | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G\_1\.2              | caffe      | VGG\-16        | 360\*480   | 11\.6G        | private data                            | private data            | 0\.5982                                                      | 0\.5921                                                      |
| 16   | ADAS Lane Detection      | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G\_1\.2     | caffe      | VGG            | 480\*640   | 2\.5G         | caltech\-lanes\-train\-dataset          | caltech lane            | 0\.8864\(F1\-score\)                                         | 0\.8882\(F1\-score\)                                         |
| 17   | Object Detection         | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G\_1\.2           | caffe      | MobileNet\_v2  | 360\*480   | 6\.57G        | bdd100k train                           | bdd100k val             | 0\.3052                                                      | 0\.2752                                                      |
| 18   | ADAS Segmentation        | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G\_1\.2                | caffe      | Google\_v1\_BN | 256\*512   | 8\.9G         | Cityscapes gtFineTrain                  | Cityscapes Val          | 0\.5669                                                      | 0\.5662                                                      |
| 19   | Pose Estimation          | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G\_1\.2           | caffe      | Google\_v1\_BN | 128\*224   | 548\.6M       | ai\_challenger                          | ai\_challenger          | 0\.9000\(PCKh0\.5\)                                          | 0\.8964\(PCKh0\.5\)                                          |
| 20   | Pose Estimation          | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G\_1\.2 | caffe      | VGG            | 368\*368   | 49\.88G       | ai\_challenger                          | ai\_challenger          | 0\.4507\(OKs\)                                               | 0\.4422\(Oks\)                                               |
| 21   | Face Detection           | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G\_1\.2               | caffe      | VGG\-16        | 320\*320   | 0\.49G        | wider\_face                             | FDDB                    | 0\.8833                                                      | 0\.8791                                                      |
| 22   | Face Detection           | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G\_1\.2               | caffe      | VGG\-16        | 360\*640   | 1\.11G        | wider\_face                             | FDDB                    | 0\.8931                                                      | 0\.8925                                                      |
| 23   | Face Recognition         | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G\_1\.2                | caffe      | lenet          | 96\*72     | 0\.14G        | celebA                                  | processed helen         | 0\.1952\(L2 loss\)                                           | 0\.1972\(L2 loss\)                                           |
| 24   | Re\-identification       | reid                           | cf\_reid\_market1501\_160\_80\_0\.95G\_1\.2               | caffe      | resnet18       | 160\*80    | 0\.95G        | Market1501\+CUHK03                      | Market1501              | 0\.7800                                                      | 0\.7790                                                      |
| 25   | Detection+Segmentation   | multi-task                     | cf\_multitask\_bdd\_288\_512\_14\.8G\_1\.2                | caffe      | ssd            | 288\*512   | 14\.8G        | BDD100K+Cityscapes                      | BDD100K+Cityscapes      | 0\.2228(Det) 0\.4088(Seg)                                    | 0\.2202(Det) 0\.4058(Seg)                                    |
| 26   | Object Detection         | yolov3\_bdd                    | dk\_yolov3\_bdd\_288\_512\_53\.7G\_1\.2                   | darknet    | darknet\-53    | 288\*512   | 53\.7G        | bdd100k                                 | bdd100k                 | 0\.5058                                                      | 0\.4914                                                      |
| 27   | Object Detection         | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G\_1\.2      | darknet    | darknet\-53    | 256\*512   | 5\.46G        | Cityscapes Train                        | Cityscape Val           | 0\.5520                                                      | 0\.5300                                                      |
| 28   | Object Detection         | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G\_1\.2                  | darknet    | darknet\-53    | 416\*416   | 65\.42G       | voc07\+12\_trainval                     | voc07\_test             | 0\.8240\(MaxIntegral\)                                       | 0\.8150\(MaxIntegral\)                                       |
| 29   | Object Detection         | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G\_1\.2                      | darknet    | darknet\-19    | 448\*448   | 34G           | voc07\+12\_trainval                     | voc07\_test             | 0\.7845\(MaxIntegral\)                                       | 0\.7739\(MaxIntegral\)                                       |
| 30   | Object Detection         | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G\_1\.2           | darknet    | darknet\-19    | 448\*448   | 11\.56G       | voc07\+12\_trainval                     | voc07\_test             | 0\.7700\(MaxIntegral\)                                       | 0\.7600\(MaxIntegral\)                                       |
| 31   | Object Detection         | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G\_1\.2            | darknet    | darknet\-19    | 448\*448   | 9\.86G        | voc07\+12\_trainval                     | voc07\_test             | 0\.7670\(MaxIntegral\)                                       | 0\.7530\(MaxIntegral\)                                       |
| 32   | Object Detection         | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G\_1\.2            | darknet    | darknet\-19    | 448\*448   | 7\.82G        | voc07\+12\_trainval                     | voc07\_test             | 0\.7576\(MaxIntegral\)                                       | 0\.7460\(MaxIntegral\)                                       |
| 33   | Face Recognition         | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G_1.2                       | caffe      | resnet20       | 112*96     | 3.5G          | private data                            | private data            | 0.9610                                                       | 0.9510                                                       |
| 34   | Face Recognition         | ResNet64-face                  | cf_facerec-resnet64_112_96_11G_1.2                        | caffe      | resnet64       | 112*96     | 11G           | private data                            | private data            | 0.9830                                                       | 0.9820                                                       |
| 35   | Medical Segmentation     | FPN_Res18_Medical_Segmentation | cf_FPN-resnet18_EDD_320_320_45.3G_1.2                     | caffe      | resnet18       | 320*320    | 45.3G         | EDD_seg                                 | EDD_seg                 | mean dice=0.8202                mean jaccard=0.7925                                                                       F2-score=0.8075 | mean dice =0.8049                     mean jaccard =0.7771                                                                              F2- score=0.7916 |
| 36   | Plate Detection          | plate_detection                | cf_plate-detection_320_320_0.49G_1.2                      | caffe      | modify_vgg     | 320*320    | 0.49G         | private data                            | private data            | 0.9720                                                       | 0.9700                                                       |
| 37   | Plate Recognition        | plate_recognition              | cf_plate-recognition_96_288_1.75G_1.2                     | caffe      | Google\_v1     | 96*288     | 1.75G         | private data                            | private data            | plate number:99.51% plate color:100%                         | plate number:99.51% plate color:100%                         |
| 38   | Image Classifiction      | Inception\_resnet\_v2          | tf\_inceptionresnetv2\_imagenet\_299\_299\_26\.35G\_1\.2  | tensorflow | inception      | 299\*299   | 26\.35G       | ImageNet Train                          | ImageNet Val            | 0\.8037                                                      | 0\.7946                                                      |
| 39   | Image Classifiction      | Inception\_v1                  | tf\_inceptionv1\_imagenet\_224\_224\_3G\_1\.2             | tensorflow | inception      | 224\*224   | 3G            | ImageNet Train                          | ImageNet Val            | 0\.6976                                                      | 0\.6794                                                      |
| 40   | Image Classifiction      | Inception\_v3                  | tf\_inceptionv3\_imagenet\_299\_299\_11\.45G\_1\.2        | tensorflow | inception      | 299\*299   | 11\.45G       | ImageNet Train                          | ImageNet Val            | 0\.7798                                                      | 0\.7607                                                      |
| 41   | Image Classifiction      | Inception\_v4                  | tf\_inceptionv4\_imagenet\_299\_299\_24\.55G\_1\.2        | tensorflow | inception      | 299\*299   | 24\.55G       | ImageNet Train                          | ImageNet Val            | 0\.8018                                                      | 0\.7928                                                      |
| 42   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_0\.25\_imagenet\_128\_128\_27\.15M\_1\.2 | tensorflow | mobilenet      | 128\*128   | 27\.15M       | ImageNet Train                          | ImageNet Val            | 0\.4144                                                      | 0\.3464                                                      |
| 43   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_0\.5\_imagenet\_160\_160\_150\.07M\_1\.2 | tensorflow | mobilenet      | 160\*160   | 150\.07M      | ImageNet Train                          | ImageNet Val            | 0\.5903                                                      | 0\.5195                                                      |
| 44   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_1\.0\_imagenet\_224\_224\_1\.14G\_1\.2   | tensorflow | mobilenet      | 224\*224   | 1\.14G        | ImageNet Train                          | ImageNet Val            | 0\.7102                                                      | 0\.6779                                                      |
| 45   | Image Classifiction      | Mobilenet\_v2                  | tf\_mobilenetv2\_1\.0\_imagenet\_224\_224\_0\.59G\_1\.2   | tensorflow | mobilenet      | 224\*224   | 0\.59G        | ImageNet Train                          | ImageNet Val            | 0\.7013                                                      | 0\.6767                                                      |
| 46   | Image Classifiction      | Mobilenet\_v2                  | tf\_mobilenetv2\_1\.4\_imagenet\_224\_224\_1\.16G\_1\.2   | tensorflow | mobilenet      | 224\*224   | 1\.16G        | ImageNet Train                          | ImageNet Val            | 0\.7411                                                      | 0\.7194                                                      |
| 47   | Image Classifiction      | resnet\_v1\_50                 | tf\_resnetv1\_50\_imagenet\_224\_224\_6\.97G\_1\.2        | tensorflow | resnetv1       | 224\*224   | 6\.97G        | ImageNet Train                          | ImageNet Val            | 0\.7520                                                      | 0\.7478                                                      |
| 48   | Image Classifiction      | resnet\_v1\_101                | tf\_resnetv1\_101\_imagenet\_224\_224\_14\.4G\_1\.2       | tensorflow | resnetv1       | 224\*224   | 14\.4G        | ImageNet Train                          | ImageNet Val            | 0\.7640                                                      | 0\.7560                                                      |
| 49   | Image Classifiction      | resnet\_v1\_152                | tf\_resnetv1\_152\_imagenet\_224\_224\_21\.83G\_1\.2      | tensorflow | resnetv1       | 224\*224   | 21\.83G       | ImageNet Train                          | ImageNet Val            | 0\.7681                                                      | 0\.7463                                                      |
| 50   | Image Classifiction      | vgg\_16                        | tf\_vgg16\_imagenet\_224\_224\_30\.96G\_1\.2              | tensorflow | vgg            | 224\*224   | 30\.96G       | ImageNet Train                          | ImageNet Val            | 0\.7089                                                      | 0\.7069                                                      |
| 51   | Image Classifiction      | vgg\_19                        | tf\_vgg19\_imagenet\_224\_224\_39\.28G\_1\.2              | tensorflow | vgg            | 224\*224   | 39\.28G       | ImageNet Train                          | ImageNet Val            | 0\.7100                                                      | 0\.7026                                                      |
| 52   | Object Detection         | ssd\_mobilenet\_v1             | tf\_ssdmobilenetv1\_coco\_300\_300\_2\.47G\_1\.2          | tensorflow | mobilenet      | 300\*300   | 2\.47G        | coco2017                                | coco2014 minival        | 0\.2080                                                      | 0\.2100                                                      |
| 53   | Object Detection         | ssd\_mobilenet\_v2             | tf\_ssdmobilenetv2\_coco\_300\_300\_3\.75G\_1\.2          | tensorflow | mobilenet      | 300\*300   | 3\.75G        | coco2017                                | coco2014 minival        | 0\.2150                                                      | 0\.2110                                                      |
| 54   | Object Detection         | ssd\_resnet50\_v1\_fpn         | tf\_ssdresnet50v1\_fpn\_coco\_640\_640\_178\.4G\_1\.2     | tensorflow | resnet50       | 300\*300   | 178\.4G       | coco2017                                | coco2014 minival        | 0\.3010                                                      | 0\.2900                                                      |
| 55   | Object Detection         | yolov3\_voc                    | tf\_yolov3\_voc\_416\_416\_65\.63G\_1\.2                  | tensorflow | darknet\-53    | 416\*416   | 65\.63G       | voc07\+12\_trainval                     | voc07\_test             | 0\.7846                                                      | 0\.7744                                                      |
| 56   | Object Detection         | mlperf\_ssd\_resnet34          | tf\_mlperf_resnet34\_coco\_1200\_1200\_433G\_1\.2         | tensorflow | resnet34       | 1200\*1200 | 433G          | coco2017                                | coco2017                | 0\.2250                                                      | 0\.2150                                                      |
| 57   | Segmentation             | ENet                           | pt_ENet_cityscapes_512_1024_8.6G_1.2                      | pytorch    | -              | 512*1024   | 8.6G          | Cityscapes                              | Cityscapes              | 0.6440                                                       | 0.6306                                                       |
| 58   | Segmentation             | SemanticFPN                    | pt_SemanticFPN_cityscapes_256_512_10G_1.2                 | pytorch    | FPN-Resnet18   | 256*512    | 10G           | Cityscapes                              | Cityscapes              | 0.6290                                                       | 0.6090                                                       |
| 59   | Face Recognition         | ResNet20-face                  | pt_facerec-resnet20_mixed_112_96_3.5G_1.2                 | pytorch    | resnet20       | 112*96     | 3.5G          | mixed                                   | mixed                   | 0.9955                                                       | 0.9952                                                       |
| 60   | Face Quality             | face quality                   | pt_face-quality_80_60_61.68M_1.2                          | pytorch    | lenet          | 80*60      | 61.68M        | private data                            | private data            | 0.1233                                                       | 0.1273                                                       |
| 61   | Multi Task               | MT-resnet18                    | pt_MT-resnet18_mixed_320_512_13.65G_1.2                   | pytorch    | resnet18       | 320*512    | 13.65G        | mixed                                   | mixed                   | mAP:  39.50%     mIOU: 44.03%                                | mAP:  38.70%     mIOU: 42.56%                                |
| 62   | Face ReID                | face_reid_large                | pt_facereid-large_96_96_515M_1.2                          | pytorch    | resnet18       | 96*96      | 515M          | private data                            | private data            | mAP: 79.5%  Rank1: 95.4%                                     | mAP: 79.0%         Rank1: 95.1%                              |
| 63   | Face ReID                | face_reid_small                | pt_facereid-small_80_80_90M_1.2                           | pytorch    | resnet_small   | 80*80      | 90M           | private data                            | private data            | mAP: 56.3%  Rank1: 86.8%                                     | mAP: 56.1%    Rank1: 86.4%                                   |
| 64   | Re\-identification       | reid                           | pt_personreid_market1501_256_128_4.2G_1.2                 | pytorch    | resnet50       | 256*128    | 4.2G          | market1501                              | market1501              | mAP: 84.0%   Rank1: 94.6%                                    | mAP: 83.5%     Rank1: 94.2%                                  |

</details>

### Naming Rules
Model name: `F_M_(D)_H_W_(P)_C_V`
* `F` specifies training framework: `cf` is Caffe, `tf` is Tensorflow, `dk` is Darknet, `pt` is PyTorch.
* `M` specifies the model feature.
* `D` specifies the dataset. It is optional depending on whether the dataset is public or private. Mixed means a mixture of multiple           public datasets.
* `H` specifies the height of input data.
* `W` specifies the width of input data.
* `P` specifies the pruning ratio, it means how much computation is reduced. It is optional depending on whether the model is pruned.
* `C` specifies the computation of the model: how many Gops per image.
* `V` specifies the version of Vitis AI.


For example, `cf_refinedet_coco_480_360_0.8_25G_1.2` is a `RefineDet` model trained with `Caffe` using `COCO` dataset, input data size is `480*360`, `80%` pruned, the computation per image is `25Gops` and Vitis AI version is `1.2`.


### caffe-xilinx 
This is a custom distribution of caffe. Please use **caffe-xilinx** to test/finetune the caffe models listed in this page.

## Model Download
The following table lists various models, download link and MD5 checksum for the zip file of each model.

**Note:** To download all the models, visit [all_models_1.2.zip](https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.2.zip).

<details>
 <summary><b>Click here to view details</b></summary>

If you are a:
 - Linux user, use the [`get_model.sh`](get_model.sh) script to download all the models.   
 - Windows user, use the download link listed in the following table to download a model.

| No\. | Model                                            | Size   | MD5                              | Download link |
| ---- | ------------------------------------------------ | ------ | -------------------------------- | ------------- |
| 1    | cf_resnet50_imagenet_224_224_7.7G_1.2            | 232MB  | a8c5880724319a46ab494e4b78cd8440 | https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet50_imagenet_224_224_7.7G_1.2.zip                  |
| 2    | cf_inceptionv1_imagenet_224_224_3.16G_1.2        | 88MB   | f997fa8dfd3a4b50078a96c0c385582f | https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv1_imagenet_224_224_3.16G_1.2.zip              |
| 3    | cf_inceptionv2_imagenet_224_224_4G_1.2           | 147MB  | 3dcd60cff65d8ef4c71943b5be818c33 | https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv2_imagenet_224_224_4G_1.2.zip                 |
| 4    | cf_inceptionv3_imagenet_299_299_11.4G_1.2        | 218MB  | 9ad9e3d30974ed38d5fbb9cb05991100 | https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv3_imagenet_299_299_11.4G_1.2.zip              |
| 5    | cf_inceptionv4_imagenet_299_299_24.5G_1.2        | 390MB  | f5e98bedc38a223be255b6c9f7cadeb8 | https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv4_imagenet_299_299_24.5G_1.2.zip              |
| 6    | cf_mobilenetv2_imagenet_224_224_0.59G_1.2        | 24.4MB | 1766c88335548f5a5777a9a658551727 | https://www.xilinx.com/bin/public/openDownload?filename=cf_mobilenetv2_imagenet_224_224_0.59G_1.2.zip              |
| 7    | cf_squeeze_imagenet_227_227_0.76G_1.2            | 11.5MB | 2b1dc4fdc41a084319cfabcf11ba5814 | https://www.xilinx.com/bin/public/openDownload?filename=cf_squeeze_imagenet_227_227_0.76G_1.2.zip                  |
| 8    | cf_resnet18_imagenet_224_224_3.65G_1.2           | 107MB  | 3f8a689b18de48eb6c959850776e20f0 | https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet18_imagenet_224_224_3.65G_1.2.zip                 |
| 9    | cf_ssdpedestrian_coco_360_640_0.97_5.9G_1.2      | 4.60MB | 380d1998d294f8127cd6762ca444553d | https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdpedestrian_coco_360_640_0.97_5.9G_1.2.zip            |
| 10   | cf_refinedet_coco_480_360_123G_1.2               | 274MB  | 0441a332c62add57fd0f53f965c0d93b | https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_123G_1.2.zip                     |
| 11   | cf_refinedet_coco_360_480_0.8_25G_1.2            | 23.3MB | b59cc0d5b7206e36257eca0ddca93ade | https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_0.8_25G_1.2.zip                  |
| 12   | cf_refinedet_coco_360_480_0.92_10.10G_1.2        | 6.45MB | 1573a7eea90de244d044182b2acf9281 | https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_0.92_10.10G_1.2.zip              |
| 13   | cf_refinedet_coco_360_480_0.96_5.08G_1.2         | 3.23MB | be21077f3eb75cc362894e35acd292a5 | https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_0.96_5.08G_1.2.zip               |
| 14   | cf_ssdadas_bdd_360_480_0.95_6.3G_1.2             | 7.10MB | bb56a43d8f11e51af946ca6c98bca9d3 | https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdadas_bdd_360_480_0.95_6.3G_1.2.zip                   |
| 15   | cf_ssdtraffic_360_480_0.9_11.6G_1.2              | 13.4MB | 6b704425288d612c037abc6f90d5ea2d | https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdtraffic_360_480_0.9_11.6G_1.2.zip                    |
| 16   | cf_VPGnet_caltechlane_480_640_0.99_2.5G_1.2      | 3.09MB | 96263ac15e378fa8c620509f97ee5144 | https://www.xilinx.com/bin/public/openDownload?filename=cf_VPGnet_caltechlane_480_640_0.99_2.5G_1.2.zip            |
| 17   | cf_ssdmobilenetv2_bdd_360_480_6.57G_1.2          | 67.4MB | 1886aa1a977ed7395c73db4c0854f979 | https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdmobilenetv2_bdd_360_480_6.57G_1.2.zip                |
| 18   | cf_fpn_cityscapes_256_512_8.9G_1.2               | 73.9MB | 10cbbb37ec8dcb6d980cdbfa8f0fad00 | https://www.xilinx.com/bin/public/openDownload?filename=cf_fpn_cityscapes_256_512_8.9G_1.2.zip                     |
| 19   | cf_SPnet_aichallenger_224_128_0.54G_1.2          | 10.4MB | af0d1706ba8aa2a24281fb534ce0ce7c | https://www.xilinx.com/bin/public/openDownload?filename=cf_SPnet_aichallenger_224_128_0.54G_1.2.zip                |
| 20   | cf_openpose_aichallenger_368_368_0.3_189.7G_1.2  | 334MB  | 1c8a8aa1734251f31e4e6c3e9815fac4 | https://www.xilinx.com/bin/public/openDownload?filename=cf_openpose_aichallenger_368_368_0.3_189.7G_1.2.zip        |
| 21   | cf_densebox_wider_320_320_0.49G_1.2              | 4.77MB | 584eb9f6be9439fcd6c0d5eb1be26da3 | https://www.xilinx.com/bin/public/openDownload?filename=cf_densebox_wider_320_320_0.49G_1.2.zip                    |
| 22   | cf_densebox_wider_360_640_1.11G_1.2              | 4.77MB | 37076a64065b4c0769835c47407f76c1 | https://www.xilinx.com/bin/public/openDownload?filename=cf_densebox_wider_360_640_1.11G_1.2.zip                    |
| 23   | cf_landmark_celeba_96_72_0.14G_1.2               | 51.7MB | d77f0f729eff3f366b08b0df32d25502 | https://www.xilinx.com/bin/public/openDownload?filename=cf_landmark_celeba_96_72_0.14G_1.2.zip                     |
| 24   | cf_reid_market1501_160_80_0.95G_1.2              | 101MB  | b6c3fd73b00e2402e24aa8bcf7a537ed | https://www.xilinx.com/bin/public/openDownload?filename=cf_reid_market1501_160_80_0.95G_1.2.zip                    |
| 25   | cf_multitask_bdd_288_512_14.8G_1.2               | 125MB  | 70f3e9331295461f946e8cec0c80a5d4 | https://www.xilinx.com/bin/public/openDownload?filename=cf_multitask_bdd_288_512_14.8G_1.2.zip                     |
| 26   | dk_yolov3_bdd_288_512_53.7G_1.2                  | 585MB  | 452587d385dede38b9f13414fbd6f903 | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_bdd_288_512_53.7G_1.2.zip                        |
| 27   | dk_yolov3_cityscapes_256_512_0.9_5.46G_1.2       | 22.5MB | 6421288c4c917b66d90481d27819bb63 | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_cityscapes_256_512_0.9_5.46G_1.2.zip             |
| 28   | dk_yolov3_voc_416_416_65.42G_1.2                 | 579MB  | b21107628cf29f16eccd990ac6c2790d | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_voc_416_416_65.42G_1.2.zip                       |
| 29   | dk_yolov2_voc_448_448_34G_1.2                    | 426MB  | 1e29052a274627f7cbe8348e2500b73d | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_34G_1.2.zip                          |
| 30   | dk_yolov2_voc_448_448_0.66_11.56G_1.2            | 141MB  | 4964e80f08999ad8737b47e2100dc638 | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.66_11.56G_1.2.zip                  |
| 31   | dk_yolov2_voc_448_448_0.71_9.86G_1.2             | 125MB  | 5b80a104ca9a14d809ebcf48240babb0 | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.71_9.86G_1.2.zip                   |
| 32   | dk_yolov2_voc_448_448_0.77_7.82G_1.2             | 91.3MB | 04f489c94d351c6d71faa0d031e2b2f6 | https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.77_7.82G_1.2.zip                   |
| 33   | cf_facerec-resnet20_112_96_3.5G_1.2              | 208MB  | fe008dcb1d6ca751fa6cd5909f575ec5 | https://www.xilinx.com/bin/public/openDownload?filename=cf_facerec-resnet20_112_96_3.5G_1.2.zip                    |
| 34   | cf_facerec-resnet64_112_96_11G_1.2               | 280MB  | 1831cdb446cece092035a444f683a654 | https://www.xilinx.com/bin/public/openDownload?filename=cf_facerec-resnet64_112_96_11G_1.2.zip                     |
| 35   | cf_FPN-resnet18_EDD_320_320_45.3G_1.2            | 107MB  | 6a4eb8b3da9f29c56f711ea1373a334e | https://www.xilinx.com/bin/public/openDownload?filename=cf_FPN-resnet18_EDD_320_320_45.3G_1.2.zip                  |
| 36   | cf_plate-detection_320_320_0.49G_1.2             | 5.13MB | 42c4b7c676f6d61135c1a419c323ed08 | https://www.xilinx.com/bin/public/openDownload?filename=cf_plate-detection_320_320_0.49G_1.2.zip                   |
| 37   | cf_plate-recognition_96_288_1.75G_1.2            | 57.4MB | 8c5a98efaa4c8146dd601635007fcd9b | https://www.xilinx.com/bin/public/openDownload?filename=cf_plate-recognition_96_288_1.75G_1.2.zip                  |
| 38   | tf_inceptionresnetv2_imagenet_299_299_26.35G_1.2 | 453MB  | c9f0d01087c4842aeef9521c5a6f3bbc | https://www.xilinx.com/bin/public/openDownload?filename=tf_inception_resnet_v2_imagenet_299_299_26.35G_1.2.zip     |
| 39   | tf_inceptionv1_imagenet_224_224_3G_1.2           | 54.2MB | ccc4de0d21947f83644ea68c7e0da2f6 | https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv1_imagenet_224_224_3G_1.2.zip                 |
| 40   | tf_inceptionv3_imagenet_299_299_11.45G_1.2       | 194MB  | 35fd3e8296d23f87ca817279bd5a0c65 | https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv3_imagenet_299_299_11.45G_1.2.zip             |
| 41   | tf_inceptionv4_imagenet_299_299_24.55G_1.2       | 347MB  | bc3e85dfe9d4b69e3300e0408f3e86e8 | https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv4_imagenet_299_299_24.55G_1.2.zip             |
| 42   | tf_mobilenetv1_0.25_imagenet_128_128_27M_1.2     | 3.89MB | 620bf6a0ee6ac762c05b90ce8346a1fd | https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_0.25_imagenet_128_128_27M_1.2.zip           |
| 43   | tf_mobilenetv1_0.5_imagenet_160_160_150M_1.2     | 10.8MB | 309a10176ee11e15d13cc4057b0dacd7 | https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_0.5_imagenet_160_160_150M_1.2.zip           |
| 44   | tf_mobilenetv1_1.0_imagenet_224_224_1.14G_1.2    | 33.9MB | 07d7ca9a88b61fc7a83a43fcaeefd044 | https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_1.0_imagenet_224_224_1.14G_1.2.zip          |
| 45   | tf_mobilenetv2_1.0_imagenet_224_224_602M_1.2     | 28.7MB | 918883dbc92315a73b760c9d04311a7d | https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv2_1.0_imagenet_224_224_602M_1.2.zip           |
| 46   | tf_mobilenetv2_1.4_imagenet_224_224_1.16G_1.2    | 49.7MB | d2e857cc35301c269047a95536e9428e | https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv2_1.4_imagenet_224_224_1.16G_1.2.zip          |
| 47   | tf_resnetv1_50_imagenet_224_224_6.97G_1.2        | 207MB  | 55576ff6afdc700ee00664642e19a6fa | https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_50_imagenet_224_224_6.97G_1.2.zip              |
| 48   | tf_resnetv1_101_imagenet_224_224_14.4G_1.2       | 361MB  | 361abf998d29f8849f5a5889226df4ff | https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_101_imagenet_224_224_14.4G_1.2.zip             |
| 49   | tf_resnetv1_152_imagenet_224_224_21.83G_1.2      | 489MB  | 851abf9a7e9656a049ebedb8de96d550 | https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_152_imagenet_224_224_21.83G_1.2.zip            |
| 50   | tf_vgg16_imagenet_224_224_30.96G_1.2             | 1.11GB | a05649569e62d516877a574184f6a1a0 | https://www.xilinx.com/bin/public/openDownload?filename=tf_vgg16_imagenet_224_224_30.96G_1.2.zip                   |
| 51   | tf_vgg19_imagenet_224_224_39.28G_1.2             | 1.15GB | 6d526ca6ac73cc37155272b9434f0c55 | https://www.xilinx.com/bin/public/openDownload?filename=tf_vgg19_imagenet_224_224_39.28G_1.2.zip                   |
| 52   | tf_ssdmobilenetv1_coco_300_300_2.47G_1.2         | 55.8MB | 5394d18fe791484bcc6388767e64b77b | https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdmobilenetv1_coco_300_300_2.47G_1.2.zip               |
| 53   | tf_ssdmobilenetv2_coco_300_300_3.75G_1.2         | 134MB  | 979e324b0f809aa82f3c86a3417865e1 | https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdmobilenetv2_coco_300_300_3.75G_1.2.zip               |
| 54   | tf_ssdresnet50v1_fpn_coco_640_640_178.4G_1.2     | 380MB  | 320b6c8303505960ff9178c030c5a5f1 | https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdresnet50_fpn_coco_640_640_178.4G_1.2.zip             |
| 55   | tf_yolov3_voc_416_416_65.63G_1.2                 | 507MB  | 0e72e4380b581077481dc370edfda986 | https://www.xilinx.com/bin/public/openDownload?filename=tf_yolov3_voc_416_416_65.63G_1.2.zip                       |
| 56   | tf_mlperf_resnet34_coco_1200_1200_433G_1.2       | 239MB  | e338922d736cc60f5feee2a3783514ce | https://www.xilinx.com/bin/public/openDownload?filename=tf_mlperf_resnet34_coco_1200_1200_433G_1.2.zip             |
| 57   | pt_ENet_cityscapes_512_1024_8.6G_1.2             | 3.74MB | b562f491cb8b848354d28230615eb18a | https://www.xilinx.com/bin/public/openDownload?filename=pt_ENet_cityscapes_512_1024_8.6G_1.2.zip                   |
| 58   | pt_SemanticFPN_cityscapes_256_512_10G_1.2        | 118MB  | 04f2ee1e43bea7ecd3d570e94cd03903 | https://www.xilinx.com/bin/public/openDownload?filename=pt_SemanticFPN_cityscapes_256_512_10G_1.2.zip              |
| 59   | pt_facerec-resnet20_mixed_112_96_3.5G_1.2        | 439MB  | 8accf60dc93e0018985c0ab8f02406b4 | https://www.xilinx.com/bin/public/openDownload?filename=pt_facerec-resnet20_mixed_112_96_3.5G_1.2.zip              |
| 60   | pt_face-quality_80_60_61.68M_1.2                 | 5.81MB | b1e8ea14e8c77ab18724b95c7751ae51 | https://www.xilinx.com/bin/public/openDownload?filename=pt_face-quality_80_60_61.68M_1.2.zip                       |
| 61   | pt_MT-resnet18_mixed_320_512_13.65G_1.2          | 67.8MB | 8a77805945daf4dea8a3c80059c6610b | https://www.xilinx.com/bin/public/openDownload?filename=pt_MT-resnet18_mixed_320_512_13.65G_1.2.zip                |
| 62   | pt_facereid-large_96_96_515M_1.2                 | 36.8MB | 94038fcd56521b1f6d7430889ef2db2e | https://www.xilinx.com/bin/public/openDownload?filename=pt_facereid-large_96_96_515M_1.2.zip                       |
| 63   | pt_facereid-small_80_80_90M_1.2                  | 10.0MB | 01ec7ddc8ee374e2488dff0a33082d44 | https://www.xilinx.com/bin/public/openDownload?filename=pt_facereid-small_80_80_90M_1.2.zip                        |
| 64   | pt_personreid_market1501_256_128_4.2G_1.2        | 109MB  | 7a578c089b50dae0a4900877289c8929 | https://www.xilinx.com/bin/public/openDownload?filename=pt_personreid_market1501_256_128_4.2G_1.2.zip              |
| all  | all_models_1.2                                   | 11.3GB | 6766149c79e762a97d8d9c004a216ffe | https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.2.zip                                         |

</details>

### Model Directory Structure
Download and extract the model archive to your working area on the local hard disk. For details on the various models, their download link and MD5 checksum for the zip file of each model, see [Model Download](#model-download).


#### Caffe Model Directory Structure
For a caffe model, you should see the following directory structure:

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
    │    │
    │    ├── Edge                                  # Used for Xilinx edge platform.
    │    │    ├── deploy.caffemodel                # Quantized weights, the output of vai_q_caffe without modification.
    │    │    ├── deploy.prototxt                  # Quantized prototxt, the output of vai_q_caffe without modification.
    │    │    ├── quantized_test.prototxt          # Used to run evaluation with quantized_train_test.caffemodel. 
    │    │    │                                      Some models don't have this file 
    │    │    │                                      if they are converted from Darknet (Yolov2, Yolov3),
    │    │    │                                      Pytorch (ReID) or there is no Caffe Test (Densebox).                           
    │    │    ├── quantized_train_test.caffemodel  # Quantized weights can be used for quantizeded-point training and evaluation.       
    │    │    └── quantized_train_test.prototxt    # Used for quantized-point training and testing with           
    │    │                                           quantized_train_test.caffemodel on GPU when datalayer modified to user's data path.
    │    │
    │    └── Cloud                                 # Used for Xilinx cloud platform.                
    │         ├── deploy.caffemodel                # Quantized weights, uesd for cloud platform.    
    │         └── deploy.prototxt                  # Quantized prototxt with fixed_neuron, used for cloud platform.           
    │                                                 
    └── float                           
         ├── trainval.caffemodel         # Trained float-point weights.
         ├── test.prototxt               # Used to run evaluation with python test codes released in near future.    
         └── trainval.prorotxt           # Used for training and testing with caffe train/test command 
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
    │   ├── deploy_model.pb             # Quantized model for the compiler (extended Tensorflow format).
    │   └── quantize_eval_model.pb      # Quantized model for evaluation.
    │
    └── float                             
        └── float.pb                    # Float-point frozen model, the input to the `vai_q_tensorflow`.The pb name of different models 
                                          may be different. Path and name in test scripts could be modified according to actual
                                          situation.                               


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
    │   ├── _int.xmodel                 # Deployed model.  
    │   ├── _int.py                     # Converted vai_q_pytorch format model.
    │   └── quant_info.json             # Quantization steps of tensors got. Please keep it for evaluation of quantized model.
    │                                           
    └── float                           
        └── _int.pth                    # Trained float-point model. The pth name of different models may be different.
                                          Path and name in test scripts could be modified according to actual situation.
                                          
                                         
                                          
**Note:** For more information on `vai_q_caffe` , `vai_q_tensorflow`and`vai_q_pytorch`, see the [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf).


## Model Performance
All the models in the Model Zoo have been deployed on Xilinx hardware with [Vitis AI](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Library). The performance number including end-to-end throughput and latency for each model on various boards with different DPU configurations are listed in the following sections.

For more information about DPU, see [DPU IP Product Guide](https://www.xilinx.com/cgi-bin/docs/ipdoc?c=dpu;v=latest;d=pg338-dpu.pdf).


**Note:** The model performance number listed in the following sections is generated with Vitis AI v1.2 and Vitis AI Lirary v1.2. For each board, a different DPU configuration is used. Vitis AI and Vitis AI Library can be downloaded for free from [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library Github](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Library).
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
Measured with Vitis AI 1.2 and Vitis AI Library 1.2  

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-05)` board with a `3 * B4096  @ 281MHz   V1.4.1` DPU configuration:


| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 13.60                            | 73.5                                  | 152.7                                |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 5.35                             | 186.9                                 | 441.6                                |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 5.62                             | 178.0                                 | 411.7                                |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 6.92                             | 144.4                                 | 317.3                                |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 17.38                            | 57.5                                  | 128.1                                |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 35.06                            | 28.5                                  | 66.2                                 |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G        | 4.41                             | 226.8                                 | 548.0                                |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 3.76                             | 265.8                                 | 1012.3                               |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 13.10                            | 76.3                                  | 282.6                                |
| 10   | refinedet\_baseline        | cf_refinedet_coco_480_360_123G_1.2           | 120.96                           | 8.3                                   | 24.4                                 |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G            | 31.58                            | 31.7                                  | 101.3                                |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G        | 16.68                            | 59.9                                  | 196.8                                |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G         | 12.07                            | 82.9                                  | 276.2                                |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G             | 12.05                            | 82.9                                  | 279.7                                |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G              | 18.33                            | 54.5                                  | 201.5                                |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 9.56                             | 104.5                                 | 381.4                                |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G          | 26.01                            | 38.4                                  | 114.4                                |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G               | 16.74                            | 59.7                                  | 175.5                                |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G          | 2.62                             | 381.6                                 | 1317.4                               |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G  | 285.58                           | 3.5                                   | 15.1                                 |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 2.56                             | 390.0                                 | 1172.3                               |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G              | 4.99                             | 200.4                                 | 588.7                                |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G               | 1.18                             | 849.4                                 | 1382.7                               |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G              | 2.74                             | 364.2                                 | 665.6                                |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G               | 28.19                            | 35.5                                  | 127.7                                |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 77.10                            | 13.0                                  | 34.3                                 |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 11.89                            | 84.1                                  | 229.7                                |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 73.76                            | 13.5                                  | 35.3                                 |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                    | 37.31                            | 26.8                                  | 71.0                                 |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G            | 15.82                            | 63.2                                  | 185.9                                |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G             | 13.73                            | 72.8                                  | 214.8                                |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G             | 11.73                            | 85.2                                  | 258.7                                |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G              | 5.98                             | 167.1                                 | 320.6                                |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G               | 13.69                            | 73.0                                  | 173.0                                |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G            | 81.86                            | 12.2                                  | 40.3                                 |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G             | 2.00                             | 500.0                                 | 1792.2                               |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G            | 8.82                             | 113.4                                 | 383.2                                |
| 38   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G | 43.19                            | 23.1                                  | 48.7                                 |
| 39   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G           | 5.43                             | 184.1                                 | 423.9                                |
| 40   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G       | 17.44                            | 57.3                                  | 126.7                                |
| 41   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G       | 35.08                            | 28.5                                  | 66.2                                 |
| 42   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M  | 0.85                             | 1170.7                                | 4043.5                               |
| 43   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M  | 1.41                             | 707.6                                 | 2007.1                               |
| 44   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G    | 3.52                             | 284.3                                 | 754.9                                |
| 45   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G    | 4.33                             | 230.8                                 | 568.4                                |
| 46   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G    | 5.98                             | 167.3                                 | 393.1                                |
| 47   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G        | 12.63                            | 79.1                                  | 161.9                                |
| 48   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G       | 23.16                            | 43.1                                  | 91.3                                 |
| 49   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G      | 33.82                            | 29.6                                  | 63.7                                 |
| 50   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G             | 49.78                            | 20.1                                  | 40.9                                 |
| 51   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G             | 57.72                            | 17.3                                  | 36.5                                 |
| 52   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G         | 11.09                            | 90.1                                  | 332.9                                |
| 53   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G         | 15.64                            | 63.9                                  | 193.2                                |
| 54   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 757.63                           | 1.3                                   | 5.1                                  |
| 55   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                 | 74.22                            | 13.5                                  | 35.0                                 |
| 56   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G       | 502.41                           | 2.0                                   | 7.2                                  |

</details>


### Performance on ZCU104
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 300MHz   V1.4.1` DPU configuration:


| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 12.70                            | 78.7                                   | 148.1                                |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 5.07                             | 197.4                                 | 411.1                                |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 5.24                             | 190.8                                 | 389.4                                |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 6.51                             | 153.5                                 | 302.1                                |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 16.43                            | 60.8                                   | 117.9                                |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 33.01                            | 30.3                                   | 58.3                                 |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G        | 4.12                             | 242.8                                 | 520.8                                |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 3.69                             | 271.0                                 | 943.6                                |
| 9    | ssd_pedestrian_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 12.73                            | 78.5                                   | 220.5                                |
| 10   | refinedet\_baseline        | cf_refinedet_coco_480_360_123G               | 115.49                           | 8.7                                   | 18.2                                 |
| 11   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G            | 30.69                            | 32.6                                   | 75.9                                 |
| 12   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G        | 16.31                            | 61.3                                   | 154.1                                |
| 13   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G         | 11.95                            | 83.7                                   | 228.4                                |
| 14   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G             | 11.77                            | 84.9                                   | 231.9                                |
| 15   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G              | 17.77                            | 56.2                                   | 152.9                                |
| 16   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 9.31                             | 107.3                                 | 354.9                                |
| 17   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G          | 38.90                            | 25.7                                   | 101.4                                |
| 18   | FPN                        | cf_fpn_cityscapes_256_512_8.9G               | 16.16                            | 61.9                                   | 169.4                                |
| 19   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G          | 2.02                             | 494.4                                 | 1209.8                               |
| 20   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G  | 273.83                           | 3.7                                   | 10.9                                 |
| 21   | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 2.52                             | 397.3                                 | 1263.9                               |
| 22   | densebox_640_360           | cf_densebox_wider_360_640_1.11G              | 4.90                             | 204.1                                 | 621.9                                |
| 23   | face_landmark              | cf_landmark_celeba_96_72_0.14G               | 1.12                             | 891.4                                 | 1449.5                               |
| 24   | reid                       | cf_reid_market1501_160_80_0.95G              | 2.58                             | 387.7                                 | 700.3                                |
| 25   | multi_task                 | cf_multitask_bdd_288_512_14.8G               | 27.75                            | 36.0                                   | 109.1                                |
| 26   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 73.57                            | 13.6                                   | 28.6                                 |
| 27   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 11.73                            | 85.2                                   | 221.5                                |
| 28   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 70.19                            | 14.2                                   | 29.5                                 |
| 29   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                    | 35.21                            | 28.4                                   | 58.7                                 |
| 30   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G            | 15.01                            | 66.6                                   | 152.9                                |
| 31   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G             | 13.05                            | 76.6                                   | 179.5                                |
| 32   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G             | 11.18                            | 89.4                                   | 216.1                                |
| 33   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G              | 5.63                             | 177.6                                 | 309.0                                |
| 34   | ResNet64-face              | cf_facerec-resnet64_112_96_11G               | 12.86                            | 77.7                                   | 147.5                                |
| 35   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G            | 78.71                            | 12.7                                   | 31.5                                 |
| 36   | plate detection            | cf_plate-detection_320_320_0.49G             | 1.99                             | 501.8                                 | 1761.2                               |
| 37   | plate recognition          | cf_plate-recognition_96_288_1.75G            | 4.44                             | 225.3                                 | 541.2                                |
| 38   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G | 40.00                            | 25.0                                   | 46.2                                 |
| 39   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G           | 5.06                             | 197.5                                 | 403.1                                |
| 40   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G       | 16.49                            | 60.6                                   | 117.4                                |
| 41   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G       | 33.03                            | 30.3                                   | 58.4                                 |
| 42   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M  | 0.83                             | 1197.2                                 | 3744.2                               |
| 43   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M  | 1.35                             | 737.6                                 | 1941.7                               |
| 44   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G    | 3.23                             | 309.1                                 | 719.3                                |
| 45   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G    | 4.08                             | 244.7                                 | 529.2                                |
| 46   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G    | 5.56                             | 179.9                                 | 370.7                                |
| 47   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G        | 11.82                            | 84.6                                   | 158.3                                |
| 48   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G       | 21.68                            | 46.1                                   | 86.2                                 |
| 49   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G      | 31.58                            | 31.6                                   | 59.1                                 |
| 50   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G             | 46.84                            | 21.3                                   | 37.0                                 |
| 51   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G             | 54.24                            | 18.4                                   | 32.6                                 |
| 52   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G         | 10.72                            | 93.2                                   | 294.6                                |
| 53   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G         | 15.07                            | 66.3                                   | 185.4                                |
| 54   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 728.24                           | 1.4                                   | 5.2                                  |
| 55   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                 | 70.63                            | 14.1                                   | 29.2                                 |
| 56   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G       | 575.68                           | 1.7                                   | 5.4                                  |

</details>


### Performance on U50
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 6 DPUv3E kernels running at 300Mhz in Gen3x4:
  

| No\. | Model                          | Name                                         | Frequency \(MHz\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :----------------------------- | :------------------------------------------- | ----------------- | ------------------------------------ |
| 1    | resnet50                       | cf_resnet50_imagenet_224_224_7.7G            | 300               | 631.21                               |
| 2    | resnet18                       | cf_resnet18_imagenet_224_224_3.65G           | 300               | 1430.05                               |
| 3    | Inception_v1                   | cf_inceptionv1_imagenet_224_224_3.16G        | 300               | 1183.30                               |
| 4    | Inception_v2                   | cf_inceptionv2_imagenet_224_224_4G           | 300               | 983.55                               |
| 5    | Inception_v3                   | cf_inceptionv3_imagenet_299_299_11.4G        | 300               | 405.39                               |
| 6    | Inception_v4                   | cf_inceptionv4_imagenet_299_299_24.5G        | 300               | 187.72                               |
| 7    | SqueezeNet                     | cf_squeeze_imagenet_227_227_0.76G            | 300               | 3016.11                               |
| 8    | ssd_pedestrian_pruned_0_97     | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 300               | 621.52                               |
| 9    | refinedet_baseline             | cf_refinedet_coco_480_360_123G               | 270               | 49.99                                 |
| 10   | refinedet_pruned_0_8           | cf_refinedet_coco_360_480_0.8_25G            | 270               | 193.63                               |
| 11   | refinedet_pruned_0_92          | cf_refinedet_coco_360_480_0.92_10.10G        | 270               | 420.59                               |
| 12   | refinedet_pruned_0_96          | cf_refinedet_coco_360_480_0.96_5.08G         | 270               | 617.22                               |
| 13   | ssd_adas_pruned_0_95           | cf_ssdadas_bdd_360_480_0.95_6.3G             | 300               | 628.96                               |
| 14   | ssd_traffic_pruned_0_9         | cf_ssdtraffic_360_480_0.9_11.6G              | 300               | 432.99                               |
| 15   | VPGnet_pruned_0_99             | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 300               | 478.81                               |
| 16   | FPN                            | cf_fpn_cityscapes_256_512_8.9G               | 300               | 450.91                               |
| 17   | SP_net                         | cf_SPnet_aichallenger_224_128_0.54G          | 300               | 1158.46                               |
| 18   | Openpose_pruned_0_3            | cf_openpose_aichallenger_368_368_0.3_189.7G  | 270               | 29.15                                 |
| 19   | densebox_320_320               | cf_densebox_wider_320_320_0.49G              | 300               | 1929.24                               |
| 20   | densebox_640_360               | cf_densebox_wider_360_640_1.11G              | 300               | 877.25                               |
| 21   | face_landmark                  | cf_landmark_celeba_96_72_0.14G               | 300               | 8513.72                               |
| 22   | reid                           | cf_reid_market1501_160_80_0.95G              | 300               | 3612.90                               |
| 23   | multi_task                     | cf_multitask_bdd_288_512_14.8G               | 300               | 237.28                               |
| 24   | yolov3_bdd                     | dk_yolov3_bdd_288_512_53.7G                  | 270               | 77.22                                 |
| 25   | yolov3_adas_pruned_0_9         | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 270               | 642.92                               |
| 26   | yolov3_voc                     | dk_yolov3_voc_416_416_65.42G                 | 270               | 78.96                                 |
| 27   | yolov2_voc                     | dk_yolov2_voc_448_448_34G                    | 270               | 165.56                               |
| 28   | yolov2_voc_pruned_0_66         | dk_yolov2_voc_448_448_0.66_11.56G            | 270               | 409.60                               |
| 29   | yolov2_voc_pruned_0_71         | dk_yolov2_voc_448_448_0.71_9.86G             | 270               | 481.52                               |
| 30   | yolov2_voc_pruned_0_77         | dk_yolov2_voc_448_448_0.77_7.82G             | 270               | 585.39                               |
| 31   | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G              | 300               | 1278.33                               |
| 32   | ResNet64-face                  | cf_facerec-resnet64_112_96_11G               | 300               | 495.71                               |
| 33   | FPN_Res18_Medical_segmentation | cf_FPN-resnet18_EDD_320_320_45.3G            | 300               | 103.08                               |
| 34   | plate detection                | cf_plate-detection_320_320_0.49G             | 300               | 5135.77                               |
| 35   | Inception_resnet_v2            | tf_inceptionresnetv2_imagenet_299_299_26.35G | 300               | 172.92                               |
| 36   | Inception_v1                   | tf_inceptionv1_imagenet_224_224_3G           | 300               | 1195.77                               |
| 37   | Inception_v3                   | tf_inceptionv3_imagenet_299_299_11.45G       | 300               | 398.48                               |
| 38   | Inception_v4                   | tf_inceptionv4_imagenet_299_299_24.55G       | 300               | 187.60                               |
| 39   | resnet_v1_50                   | tf_resnetv1_50_imagenet_224_224_6.97G        | 300               | 703.82                               |
| 40   | resnet_v1_101                  | tf_resnetv1_101_imagenet_224_224_14.4G       | 300               | 365.12                               |
| 41   | resnet_v1_152                  | tf_resnetv1_152_imagenet_224_224_21.83G      | 300               | 244.71                               |
| 42   | vgg_16                         | tf_vgg16_imagenet_224_224_30.96G             | 300               | 164.70                               |
| 43   | vgg_19                         | tf_vgg19_imagenet_224_224_39.28G             | 300               | 136.96                               |
| 44   | ssd_resnet_50_v1_fpn           | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 270               | 32.68                                 |
| 45   | yolov3_voc                     | tf_yolov3_voc_416_416_65.63G                 | 270               | 79.25                                 |
| 46   | torchvision                    | resnet50                                     | 300               | 546.43                               |
| 47   | torchvision                    | inception_v3                                 | 300               | 405.31                               |
| 48   | torchvision                    | squeezenet                                   | 300               | 2024.36                               |


</details>


### Performance on U50 lv9e
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 9 DPUv3E kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                          | Name                                         | Frequency \(MHz\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :----------------------------- | :------------------------------------------- | ----------------- | ------------------------------------ |
| 1    | resnet50                       | cf_resnet50_imagenet_224_224_7.7G            | 275               | 802.45                               |
| 2    | resnet18                       | cf_resnet18_imagenet_224_224_3.65G           | 275               | 1927.42                               |
| 3    | Inception_v1                   | cf_inceptionv1_imagenet_224_224_3.16G        | 275               | 1565.29                               |
| 4    | Inception_v2                   | cf_inceptionv2_imagenet_224_224_4G           | 275               | 1289.06                               |
| 5    | Inception_v3                   | cf_inceptionv3_imagenet_299_299_11.4G        | 275               | 552.39                               |
| 6    | Inception_v4                   | cf_inceptionv4_imagenet_299_299_24.5G        | 275               | 256.21                               |
| 7    | SqueezeNet                     | cf_squeeze_imagenet_227_227_0.76G            | 275               | 3767.09                               |
| 8    | ssd_pedestrian_pruned_0_97     | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 275               | 664.20                               |
| 9    | refinedet_baseline             | cf_refinedet_coco_480_360_123G               | 275               | 70.47                                 |
| 10   | refinedet_pruned_0_8           | cf_refinedet_coco_360_480_0.8_25G            | 275               | 235.60                               |
| 11   | refinedet_pruned_0_92          | cf_refinedet_coco_360_480_0.92_10.10G        | 275               | 514.70                               |
| 12   | refinedet_pruned_0_96          | cf_refinedet_coco_360_480_0.96_5.08G         | 275               | 725.78                               |
| 13   | ssd_adas_pruned_0_95           | cf_ssdadas_bdd_360_480_0.95_6.3G             | 275               | 714.96                               |
| 14   | ssd_traffic_pruned_0_9         | cf_ssdtraffic_360_480_0.9_11.6G              | 275               | 483.31                               |
| 15   | VPGnet_pruned_0_99             | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 275               | 595.30                               |
| 16   | FPN                            | cf_fpn_cityscapes_256_512_8.9G               | 247.5             | 530.88                               |
| 17   | SP_net                         | cf_SPnet_aichallenger_224_128_0.54G          | 275               | 2687.68                               |
| 18   | Openpose_pruned_0_3            | cf_openpose_aichallenger_368_368_0.3_189.7G  | 275               | 43.30                                 |
| 19   | densebox_320_320               | cf_densebox_wider_320_320_0.49G              | 275               | 2431.22                               |
| 20   | densebox_640_360               | cf_densebox_wider_360_640_1.11G              | 275               | 1074.43                               |
| 21   | face_landmark                  | cf_landmark_celeba_96_72_0.14G               | 275               | 11759.40                             |
| 22   | reid                           | cf_reid_market1501_160_80_0.95G              | 275               | 5013.91                               |
| 23   | multi_task                     | cf_multitask_bdd_288_512_14.8G               | 275               | 192.22                               |
| 24   | yolov3_bdd                     | dk_yolov3_bdd_288_512_53.7G                  | 247.5             | 102.96                               |
| 25   | yolov3_adas_pruned_0_9         | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 247.5             | 810.03                               |
| 26   | yolov3_voc                     | dk_yolov3_voc_416_416_65.42G                 | 247.5             | 104.24                               |
| 27   | yolov2_voc                     | dk_yolov2_voc_448_448_34G                    | 247.5             | 227.47                               |
| 28   | yolov2_voc_pruned_0_66         | dk_yolov2_voc_448_448_0.66_11.56G            | 247.5             | 565.18                               |
| 29   | yolov2_voc_pruned_0_71         | dk_yolov2_voc_448_448_0.71_9.86G             | 247.5             | 662.60                               |
| 30   | yolov2_voc_pruned_0_77         | dk_yolov2_voc_448_448_0.77_7.82G             | 247.5             | 807.82                               |
| 31   | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G              | 275               | 1760.85                               |
| 32   | ResNet64-face                  | cf_facerec-resnet64_112_96_11G               | 275               | 663.68                               |
| 33   | FPN_Res18_Medical_segmentation | cf_FPN-resnet18_EDD_320_320_45.3G            | 275               | 140.17                               |
| 34   | plate detection                | cf_plate-detection_320_320_0.49G             | 275               | 5563.75                               |
| 35   | Inception_resnet_v2            | tf_inceptionresnetv2_imagenet_299_299_26.35G | 275               | 224.08                               |
| 36   | Inception_v1                   | tf_inceptionv1_imagenet_224_224_3G           | 275               | 1607.43                               |
| 37   | Inception_v3                   | tf_inceptionv3_imagenet_299_299_11.45G       | 275               | 549.69                               |
| 38   | Inception_v4                   | tf_inceptionv4_imagenet_299_299_24.55G       | 275               | 256.50                               |
| 39   | resnet_v1_50                   | tf_resnetv1_50_imagenet_224_224_6.97G        | 275               | 880.59                               |
| 40   | resnet_v1_101                  | tf_resnetv1_101_imagenet_224_224_14.4G       | 275               | 458.12                               |
| 41   | resnet_v1_152                  | tf_resnetv1_152_imagenet_224_224_21.83G      | 275               | 305.93                               |
| 42   | vgg_16                         | tf_vgg16_imagenet_224_224_30.96G             | 275               | 228.86                               |
| 43   | vgg_19                         | tf_vgg19_imagenet_224_224_39.28G             | 275               | 189.88                               |
| 44   | ssd_resnet_50_v1_fpn           | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 247.5             | 42.59                                 |
| 45   | yolov3_voc                     | tf_yolov3_voc_416_416_65.63G                 | 247.5             | 103.98                               |
| 46   | torchvision                    | resnet50                                     | 275               | 768.14                               |
| 47   | torchvision                    | inception_v3                                 | 275               | 551.55                               |
| 48   | torchvision                    | squeezenet                                   | 275               | 2540.55                               |


</details>


### Performance on U50 lv10e
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 10 DPUv3E kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                          | Name                                        | Frequency \(MHz\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :----------------------------- | :------------------------------------------ | ----------------- | ------------------------------------ |
| 1    | resnet50                       | cf_resnet50_imagenet_224_224_7.7G           | 247.5             | 802.46                                |
| 2    | resnet18                       | cf_resnet18_imagenet_224_224_3.65G          | 247.5             | 1934.48                               |
| 3    | Inception_v1                   | cf_inceptionv1_imagenet_224_224_3.16G       | 247.5             | 1536.64                               |
| 4    | Inception_v2                   | cf_inceptionv2_imagenet_224_224_4G          | 247.5             | 1313.99                               |
| 5    | SqueezeNet                     | cf_squeeze_imagenet_227_227_0.76G           | 247.5             | 3451.05                               |
| 6    | ssd_pedestrian_pruned_0_97     | cf_ssdpedestrian_coco_360_640_0.97_5.9G     | 247.5             | 755.24                                |
| 7    | refinedet_pruned_0_8           | cf_refinedet_coco_360_480_0.8_25G           | 247.5             | 273.79                                |
| 8    | refinedet_pruned_0_92          | cf_refinedet_coco_360_480_0.92_10.10G       | 247.5             | 574.76                                |
| 9    | refinedet_pruned_0_96          | cf_refinedet_coco_360_480_0.96_5.08G        | 247.5             | 795.12                                |
| 10   | ssd_adas_pruned_0_95           | cf_ssdadas_bdd_360_480_0.95_6.3G            | 247.5             | 818.22                                |
| 11   | ssd_traffic_pruned_0_9         | cf_ssdtraffic_360_480_0.9_11.6G             | 247.5             | 570.84                                |
| 12   | VPGnet_pruned_0_99             | cf_VPGnet_caltechlane_480_640_0.99_2.5G     | 275               | 658.99                                |
| 13   | FPN                            | cf_fpn_cityscapes_256_512_8.9G              | 247.5             | 552.17                                |
| 14   | SP_net                         | cf_SPnet_aichallenger_224_128_0.54G         | 275               | 1706.95                              |
| 15   | Openpose_pruned_0_3            | cf_openpose_aichallenger_368_368_0.3_189.7G | 220               | 39.68                                |
| 16   | densebox_320_320               | cf_densebox_wider_320_320_0.49G             | 275               | 2572.69                              |
| 17   | densebox_640_360               | cf_densebox_wider_360_640_1.11G             | 275               | 1125.09                              |
| 18   | face_landmark                  | cf_landmark_celeba_96_72_0.14G              | 275               | 12917.20                              |
| 19   | reid                           | cf_reid_market1501_160_80_0.95G             | 275               | 5548.10                               |
| 20   | multi_task                     | cf_multitask_bdd_288_512_14.8G              | 247.5             | 176.96                                |
| 21   | yolov3_bdd                     | dk_yolov3_bdd_288_512_53.7G                 | 220               | 100.58                                |
| 22   | yolov3_adas_pruned_0_9         | dk_yolov3_cityscapes_256_512_0.9_5.46G      | 220               | 771.32                                |
| 23   | yolov3_voc                     | dk_yolov3_voc_416_416_65.42G                | 220               | 102.19                                |
| 24   | yolov2_voc                     | dk_yolov2_voc_448_448_34G                   | 220               | 223.30                                |
| 25   | yolov2_voc_pruned_0_66         | dk_yolov2_voc_448_448_0.66_11.56G           | 220               | 547.63                                |
| 26   | yolov2_voc_pruned_0_71         | dk_yolov2_voc_448_448_0.71_9.86G            | 220               | 639.09                                |
| 27   | yolov2_voc_pruned_0_77         | dk_yolov2_voc_448_448_0.77_7.82G            | 220               | 770.95                                |
| 28   | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G             | 275               | 1943.44                               |
| 29   | ResNet64-face                  | cf_facerec-resnet64_112_96_11G              | 275               | 736.43                                |
| 30   | FPN_Res18_Medical_segmentation | cf_FPN-resnet18_EDD_320_320_45.3G           | 247.5             | 139.76                                |
| 31   | plate detection                | cf_plate-detection_320_320_0.49G            | 275               | 5521.41                               |
| 32   | Inception_v1                   | tf_inceptionv1_imagenet_224_224_3G          | 247.5             | 1552.49                               |
| 33   | resnet_v1_50                   | tf_resnetv1_50_imagenet_224_224_6.97G       | 247.5             | 882.28                                |
| 34   | resnet_v1_101                  | tf_resnetv1_101_imagenet_224_224_14.4G      | 247.5             | 458.93                                |
| 35   | resnet_v1_152                  | tf_resnetv1_152_imagenet_224_224_21.83G     | 247.5             | 306.47                                |
| 36   | vgg_16                         | tf_vgg16_imagenet_224_224_30.96G            | 247.5             | 229.26                                |
| 37   | vgg_19                         | tf_vgg19_imagenet_224_224_39.28G            | 247.5             | 189.91                                |
| 38   | ssd_resnet_50_v1_fpn           | tf_ssdresnet50v1_fpn_coco_640_640_178.4G    | 220               | 41.84                                 |
| 39   | yolov3_voc                     | tf_yolov3_voc_416_416_65.63G                | 220               | 102.40                                |
| 40   | torchvision                    | resnet50                                    | 247.5             | 764.57                                |
| 41   | torchvision                    | squeezenet                                  | 247.5             | 2393.23                               |


</details>


### Performance on U200
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U200` board with 2 DPUv1 kernels running at 350Mhz with xilinx_u200_xdma_201830_2 shell:
  

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
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U250` board with 4 DPUv1 kernels running at 350Mhz with xilinx_u250_xdma_201830_1 shell:
  

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
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U280` board with 14 DPUv3E kernels running at 300Mhz in Gen3x4:
  

| No\. | Model                          | Name                                         | Frequency \(MHz\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :----------------------------- | :------------------------------------------- | ----------------- | ------------------------------------ |
| 1    | resnet50                       | cf_resnet50_imagenet_224_224_7.7G            | 210               | 918.07                               |
| 2    | resnet18                       | cf_resnet18_imagenet_224_224_3.65G           | 150               | 1634.40                               |
| 3    | Inception_v1                   | cf_inceptionv1_imagenet_224_224_3.16G        | 150               | 1169.53                               |
| 4    | Inception_v2                   | cf_inceptionv2_imagenet_224_224_4G           | 150               | 937.03                               |
| 5    | Inception_v3                   | cf_inceptionv3_imagenet_299_299_11.4G        | 150               | 371.96                               |
| 6    | Inception_v4                   | cf_inceptionv4_imagenet_299_299_24.5G        | 150               | 167.02                               |
| 7    | SqueezeNet                     | cf_squeeze_imagenet_227_227_0.76G            | 150               | 2821.66                               |
| 8    | ssd_pedestrian_pruned_0_97     | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 150               | 423.29                               |
| 9    | ssd_adas_pruned_0_95           | cf_ssdadas_bdd_360_480_0.95_6.3G             | 150               | 476.11                               |
| 10   | ssd_traffic_pruned_0_9         | cf_ssdtraffic_360_480_0.9_11.6G              | 150               | 306.03                               |
| 11   | VPGnet_pruned_0_99             | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 150               | 567.75                               |
| 12   | FPN                            | cf_fpn_cityscapes_256_512_8.9G               | 150               | 362.94                               |
| 13   | SP_net                         | cf_SPnet_aichallenger_224_128_0.54G          | 150               | 2126.61                               |
| 14   | Openpose_pruned_0_3            | cf_openpose_aichallenger_368_368_0.3_189.7G  | 150               | 36.54                                 |
| 15   | densebox_320_320               | cf_densebox_wider_320_320_0.49G              | 150               | 2                                     |
| 16   | densebox_640_360               | cf_densebox_wider_360_640_1.11G              | 150               | 1138.79                               |
| 17   | face_landmark                  | cf_landmark_celeba_96_72_0.14G               | 150               | 11302.40                             |
| 18   | reid                           | cf_reid_market1501_160_80_0.95G              | 150               | 4608.02                               |
| 19   | multi_task                     | cf_multitask_bdd_288_512_14.8G               | 150               | 128.32                               |
| 20   | yolov3_bdd                     | dk_yolov3_bdd_288_512_53.7G                  | 180               | 108.62                               |
| 21   | yolov3_adas_pruned_0_9         | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 180               | 893.08                               |
| 22   | yolov3_voc                     | dk_yolov3_voc_416_416_65.42G                 | 180               | 113.56                               |
| 23   | yolov2_voc_pruned_0_66         | dk_yolov2_voc_448_448_0.66_11.56G            | 150               | 490.34                               |
| 24   | yolov2_voc_pruned_0_71         | dk_yolov2_voc_448_448_0.71_9.86G             | 150               | 570.32                               |
| 25   | yolov2_voc_pruned_0_77         | dk_yolov2_voc_448_448_0.77_7.82G             | 150               | 679.56                               |
| 26   | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G              | 150               | 1576.89                               |
| 27   | ResNet64-face                  | cf_facerec-resnet64_112_96_11G               | 150               | 575.43                               |
| 28   | FPN_Res18_Medical_segmentation | cf_FPN-resnet18_EDD_320_320_45.3G            | 150               | 104.89                               |
| 29   | plate detection                | cf_plate-detection_320_320_0.49G             | 150               | 4235.68                               |
| 30   | Inception_resnet_v2            | tf_inceptionresnetv2_imagenet_299_299_26.35G | 150               | 150.08                               |
| 31   | Inception_v1                   | tf_inceptionv1_imagenet_224_224_3G           | 150               | 1117.86                               |
| 32   | Inception_v3                   | tf_inceptionv3_imagenet_299_299_11.45G       | 150               | 371.75                               |
| 33   | Inception_v4                   | tf_inceptionv4_imagenet_299_299_24.55G       | 150               | 167.97                               |
| 34   | resnet_v1_50                   | tf_resnetv1_50_imagenet_224_224_6.97G        | 180               | 890.26                               |
| 35   | resnet_v1_101                  | tf_resnetv1_101_imagenet_224_224_14.4G       | 150               | 387.53                               |
| 36   | resnet_v1_152                  | tf_resnetv1_152_imagenet_224_224_21.83G      | 150               | 258.93                               |
| 37   | vgg_16                         | tf_vgg16_imagenet_224_224_30.96G             | 150               | 182.70                               |
| 38   | vgg_19                         | tf_vgg19_imagenet_224_224_39.28G             | 150               | 153.15                               |
| 39   | ssd_resnet_50_v1_fpn           | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 150               | 28.81                                 |
| 40   | yolov3_voc                     | tf_yolov3_voc_416_416_65.63G                 | 180               | 112.37                               |
| 41   | torchvision                    | resnet50                                     | 210               | 878.38                               |
| 42   | torchvision                    | inception_v3                                 | 150               | 371.01                               |
| 43   | torchvision                    | squeezenet                                   | 150               | 1655.73                               |

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

* [GitHub Issues](https://github.com/Xilinx/TechDocs/issues)
* [Forum](https://forums.xilinx.com/t5/Deephi-DNNDK/bd-p/Deephi)
* <a href="mailto:xilinx_ai_model_zoo@xilinx.com">Email</a>

You can also submit a pull request with details on how to improve the product. Prior to submitting your pull request, ensure that you can build the product and run all the demos with your patch. In case of a larger feature, provide a relevant demo.

## License

Xilinx AI Model Zoo is licensed under [Apache License Version 2.0](reference-files/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

<hr/>
<p align="center"><sup>Copyright&copy; 2019 Xilinx</sup></p>
