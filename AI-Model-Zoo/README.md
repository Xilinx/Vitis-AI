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
2.The variety and quantity of Caffe models have been improved, newly added such as license plate recognition and medical segmentation models.</br>
3.Updated information about more supported edge and cloud hardware platforms and performance.

## Model Information
The following table includes comprehensive information about each model, including application, framework, training and validation dataset, backbone, input size, computation as well as float and quantized precision.<br>
At present, most of the python scripts we provided are compatible with python2/3, except a few models(No.20~22 and 24)need python2 environment.

<details>
 <summary><b>Click here to view details</b></summary>

| No\. | Application              | Model                          | Name                                                      | Framework  | Backbone       | Input Size | OPS per image | Training Set                            | Val Set                 | Float \(Top1, Top5\)/ mAP/mIoU                               | Quantized \(Top1, Top5\)/mAP/mIoU                            |
| :--- | :----------------------- | :----------------------------- | :-------------------------------------------------------- | :--------- | -------------- | ---------- | ------------- | --------------------------------------- | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Image Classification     | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G\_1\.2             | caffe      | resnet50       | 224\*224   | 7\.7G         | ImageNet Train                          | ImageNet Val            | 0\.7444/0\.9185                                              | 0\.7334/0\.9131                                              |
| 2    | Image Classifiction      | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G\_1\.2            | caffe      | resnet18       | 224\*224   | 3\.65G        | ImageNet Train                          | ImageNet Val            | 0\.6832/0\.8848                                              | 0.66\.94%/88\.25%                                            |
| 3    | Image Classification     | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G\_1\.2         | caffe      | inception\_v1  | 224\*224   | 3\.16G        | ImageNet Train                          | ImageNet Val            | 0\.7030/0\.8971                                              | 0\.6984/0\.8942                                              |
| 4    | Image Classification     | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G\_1\.2             | caffe      | bn\-inception  | 224\*224   | 4G            | ImageNet Train                          | ImageNet Val            | 0\.7275/0\.9111                                              | 0\.7168/0\.9029                                              |
| 5    | Image Classification     | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G\_1\.2         | caffe      | inception\_v3  | 299\*299   | 11\.4G        | ImageNet Train                          | ImageNet Val            | 0\.7701/0\.9329                                              | 0\.7626/0\.9303                                              |
| 6    | Image Classification     | Inception\_v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G\_1\.2         | caffe      | inception\_v3  | 299\*299   | 24\.5G        | ImageNet Train                          | ImageNet Val            | 0\.7958/0\.9470                                              | 0\.7898/0\.9445                                              |
| 7    | Image Classification     | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G\_1\.2         | caffe      | MobileNet\_v2  | 224\*224   | 608M          | ImageNet Train                          | ImageNet Val            | 0\.6475/0\.8609                                              | 0\.6354/0\.8506                                              |
| 8    | Image Classifiction      | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G\_1\.2             | caffe      | squeezenet     | 227\*227   | 0\.76G        | ImageNet Train                          | ImageNet Val            | 0\.5438/0\.7813                                              | 0\.5026/0\.7658                                              |
| 9    | ADAS Pedstrain Detection | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G\_1\.2     | caffe      | VGG\-bn\-16    | 360\*640   | 5\.9G         | coco2014\_train\_person and crowndhuman | coco2014\_val\_person   | 0\.5903                                                      | 0\.5876                                                      |
| 10   | Object Detection         | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G\_1\.2            | caffe      | VGG\-bn\-16    | 360\*480   | 25G           | coco2014\_train\_person                 | coco2014\_val\_person   | 0\.6794                                                      | 0\.6780                                                      |
| 11   | Object Detection         | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G\_1\.2       | caffe      | VGG\-bn\-16    | 360\*480   | 10\.10G       | coco2014\_train\_person                 | coco2014\_val\_person   | 0\.6489                                                      | 0\.6486                                                      |
| 12   | Object Detection         | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G\_1\.2        | caffe      | VGG\-bn\-16    | 360\*480   | 5\.08G        | coco2014\_train\_person                 | coco2014\_val\_person   | 0\.6120                                                      | 0\.6113                                                      |
| 13   | ADAS Vehicle Detection   | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G\_1\.2            | caffe      | VGG\-16        | 360\*480   | 6\.3G         | bdd100k \+ private data                 | bdd100k \+ private data | 0\.4207                                                      | 0\.4200                                                      |
| 14   | Traffic Detection        | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G\_1\.2              | caffe      | VGG\-16        | 360\*480   | 11\.6G        | private data                            | private data            | 0\.5982                                                      | 0\.5921                                                      |
| 15   | ADAS Lane Detection      | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G\_1\.2     | caffe      | VGG            | 480\*640   | 2\.5G         | caltech\-lanes\-train\-dataset          | caltech lane            | 0\.8864\(F1\-score\)                                         | 0\.8882\(F1\-score\)                                         |
| 16   | Object Detection         | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G\_1\.2           | caffe      | MobileNet\_v2  | 360\*480   | 6\.57G        | bdd100k train                           | bdd100k val             | 0\.3052                                                      | 0\.2752                                                      |
| 17   | ADAS Segmentation        | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G\_1\.2                | caffe      | Google\_v1\_BN | 256\*512   | 8\.9G         | Cityscapes gtFineTrain                  | Cityscapes Val          | 0\.5669                                                      | 0\.5662                                                      |
| 18   | Pose Estimation          | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G\_1\.2           | caffe      | Google\_v1\_BN | 128\*224   | 548\.6M       | ai\_challenger                          | ai\_challenger          | 0\.9000\(PCKh0\.5\)                                          | 0\.8964\(PCKh0\.5\)                                          |
| 19   | Pose Estimation          | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G\_1\.2 | caffe      | VGG            | 368\*368   | 49\.88G       | ai\_challenger                          | ai\_challenger          | 0\.4507\(OKs\)                                               | 0\.4422\(Oks\)                                               |
| 20   | Face Detection           | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G\_1\.2               | caffe      | VGG\-16        | 320\*320   | 0\.49G        | wider\_face                             | FDDB                    | 0\.8833                                                      | 0\.8791                                                      |
| 21   | Face Detection           | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G\_1\.2               | caffe      | VGG\-16        | 360\*640   | 1\.11G        | wider\_face                             | FDDB                    | 0\.8931                                                      | 0\.8925                                                      |
| 22   | Face Recognition         | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G\_1\.2                | caffe      | lenet          | 96\*72     | 0\.14G        | celebA                                  | processed helen         | 0\.1952\(L2 loss\)                                           | 0\.1972\(L2 loss\)                                           |
| 23   | Re\-identification       | reid                           | cf\_reid\_market1501\_160\_80\_0\.95G\_1\.2               | caffe      | resnet18       | 160\*80    | 0\.95G        | Market1501\+CUHK03                      | Market1501              | 0\.7800                                                      | 0\.7790                                                      |
| 24   | Detection+Segmentation   | multi-task                     | cf\_multitask\_bdd\_288\_512\_14\.8G\_1\.2                | caffe      | ssd            | 288\*512   | 14\.8G        | BDD100K+Cityscapes                      | BDD100K+Cityscapes      | 0\.2228(Det) 0\.4088(Seg)                                    | 0\.2202(Det) 0\.4058(Seg)                                    |
| 25   | Object Detection         | yolov3\_bdd                    | dk\_yolov3\_bdd\_288\_512\_53\.7G\_1\.2                   | darknet    | darknet\-53    | 288\*512   | 53\.7G        | bdd100k                                 | bdd100k                 | 0\.5058                                                      | 0\.4914                                                      |
| 26   | Object Detection         | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G\_1\.2      | darknet    | darknet\-53    | 256\*512   | 5\.46G        | Cityscapes Train                        | Cityscape Val           | 0\.5520                                                      | 0\.5300                                                      |
| 27   | Object Detection         | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G\_1\.2                  | darknet    | darknet\-53    | 416\*416   | 65\.42G       | voc07\+12\_trainval                     | voc07\_test             | 0\.8240\(MaxIntegral\)                                       | 0\.8150\(MaxIntegral\)                                       |
| 28   | Object Detection         | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G\_1\.2                      | darknet    | darknet\-19    | 448\*448   | 34G           | voc07\+12\_trainval                     | voc07\_test             | 0\.7845\(MaxIntegral\)                                       | 0\.7739\(MaxIntegral\)                                       |
| 29   | Object Detection         | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G\_1\.2           | darknet    | darknet\-19    | 448\*448   | 11\.56G       | voc07\+12\_trainval                     | voc07\_test             | 0\.7700\(MaxIntegral\)                                       | 0\.7600\(MaxIntegral\)                                       |
| 30   | Object Detection         | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G\_1\.2            | darknet    | darknet\-19    | 448\*448   | 9\.86G        | voc07\+12\_trainval                     | voc07\_test             | 0\.7670\(MaxIntegral\)                                       | 0\.7530\(MaxIntegral\)                                       |
| 31   | Object Detection         | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G\_1\.2            | darknet    | darknet\-19    | 448\*448   | 7\.82G        | voc07\+12\_trainval                     | voc07\_test             | 0\.7576\(MaxIntegral\)                                       | 0\.7460\(MaxIntegral\)                                       |
| 32   | Face Recognition         | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G_1.2                       | caffe      | resnet20       | 112*96     | 3.5G          | private data                            | private data            | 0.9610                                                       | 0.9510                                                       |
| 33   | Face Recognition         | ResNet64-face                  | cf_facerec-resnet64_112_96_11G_1.2                        | caffe      | resnet64       | 112*96     | 11G           | private data                            | private data            | 0.9830                                                       | 0.9820                                                       |
| 34   | Medical Segmentation     | FPN_Res18_Medical_Segmentation | cf_FPN-resnet18_EDD_320_320_45.3G_1.2                     | caffe      | resnet18       | 320*320    | 45.3G         | EDD_seg                                 | EDD_seg                 | mean dice=0.8202                mean jaccard=0.7925                                                                       F2-score=0.8075 | mean dice =0.8049                     mean jaccard =0.7771                                                                              F2- score=0.7916 |
| 35   | Plate Detection          | plate_detection                | cf_plate-detection_320_320_0.49G_1.2                      | caffe      | modify_vgg     | 320*320    | 0.49G         | private data                            | private data            | 0.9720                                                       | 0.9700                                                       |
| 36   | Plate Recognition        | plate_recognition              | cf_plate-recognition_96_288_1.75G_1.2                     | caffe      | Google\_v1     | 96*288     | 1.75G         | private data                            | private data            | plate number:99.51% plate color:100%                         | plate number:99.51% plate color:100%                         |
| 37   | Image Classifiction      | Inception\_resnet\_v2          | tf\_inceptionresnetv2\_imagenet\_299\_299\_26\.35G\_1\.2  | tensorflow | inception      | 299\*299   | 26\.35G       | ImageNet Train                          | ImageNet Val            | 0\.8037                                                      | 0\.7946                                                      |
| 38   | Image Classifiction      | Inception\_v1                  | tf\_inceptionv1\_imagenet\_224\_224\_3G\_1\.2             | tensorflow | inception      | 224\*224   | 3G            | ImageNet Train                          | ImageNet Val            | 0\.6976                                                      | 0\.6794                                                      |
| 39   | Image Classifiction      | Inception\_v3                  | tf\_inceptionv3\_imagenet\_299\_299\_11\.45G\_1\.2        | tensorflow | inception      | 299\*299   | 11\.45G       | ImageNet Train                          | ImageNet Val            | 0\.7798                                                      | 0\.7607                                                      |
| 40   | Image Classifiction      | Inception\_v4                  | tf\_inceptionv4\_imagenet\_299\_299\_24\.55G\_1\.2        | tensorflow | inception      | 299\*299   | 24\.55G       | ImageNet Train                          | ImageNet Val            | 0\.8018                                                      | 0\.7928                                                      |
| 41   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_0\.25\_imagenet\_128\_128\_27\.15M\_1\.2 | tensorflow | mobilenet      | 128\*128   | 27\.15M       | ImageNet Train                          | ImageNet Val            | 0\.4144                                                      | 0\.3464                                                      |
| 42   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_0\.5\_imagenet\_160\_160\_150\.07M\_1\.2 | tensorflow | mobilenet      | 160\*160   | 150\.07M      | ImageNet Train                          | ImageNet Val            | 0\.5903                                                      | 0\.5195                                                      |
| 43   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_1\.0\_imagenet\_224\_224\_1\.14G\_1\.2   | tensorflow | mobilenet      | 224\*224   | 1\.14G        | ImageNet Train                          | ImageNet Val            | 0\.7102                                                      | 0\.6779                                                      |
| 44   | Image Classifiction      | Mobilenet\_v2                  | tf\_mobilenetv2\_1\.0\_imagenet\_224\_224\_0\.59G\_1\.2   | tensorflow | mobilenet      | 224\*224   | 0\.59G        | ImageNet Train                          | ImageNet Val            | 0\.7013                                                      | 0\.6767                                                      |
| 45   | Image Classifiction      | Mobilenet\_v2                  | tf\_mobilenetv2\_1\.4\_imagenet\_224\_224\_1\.16G\_1\.2   | tensorflow | mobilenet      | 224\*224   | 1\.16G        | ImageNet Train                          | ImageNet Val            | 0\.7411                                                      | 0\.7194                                                      |
| 46   | Image Classifiction      | resnet\_v1\_50                 | tf\_resnetv1\_50\_imagenet\_224\_224\_6\.97G\_1\.2        | tensorflow | resnetv1       | 224\*224   | 6\.97G        | ImageNet Train                          | ImageNet Val            | 0\.7520                                                      | 0\.7478                                                      |
| 47   | Image Classifiction      | resnet\_v1\_101                | tf\_resnetv1\_101\_imagenet\_224\_224\_14\.4G\_1\.2       | tensorflow | resnetv1       | 224\*224   | 14\.4G        | ImageNet Train                          | ImageNet Val            | 0\.7640                                                      | 0\.7560                                                      |
| 48   | Image Classifiction      | resnet\_v1\_152                | tf\_resnetv1\_152\_imagenet\_224\_224\_21\.83G\_1\.2      | tensorflow | resnetv1       | 224\*224   | 21\.83G       | ImageNet Train                          | ImageNet Val            | 0\.7681                                                      | 0\.7463                                                      |
| 49   | Image Classifiction      | vgg\_16                        | tf\_vgg16\_imagenet\_224\_224\_30\.96G\_1\.2              | tensorflow | vgg            | 224\*224   | 30\.96G       | ImageNet Train                          | ImageNet Val            | 0\.7089                                                      | 0\.7069                                                      |
| 50   | Image Classifiction      | vgg\_19                        | tf\_vgg19\_imagenet\_224\_224\_39\.28G\_1\.2              | tensorflow | vgg            | 224\*224   | 39\.28G       | ImageNet Train                          | ImageNet Val            | 0\.7100                                                      | 0\.7026                                                      |
| 51   | Object Detection         | ssd\_mobilenet\_v1             | tf\_ssdmobilenetv1\_coco\_300\_300\_2\.47G\_1\.2          | tensorflow | mobilenet      | 300\*300   | 2\.47G        | coco2017                                | coco2014 minival        | 0\.2080                                                      | 0\.2100                                                      |
| 52   | Object Detection         | ssd\_mobilenet\_v2             | tf\_ssdmobilenetv2\_coco\_300\_300\_3\.75G\_1\.2          | tensorflow | mobilenet      | 300\*300   | 3\.75G        | coco2017                                | coco2014 minival        | 0\.2150                                                      | 0\.2110                                                      |
| 53   | Object Detection         | ssd\_resnet50\_v1\_fpn         | tf\_ssdresnet50v1\_fpn\_coco\_640\_640\_178\.4G\_1\.2     | tensorflow | resnet50       | 300\*300   | 178\.4G       | coco2017                                | coco2014 minival        | 0\.3010                                                      | 0\.2900                                                      |
| 54   | Object Detection         | yolov3\_voc                    | tf\_yolov3\_voc\_416\_416\_65\.63G\_1\.2                  | tensorflow | darknet\-53    | 416\*416   | 65\.63G       | voc07\+12\_trainval                     | voc07\_test             | 0\.7846                                                      | 0\.7744                                                      |
| 55   | Object Detection         | mlperf\_ssd\_resnet34          | tf\_mlperf_resnet34\_coco\_1200\_1200\_433G\_1\.2         | tensorflow | resnet34       | 1200\*1200 | 433G          | coco2017                                | coco2017                | 0\.2250                                                      | 0\.2150                                                      |
| 56   | Segmentation             | ENet                           | pt_ENet_cityscapes_512_1024_8.6G_1.2                      | pytorch    | -              | 512*1024   | 8.6G          | Cityscapes                              | Cityscapes              | 0.6440                                                       | 0.6306                                                       |
| 57   | Segmentation             | SemanticFPN                    | pt_SemanticFPN_cityscapes_256_512_10G_1.2                 | pytorch    | FPN-Resnet18   | 256*512    | 10G           | Cityscapes                              | Cityscapes              | 0.6290                                                       | 0.6090                                                       |
| 58   | Face Recognition         | ResNet20-face                  | pt_facerec-resnet20_mixed_112_96_3.5G_1.2                 | pytorch    | resnet20       | 112*96     | 3.5G          | mixed                                   | mixed                   | 0.9955                                                       | 0.9952                                                       |
| 59   | Face Quality             | face quality                   | pt_face-quality_80_60_61.68M_1.2                          | pytorch    | lenet          | 80*60      | 61.68M        | private data                            | private data            | 0.1233                                                       | 0.1273                                                       |
| 60   | Multi Task               | MT-resnet18                    | pt_MT-resnet18_mixed_320_512_13.65G_1.2                   | pytorch    | resnet18       | 320*512    | 13.65G        | mixed                                   | mixed                   | mAP:  39.50%     mIOU: 44.03%                                | mAP:  38.70%     mIOU: 42.56%                                |
| 61   | Face ReID                | face_reid_large                | pt_facereid-large_96_96_515M_1.2                          | pytorch    | resnet18       | 96*96      | 515M          | private data                            | private data            | mAP: 79.5%  Rank1: 95.4%                                     | mAP: 79.0%         Rank1: 95.1%                              |
| 62   | Face ReID                | face_reid_small                | pt_facereid-small_80_80_90M_1.2                           | pytorch    | resnet_small   | 80*80      | 90M           | private data                            | private data            | mAP: 56.3%  Rank1: 86.8%                                     | mAP: 56.1%    Rank1: 86.4%                                   |
| 63   | Re\-identification       | reid                           | pt_personreid_market1501_256_128_4.2G_1.2                 | pytorch    | resnet50       | 256*128    | 4.2G          | market1501                              | market1501              | mAP: 84.0%   Rank1: 94.6%                                    | mAP: 83.5%     Rank1: 94.2%                                  |

</details>

### Naming Rules
Model name: `F_M_(D)_H_W_(P)_C_V`
* `F` specifies training framework: `cf` is Caffe, `tf` is Tensorflow, `dk` is Darknet, `pt` is PyTorch
* `M` specifies the model
* `D` specifies the dataset. It is optional depending on whether the dataset is public or private.
* `H` specifies the height of input data
* `W` specifies the width of input data
* `P` specifies the pruning ratio, it means how much computation is reduced. It is optional depending on whether the model is pruned.
* `C` specifies the computation of the model: how many Gops per image
* `V` specifies the version of Vitis AI


For example, `cf_refinedet_coco_480_360_0.8_25G_1.2` is a `RefineDet` model trained with `Caffe` using `COCO` dataset, input data size is `480*360`, `80%` pruned, the computation per image is `25Gops` and Vitis AI version is `1.2`.


### caffe-xilinx 
This is a custom distribution of caffe. Please use caffe-xilinx to test/finetune the caffe models listed in this page.

**Note:** To download caffe-xlinx, visit [caffe-xilinx.zip](https://www.xilinx.com/bin/public/openDownload?filename=caffe-xilinx-1.1.zip)



## Model Download
The following table lists various models, download link and MD5 checksum for the zip file of each model.

**Note:** To download all the models, visit [all_models_1.2.zip](https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.2.zip).

<details>
 <summary><b>Click here to view details</b></summary>

If you are a:
 - Linux user, use the [`get_model.sh`](reference-files/get_model.sh) script to download all the models.   
 - Windows user, use the download link listed in the following table to download a model.

| No\. | Model                                            | Size | Download link | Checksum |
| ---- | ------------------------------------------------ | ---- | ------------- | -------- |
| 1    | cf_resnet50_imagenet_224_224_7.7G_1.2            |      |               |          |
| 2    | cf_inceptionv1_imagenet_224_224_3.16G_1.2        |      |               |          |
| 3    | cf_inceptionv2_imagenet_224_224_4G_1.2           |      |               |          |
| 4    | cf_inceptionv3_imagenet_299_299_11.4G_1.2        |      |               |          |
| 5    | cf_inceptionv4_imagenet_299_299_24.5G_1.2        |      |               |          |
| 6    | cf_mobilenetv2_imagenet_224_224_0.59G_1.2        |      |               |          |
| 7    | cf_squeeze_imagenet_227_227_0.76G_1.2            |      |               |          |
| 8    | cf_resnet18_imagenet_224_224_3.65G_1.2           |      |               |          |
| 9    | cf_ssdpedestrian_coco_360_640_0.97_5.9G_1.2      |      |               |          |
| 10   | cf_refinedet_coco_360_480_0.8_25G_1.2            |      |               |          |
| 11   | cf_refinedet_coco_360_480_0.92_10.10G_1.2        |      |               |          |
| 12   | cf_refinedet_coco_360_480_0.96_5.08G_1.2         |      |               |          |
| 13   | cf_ssdadas_bdd_360_480_0.95_6.3G_1.2             |      |               |          |
| 14   | cf_ssdtraffic_360_480_0.9_11.6G_1.2              |      |               |          |
| 15   | cf_VPGnet_caltechlane_480_640_0.99_2.5G_1.2      |      |               |          |
| 16   | cf_ssdmobilenetv2_bdd_360_480_6.57G_1.2          |      |               |          |
| 17   | cf_fpn_cityscapes_256_512_8.9G_1.2               |      |               |          |
| 18   | cf_SPnet_aichallenger_224_128_0.54G_1.2          |      |               |          |
| 19   | cf_openpose_aichallenger_368_368_0.3_189.7G_1.2  |      |               |          |
| 20   | cf_densebox_wider_320_320_0.49G_1.2              |      |               |          |
| 21   | cf_densebox_wider_360_640_1.11G_1.2              |      |               |          |
| 22   | cf_landmark_celeba_96_72_0.14G_1.2               |      |               |          |
| 23   | cf_reid_market1501_160_80_0.95G_1.2              |      |               |          |
| 24   | cf_multitask_bdd_288_512_14.8G_1.2               |      |               |          |
| 25   | dk_yolov3_bdd_288_512_53.7G_1.2                  |      |               |          |
| 26   | dk_yolov3_cityscapes_256_512_0.9_5.46G_1.2       |      |               |          |
| 27   | dk_yolov3_voc_416_416_65.42G_1.2                 |      |               |          |
| 28   | dk_yolov2_voc_448_448_34G_1.2                    |      |               |          |
| 29   | dk_yolov2_voc_448_448_0.66_11.56G_1.2            |      |               |          |
| 30   | dk_yolov2_voc_448_448_0.71_9.86G_1.2             |      |               |          |
| 31   | dk_yolov2_voc_448_448_0.77_7.82G_1.2             |      |               |          |
| 32   | cf_facerec-resnet20_112_96_3.5G_1.2              |      |               |          |
| 33   | cf_facerec-resnet64_112_96_11G_1.2               |      |               |          |
| 34   | cf_FPN-resnet18_EDD_320_320_45.3G_1.2            |      |               |          |
| 35   | cf_plate-detection_320_320_0.49G_1.2             |      |               |          |
| 36   | cf_plate-recognition_96_288_1.75G_1.2            |      |               |          |
| 37   | tf_inceptionresnetv2_imagenet_299_299_26.35G_1.2 |      |               |          |
| 38   | tf_inceptionv1_imagenet_224_224_3G_1.2           |      |               |          |
| 39   | tf_inceptionv3_imagenet_299_299_11.45G_1.2       |      |               |          |
| 40   | tf_inceptionv4_imagenet_299_299_24.55G_1.2       |      |               |          |
| 41   | tf_mobilenetv1_0.25_imagenet_128_128_27.15M_1.2  |      |               |          |
| 42   | tf_mobilenetv1_0.5_imagenet_160_160_150.07M_1.2  |      |               |          |
| 43   | tf_mobilenetv1_1.0_imagenet_224_224_1.14G_1.2    |      |               |          |
| 44   | tf_mobilenetv2_1.0_imagenet_224_224_0.59G_1.2    |      |               |          |
| 45   | tf_mobilenetv2_1.4_imagenet_224_224_1.16G_1.2    |      |               |          |
| 46   | tf_resnetv1_50_imagenet_224_224_6.97G_1.2        |      |               |          |
| 47   | tf_resnetv1_101_imagenet_224_224_14.4G_1.2       |      |               |          |
| 48   | tf_resnetv1_152_imagenet_224_224_21.83G_1.2      |      |               |          |
| 49   | tf_vgg16_imagenet_224_224_30.96G_1.2             |      |               |          |
| 50   | tf_vgg19_imagenet_224_224_39.28G_1.2             |      |               |          |
| 51   | tf_ssdmobilenetv1_coco_300_300_2.47G_1.2         |      |               |          |
| 52   | tf_ssdmobilenetv2_coco_300_300_3.75G_1.2         |      |               |          |
| 53   | tf_ssdresnet50v1_fpn_coco_640_640_178.4G_1.2     |      |               |          |
| 54   | tf_yolov3_voc_416_416_65.63G_1.2                 |      |               |          |
| 55   | tf_mlperf_resnet34_coco_1200_1200_433G_1.2       |      |               |          |
| 56   | pt_ENet_cityscapes_512_1024_8.6G_1.2             |      |               |          |
| 57   | pt_SemanticFPN_cityscapes_256_512_10G_1.2        |      |               |          |
| 58   | pt_facerec-resnet20_mixed_112_96_3.5G_1.2        |      |               |          |
| 59   | pt_face-quality_80_60_61.68M_1.2                 |      |               |          |
| 60   | pt_MT-resnet18_mixed_320_512_13.65G_1.2          |      |               |          |
| 61   | pt_facereid-large_96_96_515M_1.2                 |      |               |          |
| 62   | pt_facereid-small_80_80_90M_1.2                  |      |               |          |
| 63   | pt_personreid_market1501_256_128_4.2G_1.2        |      |               |          |
| all  | all_models_1.2                                   |      |               |          |


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
    │   ├── deploy.model.pb             # Quantized model for the compiler (extended Tensorflow format).
    │   └── quantize_eval_model.pb      # Quantized model for evaluation.
    │
    └── float                             
        └── frozen.pb                   # Float-point frozen model, the input to the `vai_q_tensorflow`.


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
    │   ├── _int.py                     # Converted NNDCT format model.
    │   └── quant_info.json             # Quantization steps of tensors got. Please keep it for evaluation of quantized model.
    │                                           
    └── float                           
        └── _int.pth                    # Trained float-point model.
        
                                          
                                          
**Note:** For more information on `vai_q_caffe` and `vai_q_tensorflow`, see the [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf).


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
| 1    | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G             | 12\.85                          | 77\.8                                   | 179\.3                                 |
| 2    | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G         | 5\.47                           | 182\.683                                | 485\.533                               |
| 3    | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G             | 6\.76                           | 147\.933                                | 373\.267                               |
| 4    | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G         | 17                              | 58\.8333                                | 155\.4                                 |
| 5    | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G         | 4\.09                           | 244\.617                                | 638\.067                               |
| 6    | tf\_resnet50                   | tf\_resnet50\_imagenet\_224\_224\_6\.97G            | 11\.94                          | 83\.7833                                | 191\.417                               |
| 7    | tf\_inception\_v1              | tf\_inceptionv1\_imagenet\_224\_224\_3G             | 6\.72                           | 148\.867                                | 358\.283                               |
| 8    | tf\_mobilenet\_v2              | tf\_mobilenetv2\_imagenet\_224\_224\_1\.17G         | 5\.46                           | 183\.117                                | 458\.65                                |
| 9    | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G            | 11\.33                          | 88\.2667                                | 320\.5                                 |
| 10   | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | 12\.96                          | 77\.1833                                | 314\.717                               |
| 11   | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G              | 17\.49                          | 57\.1833                                | 218\.183                               |
| 12   | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | 24\.21                          | 41\.3                                   | 141\.233                               |
| 13   | tf\_ssd\_voc                   | tf\_ssd\_voc\_300\_300\_64\.81G                     | 69\.28                          | 14\.4333                                | 46\.7833                               |
| 14   | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G               | 2\.43                           | 412\.183                                | 1416\.63                               |
| 15   | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G               | 5\.01                           | 199\.717                                | 719\.75                                |
| 16   | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G      | 11\.09                          | 90\.1667                                | 259\.65                                |
| 17   | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G                  | 70\.51                          | 14\.1833                                | 44\.4                                  |
| 18   | tf\_yolov3\_voc                | tf\_yolov3\_voc\_416\_416\_65\.63G                  | 70\.75                          | 14\.1333                                | 44\.0167                               |
| 19   | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G            | 29\.91                          | 33\.4333                                | 109\.067                               |
| 20   | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G       | 15\.39                          | 64\.9667                                | 216\.317                               |
| 21   | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G        | 11\.04                          | 90\.5833                                | 312                                    |
| 22   | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G                | 16\.58                          | 60\.3                                   | 203\.867                               |
| 23   | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G     | 9\.44                           | 105\.9                                  | 424\.667                               |
| 24   | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G           | 1\.73                           | 579\.067                                | 1620\.67                               |
| 25   | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G | 279\.07                         | 3\.58333                                | 38\.5                                  |
| 26   | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G                      | 39\.76                          | 25\.15                                  | 86\.35                                 |
| 27   | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G           | 18\.42                          | 54\.2833                                | 211\.217                               |
| 28   | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G            | 16\.42                          | 60\.9167                                | 242\.433                               |
| 29   | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G            | 14\.46                          | 69\.1667                                | 286\.733                               |
| 30   | Inception\-v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G         | 34\.25                          | 29\.2                                   | 84\.25                                 |
| 31   | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G             | 3\.6                            | 277\.65                                 | 1080\.77                               |
| 32   | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G                | 1\.13                           | 885\.033                                | 1623\.3                                |
| 33   | reid                           | cf\_reid\_marketcuhk\_160\_80\_0\.95G               | 2\.67                           | 375                                     | 773\.533                               |
| 34   | yolov3\_bdd                    | dk\_yolov3\_bdd\_288\_512\_53\.7G                   | 73\.89                          | 13\.5333                                | 42\.8833                               |
| 35   | tf\_mobilenet\_v1              | tf\_mobilenetv1\_imagenet\_224\_224\_1\.14G         | 3\.2                            | 312\.067                                | 875\.967                               |
| 36   | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | 5\.1                            | 195\.95                                 | 524\.433                               |
| 37   | resnet18\_wide                 | tf\_resnet18\_imagenet\_224\_224\_28G               | 33\.28                          | 30\.05                                  | 83\.4167                               |
</details>


### Performance on ZCU102 (0432055-05)
Measured with Vitis AI 1.2 and Vitis AI Library 1.2  

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-05)` board with a `3 * B4096  @ 281MHz   V1.4.1` DPU configuration:


| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 13.97                            | 71.6                                  | 157.8                                |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 5.73                             | 174.5                                | 449.9                                |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 6.00                             | 166.7                                | 420.7                                |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 7.30                             | 136.9                                | 324.2                                |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 17.77                            | 56.2                                  | 130.7                                |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 35.46                            | 28.2                                  | 67.1                                 |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G        | 4.78                             | 209.1                                | 563.4                                |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 3.75                             | 266.7                                | 1035.9                               |
| 9    | ssd_pedestrain_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 13.11                            | 76.2                                  | 283.9                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G            | 31.58                            | 31.7                                  | 101.5                                |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G        | 16.68                            | 59.9                                  | 198.1                                |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G         | 12.07                            | 82.8                                  | 278.7                                |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G             | 12.07                            | 82.8                                  | 283.2                                |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G              | 18.35                            | 54.5                                  | 202.4                                |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 9.58                             | 104.4                                 | 382.9                                |
| 16   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G          | 26.26                            | 38.1                                  | 114.7                                |
| 17   | FPN                        | cf_fpn_cityscapes_256_512_8.9G               | 16.76                            | 59.7                                  | 177.3                                |
| 18   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G          | 2.62                             | 381.8                                | 1351.2                               |
| 19   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G  | 285.86                           | 3.5                                  | 15.1                                 |
| 20   | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 2.55                             | 392.4                                | 1212.5                               |
| 21   | densebox_640_360           | cf_densebox_wider_360_640_1.11G              | 4.99                             | 200.5                                | 605.3                                |
| 22   | face_landmark              | cf_landmark_celeba_96_72_0.14G               | 1.18                             | 849.3                                | 1393.2                               |
| 23   | reid                       | cf_reid_market1501_160_80_0.95G              | 2.74                             | 364.6                                | 679.5                                |
| 24   | multi_task                 | cf_multitask_bdd_288_512_14.8G               | 28.26                            | 35.4                                  | 129.0                                |
| 25   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 77.09                            | 13.0                                  | 34.7                                 |
| 26   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 11.91                            | 84.0                                  | 231.2                                |
| 27   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 73.76                            | 13.5                                  | 35.6                                 |
| 28   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                    | 37.32                            | 26.8                                  | 71.5                                 |
| 29   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G            | 15.82                            | 63.2                                  | 187.4                                |
| 30   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G             | 13.73                            | 72.8                                  | 216.7                                |
| 31   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G             | 11.73                            | 85.3                                  | 260.2                                |
| 32   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G              | 5.98                             | 167.1                                | 324.1                                |
| 33   | ResNet64-face              | cf_facerec-resnet64_112_96_11G               | 13.69                            | 73.0                                  | 175.2                                |
| 34   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G            | 78.85                            | 12.7                                  | 40.3                                 |
| 35   | plate detection            | cf_plate-detection_320_320_0.49G             | 1.99                             | 503.0                                | 1838.4                               |
| 36   | plate recognition          | cf_plate-recognition_96_288_1.75G            | 8.88                             | 112.6                                | 385.4                                |
| 37   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G | 43.44                            | 23.0                                  | 50.3                                 |
| 38   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G           | 5.81                             | 172.1                                | 433.9                                |
| 39   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G       | 17.84                            | 56.0                                  | 129.4                                |
| 40   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G       | 35.48                            | 28.2                                  | 67.2                                 |
| 41   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M  | 1.24                             | 807.7                                | 2191.6                               |
| 42   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M  | 1.78                             | 559.8                                | 1929.2                               |
| 43   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G    | 3.89                             | 257                                  | 774.6                                |
| 44   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G    | 4.70                             | 212.5                                | 584.0                                |
| 45   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G    | 6.34                             | 157.6                                | 403.3                                |
| 46   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G        | 13.00                            | 76.9                                  | 167.4                                |
| 47   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G       | 23.53                            | 42.5                                  | 94.9                                 |
| 48   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G      | 34.17                            | 29.2                                  | 66.2                                 |
| 49   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G             | 50.17                            | 19.9                                  | 41.1                                 |
| 50   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G             | 58.12                            | 17.2                                  | 36.8                                 |
| 51   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G         | 11.12                            | 89.9                                  | 306.1                                |
| 52   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G         | 16.00                            | 62.5                                  | 196.8                                |
| 53   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 732.89                           | 1.4                                  | 5.2                                  |
| 54   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                 | 74.22                            | 13.5                                  | 35.3                                 |
| 55   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G       | 502.76                           | 2.0                                  | 7.3                                  |

</details>


### Performance on ZCU104
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 300MHz   V1.4.1` DPU configuration:


| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 12.69                            | 78.8                                  | 148.0                                |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 5.06                             | 197.4                                | 411.1                                |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 5.24                             | 190.8                                | 388.2                                |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 6.51                             | 153.7                                | 301.8                                |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 16.42                            | 60.9                                  | 117.8                                |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 33.00                            | 30.3                                  | 58.4                                 |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G        | 4.12                             | 242.9                                | 519.9                                |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 3.69                             | 271.1                                | 960.1                                |
| 9    | ssd_pedestrain_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 12.70                            | 78.7                                  | 220.7                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G            | 30.70                            | 32.6                                  | 76.0                                 |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G        | 16.30                            | 61.3                                  | 154.0                                |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G         | 11.93                            | 83.8                                  | 228.2                                |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G             | 11.79                            | 84.8                                  | 232.2                                |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G              | 17.77                            | 56.2                                  | 152.8                                |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 9.31                             | 107.4                                | 352.5                                |
| 16   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G          | 39.01                            | 25.6                                  | 101.5                                |
| 17   | FPN                        | cf_fpn_cityscapes_256_512_8.9G               | 16.16                            | 61.9                                  | 168.6                                |
| 18   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G          | 2.02                             | 494.7                                | 1209.2                               |
| 19   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G  | 273.75                           | 3.7                                  | 10.9                                 |
| 20   | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 2.53                             | 395.4                                | 1271.4                               |
| 21   | densebox_640_360           | cf_densebox_wider_360_640_1.11G              | 4.90                             | 203.9                                | 619.8                                |
| 22   | face_landmark              | cf_landmark_celeba_96_72_0.14G               | 1.12                             | 890.6                                | 1450.4                               |
| 23   | reid                       | cf_reid_market1501_160_80_0.95G              | 2.58                             | 387.4                                | 700.2                                |
| 24   | multi_task                 | cf_multitask_bdd_288_512_14.8G               | 27.73                            | 36.0                                  | 108.6                                |
| 25   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 73.56                            | 13.6                                  | 28.5                                 |
| 26   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 11.79                            | 84.8                                  | 218.9                                |
| 27   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 70.19                            | 14.2                                  | 29.4                                 |
| 28   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                    | 35.22                            | 28.4                                  | 58.8                                 |
| 29   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G            | 15.04                            | 66.5                                  | 152.9                                |
| 30   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G             | 13.05                            | 76.6                                  | 179.6                                |
| 31   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G             | 11.17                            | 89.5                                  | 216.2                                |
| 32   | ResNet20-face              | cf_facerec-resnet20_112_96_3.5G              | 5.63                             | 177.7                                | 309.0                                |
| 33   | ResNet64-face              | cf_facerec-resnet64_112_96_11G               | 12.86                            | 77.7                                  | 147.4                                |
| 34   | FPN_Res18_segmentation     | cf_FPN-resnet18_EDD_320_320_45.3G            | 75.69                            | 13.2                                  | 31.5                                 |
| 35   | plate detection            | cf_plate-detection_320_320_0.49G             | 2.01                             | 498.1                                | 1811.9                               |
| 36   | plate recognition          | cf_plate-recognition_96_288_1.75G            | 4.52                             | 221.3                                | 541.5                                |
| 37   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G | 40.00                            | 25.0                                  | 46.2                                 |
| 38   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G           | 5.06                             | 197.5                                | 402.1                                |
| 39   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G       | 16.49                            | 60.6                                  | 117.1                                |
| 40   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G       | 33.04                            | 30.2                                  | 58.3                                 |
| 41   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M  | 0.84                             | 1192                                  | 3735.7                               |
| 42   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M  | 1.36                             | 736.6                                | 1938.6                               |
| 43   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G    | 3.24                             | 308.3                                | 718.2                                |
| 44   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G    | 4.09                             | 244.5                                | 527.6                                |
| 45   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G    | 5.54                             | 180.3                                | 370.1                                |
| 46   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G        | 11.82                            | 84.5                                  | 158.2                                |
| 47   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G       | 21.67                            | 46.1                                  | 86                                   |
| 48   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G      | 31.56                            | 31.7                                  | 59.1                                 |
| 49   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G             | 46.83                            | 21.3                                  | 36.8                                 |
| 50   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G             | 54.44                            | 18.3                                  | 32.6                                 |
| 51   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G         | 10.65                            | 93.9                                  | 333.9                                |
| 52   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G         | 15.02                            | 66.6                                  | 184.1                                |
| 53   | ssd_resnet_50_v1_fpn       | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 743.43                           | 1.3                                  | 5.3                                  |
| 54   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                 | 70.61                            | 14.2                                  | 29.1                                 |
| 55   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G       | 543.17                           | 1.8                                  | 5.4                                  |

</details>


### Performance on U50
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 6 DPUv3E kernels running at 300Mhz in Gen3x4:
  

| No\. | Model                          | Name                                         | Frequency \(MHz\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :----------------------------- | :------------------------------------------- | ----------------- | ------------------------------------ |
| 1    | resnet50                       | cf_resnet50_imagenet_224_224_7.7G            | 270               | 476.54                               |
| 2    | resnet18                       | cf_resnet18_imagenet_224_224_3.65G           | 270               | 1008.38                              |
| 3    | Inception_v1                   | cf_inceptionv1_imagenet_224_224_3.16G        | 270               | 897.48                               |
| 4    | Inception_v2                   | cf_inceptionv2_imagenet_224_224_4G           | 270               | 762.22                               |
| 5    | Inception_v3                   | cf_inceptionv3_imagenet_299_299_11.4G        | 270               | 341.20                               |
| 6    | Inception_v4                   | cf_inceptionv4_imagenet_299_299_24.5G        | 270               | 164.10                               |
| 7    | SqueezeNet                     | cf_squeeze_imagenet_227_227_0.76G            | 270               | 1869.71                              |
| 8    | ssd_pedestrain_pruned_0_97     | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 270               | 496.47                               |
| 9    | refinedet_pruned_0_8           | cf_refinedet_coco_360_480_0.8_25G            | 270               | 184.52                               |
| 10   | refinedet_pruned_0_92          | cf_refinedet_coco_360_480_0.92_10.10G        | 270               | 382.02                               |
| 11   | refinedet_pruned_0_96          | cf_refinedet_coco_360_480_0.96_5.08G         | 270               | 540.90                               |
| 12   | ssd_adas_pruned_0_95           | cf_ssdadas_bdd_360_480_0.95_6.3G             | 270               | 514.58                               |
| 13   | ssd_traffic_pruned_0_9         | cf_ssdtraffic_360_480_0.9_11.6G              | 270               | 368.53                               |
| 14   | VPGnet_pruned_0_99             | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 270               | 412.08                               |
| 15   | FPN                            | cf_fpn_cityscapes_256_512_8.9G               | 270               | 383.77                               |
| 16   | SP_net                         | cf_SPnet_aichallenger_224_128_0.54G          | 270               | 773.61                               |
| 17   | Openpose_pruned_0_3            | cf_openpose_aichallenger_368_368_0.3_189.7G  | 270               | 29.12                                |
| 18   | densebox_320_320               | cf_densebox_wider_320_320_0.49G              | 270               | 1365.22                              |
| 19   | densebox_640_360               | cf_densebox_wider_360_640_1.11G              | 270               | 673.76                               |
| 20   | face_landmark                  | cf_landmark_celeba_96_72_0.14G               | 270               | 2940.13                              |
| 21   | reid                           | cf_reid_market1501_160_80_0.95G              | 270               | 1947.30                              |
| 22   | multi_task                     | cf_multitask_bdd_288_512_14.8G               | 270               | 172.99                               |
| 23   | yolov3_bdd                     | dk_yolov3_bdd_288_512_53.7G                  | 270               | 73.93                                |
| 24   | yolov3_adas_pruned_0_9         | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 270               | 558.32                               |
| 25   | yolov3_voc                     | dk_yolov3_voc_416_416_65.42G                 | 270               | 75.61                                |
| 26   | yolov2_voc                     | dk_yolov2_voc_448_448_34G                    | 270               | 160.53                               |
| 27   | yolov2_voc_pruned_0_66         | dk_yolov2_voc_448_448_0.66_11.56G            | 270               | 379.02                               |
| 28   | yolov2_voc_pruned_0_71         | dk_yolov2_voc_448_448_0.71_9.86G             | 270               | 440.05                               |
| 29   | yolov2_voc_pruned_0_77         | dk_yolov2_voc_448_448_0.77_7.82G             | 270               | 710.40                               |
| 30   | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G              | 270               | 1173.84                              |
| 31   | ResNet64-face                  | cf_facerec-resnet64_112_96_11G               | 270               | 449.57                               |
| 32   | FPN_Res18_Medical_segmentation | cf_FPN-resnet18_EDD_320_320_45.3G            | 270               | 92.89                                |
| 33   | plate detection                | cf_plate-detection_320_320_0.49G             | 270               | 3938.60                              |
| 34   | Inception_resnet_v2            | tf_inceptionresnetv2_imagenet_299_299_26.35G | 270               | 143.27                               |
| 35   | Inception_v1                   | tf_inceptionv1_imagenet_224_224_3G           | 270               | 900.64                               |
| 36   | Inception_v3                   | tf_inceptionv3_imagenet_299_299_11.45G       | 270               | 336.09                               |
| 37   | Inception_v4                   | tf_inceptionv4_imagenet_299_299_24.55G       | 270               | 164.28                               |
| 38   | resnet_v1_50                   | tf_resnetv1_50_imagenet_224_224_6.97G        | 270               | 517.66                               |
| 39   | resnet_v1_101                  | tf_resnetv1_101_imagenet_224_224_14.4G       | 270               | 283.20                               |
| 40   | resnet_v1_152                  | tf_resnetv1_152_imagenet_224_224_21.83G      | 270               | 192.75                               |
| 41   | vgg_16                         | tf_vgg16_imagenet_224_224_30.96G             | 270               | 146.55                               |
| 42   | vgg_19                         | tf_vgg19_imagenet_224_224_39.28G             | 270               | 121.93                               |
| 43   | ssd_resnet_50_v1_fpn           | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 270               | 31.33                                |
| 44   | yolov3_voc                     | tf_yolov3_voc_416_416_65.63G                 | 270               | 75.61                                |
| 45   | mlperf_ssd_resnet34            | tf_mlperf_resnet34_coco_1200_1200_433G       | 270               | 13.96                                |
| 46   | torchvision                    | resnet50                                     | 270               | 494.94                               |
| 47   | torchvision                    | inception_v3                                 | 270               | 367.40                               |
| 48   | torchvision                    | squeezenet                                   | 270               | 1883.33                              |


</details>


### Performance on U50 lv9e
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 9 DPUv3E kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                          | Name                                         | Frequency \(MHz\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :----------------------------- | :------------------------------------------- | ----------------- | ------------------------------------ |
| 1    | resnet50                       | cf_resnet50_imagenet_224_224_7.7G            | 275               | 802.49                               |
| 2    | resnet18                       | cf_resnet18_imagenet_224_224_3.65G           | 275               | 1929.69                              |
| 3    | Inception_v1                   | cf_inceptionv1_imagenet_224_224_3.16G        | 275               | 1603.52                              |
| 4    | Inception_v2                   | cf_inceptionv2_imagenet_224_224_4G           | 275               | 1332.78                              |
| 5    | Inception_v3                   | cf_inceptionv3_imagenet_299_299_11.4G        | 275               | 555.44                               |
| 6    | Inception_v4                   | cf_inceptionv4_imagenet_299_299_24.5G        | 275               | 257.57                               |
| 7    | SqueezeNet                     | cf_squeeze_imagenet_227_227_0.76G            | 275               | 3783.85                              |
| 8    | ssd_pedestrain_pruned_0_97     | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 275               | 597.49                               |
| 9    | refinedet_pruned_0_8           | cf_refinedet_coco_360_480_0.8_25G            | 275               | 240.07                               |
| 10   | refinedet_pruned_0_92          | cf_refinedet_coco_360_480_0.92_10.10G        | 275               | 521.47                               |
| 11   | refinedet_pruned_0_96          | cf_refinedet_coco_360_480_0.96_5.08G         | 275               | 738.65                               |
| 12   | ssd_adas_pruned_0_95           | cf_ssdadas_bdd_360_480_0.95_6.3G             | 275               | 675.31                               |
| 13   | ssd_traffic_pruned_0_9         | cf_ssdtraffic_360_480_0.9_11.6G              | 275               | 468.43                               |
| 14   | VPGnet_pruned_0_99             | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 275               | 576.29                               |
| 15   | FPN                            | cf_fpn_cityscapes_256_512_8.9G               | 275               | 507.75                               |
| 16   | SP_net                         | cf_SPnet_aichallenger_224_128_0.54G          | 275               | 1487.46                              |
| 17   | Openpose_pruned_0_3            | cf_openpose_aichallenger_368_368_0.3_189.7G  | 275               | 43.33                                |
| 18   | densebox_320_320               | cf_densebox_wider_320_320_0.49G              | 275               | 2178.59                              |
| 19   | densebox_640_360               | cf_densebox_wider_360_640_1.11G              | 275               | 959.82                               |
| 20   | face_landmark                  | cf_landmark_celeba_96_72_0.14G               | 275               | 11882.00                             |
| 21   | reid                           | cf_reid_market1501_160_80_0.95G              | 275               | 5103.03                              |
| 22   | multi_task                     | cf_multitask_bdd_288_512_14.8G               | 275               | 144.24                               |
| 23   | yolov3_bdd                     | dk_yolov3_bdd_288_512_53.7G                  | 247.5             | 103.00                               |
| 24   | yolov3_adas_pruned_0_9         | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 247.5             | 808.06                               |
| 25   | yolov3_voc                     | dk_yolov3_voc_416_416_65.42G                 | 247.5             | 104.44                               |
| 26   | yolov2_voc                     | dk_yolov2_voc_448_448_34G                    | 247.5             | 227.73                               |
| 27   | yolov2_voc_pruned_0_66         | dk_yolov2_voc_448_448_0.66_11.56G            | 275               | 623.42                               |
| 28   | yolov2_voc_pruned_0_71         | dk_yolov2_voc_448_448_0.71_9.86G             | 275               | 720.38                               |
| 29   | yolov2_voc_pruned_0_77         | dk_yolov2_voc_448_448_0.77_7.82G             | 275               | 822.57                               |
| 30   | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G              | 275               | 1754.72                              |
| 31   | ResNet64-face                  | cf_facerec-resnet64_112_96_11G               | 275               | 664.02                               |
| 32   | FPN_Res18_Medical_segmentation | cf_FPN-resnet18_EDD_320_320_45.3G            | 275               | 140.71                               |
| 33   | plate detection                | cf_plate-detection_320_320_0.49G             | 275               | 4292.56                              |
| 34   | Inception_resnet_v2            | tf_inceptionresnetv2_imagenet_299_299_26.35G | 275               | 223.98                               |
| 35   | Inception_v1                   | tf_inceptionv1_imagenet_224_224_3G           | 275               | 1636.82                              |
| 36   | Inception_v3                   | tf_inceptionv3_imagenet_299_299_11.45G       | 275               | 550.95                               |
| 37   | Inception_v4                   | tf_inceptionv4_imagenet_299_299_24.55G       | 275               | 257.10                               |
| 38   | resnet_v1_50                   | tf_resnetv1_50_imagenet_224_224_6.97G        | 275               | 881.36                               |
| 39   | resnet_v1_101                  | tf_resnetv1_101_imagenet_224_224_14.4G       | 275               | 458.21                               |
| 40   | resnet_v1_152                  | tf_resnetv1_152_imagenet_224_224_21.83G      | 275               | 306.10                               |
| 41   | vgg_16                         | tf_vgg16_imagenet_224_224_30.96G             | 275               | 229.28                               |
| 42   | vgg_19                         | tf_vgg19_imagenet_224_224_39.28G             | 275               | 190.06                               |
| 43   | ssd_resnet_50_v1_fpn           | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 247.5             | 42.82                                |
| 44   | yolov3_voc                     | tf_yolov3_voc_416_416_65.63G                 | 247.5             | 104.65                               |
| 45   | mlperf_ssd_resnet34            | tf_mlperf_resnet34_coco_1200_1200_433G       | 275               | 19.54                                |
| 46   | torchvision                    | resnet50                                     | 275               | 752.58                               |
| 47   | torchvision                    | inception_v3                                 | 275               | 554.26                               |
| 48   | torchvision                    | squeezenet                                   | 275               | 2644.31                              |


</details>


### Performance on U50 lv10e
Measured with Vitis AI 1.2 and Vitis AI Library 1.2 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 10 DPUv3E kernels running at 275Mhz in Gen3x4:
  

| No\. | Model                          | Name                                        | Frequency \(MHz\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :----------------------------- | :------------------------------------------ | ----------------- | ------------------------------------ |
| 1    | resnet50                       | cf_resnet50_imagenet_224_224_7.7G           | 275               | 887.20                               |
| 2    | resnet18                       | cf_resnet18_imagenet_224_224_3.65G          | 275               | 2141.76                              |
| 3    | SqueezeNet                     | cf_squeeze_imagenet_227_227_0.76G           | 275               | 3793.70                              |
| 4    | ssd_pedestrain_pruned_0_97     | cf_ssdpedestrian_coco_360_640_0.97_5.9G     | 275               | 630.87                               |
| 5    | refinedet_pruned_0_8           | cf_refinedet_coco_360_480_0.8_25G           | 220               | 240.64                               |
| 6    | refinedet_pruned_0_92          | cf_refinedet_coco_360_480_0.92_10.10G       | 275               | 635.24                               |
| 7    | refinedet_pruned_0_96          | cf_refinedet_coco_360_480_0.96_5.08G        | 275               | 912.87                               |
| 8    | ssd_adas_pruned_0_95           | cf_ssdadas_bdd_360_480_0.95_6.3G            | 275               | 730.13                               |
| 9    | ssd_traffic_pruned_0_9         | cf_ssdtraffic_360_480_0.9_11.6G             | 275               | 563.29                               |
| 10   | VPGnet_pruned_0_99             | cf_VPGnet_caltechlane_480_640_0.99_2.5G     | 275               | 644.82                               |
| 11   | FPN                            | cf_fpn_cityscapes_256_512_8.9G              | 247.5             | 496.27                               |
| 12   | SP_net                         | cf_SPnet_aichallenger_224_128_0.54G         | 247.5             | 140.96                               |
| 13   | Openpose_pruned_0_3            | cf_openpose_aichallenger_368_368_0.3_189.7G | 275               | 49.13                                |
| 14   | densebox_320_320               | cf_densebox_wider_320_320_0.49G             | 275               | 2300.29                              |
| 15   | densebox_640_360               | cf_densebox_wider_360_640_1.11G             | 275               | 999.27                               |
| 16   | face_landmark                  | cf_landmark_celeba_96_72_0.14G              | 275               | 13152.50                             |
| 17   | reid                           | cf_reid_market1501_160_80_0.95G             | 275               | 5649.24                              |
| 18   | multi_task                     | cf_multitask_bdd_288_512_14.8G              | 247.5             | 134.06                               |
| 19   | yolov3_bdd                     | dk_yolov3_bdd_288_512_53.7G                 | 220               | 98.99                                |
| 20   | yolov3_adas_pruned_0_9         | dk_yolov3_cityscapes_256_512_0.9_5.46G      | 247.5             | 844.36                               |
| 21   | yolov3_voc                     | dk_yolov3_voc_416_416_65.42G                | 220               | 101.11                               |
| 22   | yolov2_voc                     | dk_yolov2_voc_448_448_34G                   | 247.5             | 251.14                               |
| 23   | yolov2_voc_pruned_0_66         | dk_yolov2_voc_448_448_0.66_11.56G           | 247.5             | 625.53                               |
| 24   | yolov2_voc_pruned_0_71         | dk_yolov2_voc_448_448_0.71_9.86G            | 247.5             | 735.86                               |
| 25   | yolov2_voc_pruned_0_77         | dk_yolov2_voc_448_448_0.77_7.82G            | 247.5             | 837.44                               |
| 26   | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G             | 275               | 1943.16                              |
| 27   | ResNet64-face                  | cf_facerec-resnet64_112_96_11G              | 275               | 737.50                               |
| 28   | FPN_Res18_Medical_segmentation | cf_FPN-resnet18_EDD_320_320_45.3G           | 247.5             | 140.96                               |
| 29   | plate detection                | cf_plate-detection_320_320_0.49G            | 275               | 4220.50                              |
| 30   | resnet_v1_50                   | tf_resnetv1_50_imagenet_224_224_6.97G       | 247.5             | 883.22                               |
| 31   | resnet_v1_101                  | tf_resnetv1_101_imagenet_224_224_14.4G      | 275               | 508.86                               |
| 32   | resnet_v1_152                  | tf_resnetv1_152_imagenet_224_224_21.83G     | 275               | 339.89                               |
| 33   | vgg_16                         | tf_vgg16_imagenet_224_224_30.96G            | 275               | 252.85                               |
| 34   | vgg_19                         | tf_vgg19_imagenet_224_224_39.28G            | 275               | 210.20                               |
| 35   | ssd_resnet_50_v1_fpn           | tf_ssdresnet50v1_fpn_coco_640_640_178.4G    | 220               | 41.56                                |
| 36   | yolov3_voc                     | tf_yolov3_voc_416_416_65.63G                | 220               | 101.55                               |
| 37   | mlperf_ssd_resnet34            | tf_mlperf_resnet34_coco_1200_1200_433G      | 275               | 23.11                                |
| 38   | torchvision                    | resnet50                                    | 275               | 830.88                               |
| 39   | torchvision                    | inception_v3                                | 275               | 591.09                               |
| 40   | torchvision                    | squeezenet                                  | 275               | 2683.77                              |


</details>


### Performance on U200
Measured with Vitis AI 1.1 and Vitis AI Library 1.1 

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
Measured with Vitis AI 1.1 and Vitis AI Library 1.1 

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
| 1    | resnet50                       | cf_resnet50_imagenet_224_224_7.7G            | 210               | 865.48                               |
| 2    | resnet18                       | cf_resnet18_imagenet_224_224_3.65G           | 210               | 2010.82                              |
| 3    | Inception_v1                   | cf_inceptionv1_imagenet_224_224_3.16G        | 210               | 1356.82                              |
| 4    | Inception_v2                   | cf_inceptionv2_imagenet_224_224_4G           | 210               | 1193.70                              |
| 5    | Inception_v3                   | cf_inceptionv3_imagenet_299_299_11.4G        | 210               | 485.59                               |
| 6    | Inception_v4                   | cf_inceptionv4_imagenet_299_299_24.5G        | 210               | 220.97                               |
| 7    | SqueezeNet                     | cf_squeeze_imagenet_227_227_0.76G            | 210               | 2965.10                              |
| 8    | ssd_pedestrain_pruned_0_97     | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 210               | 467.73                               |
| 9    | refinedet_pruned_0_8           | cf_refinedet_coco_360_480_0.8_25G            | 210               | 187.35                               |
| 10   | refinedet_pruned_0_92          | cf_refinedet_coco_360_480_0.92_10.10G        | 210               | 398.22                               |
| 11   | refinedet_pruned_0_96          | cf_refinedet_coco_360_480_0.96_5.08G         | 210               | 582.97                               |
| 12   | ssd_adas_pruned_0_95           | cf_ssdadas_bdd_360_480_0.95_6.3G             | 210               | 546.51                               |
| 13   | ssd_traffic_pruned_0_9         | cf_ssdtraffic_360_480_0.9_11.6G              | 210               | 361.55                               |
| 14   | VPGnet_pruned_0_99             | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 210               | 670.48                               |
| 15   | FPN                            | cf_fpn_cityscapes_256_512_8.9G               | 210               | 433.93                               |
| 16   | SP_net                         | cf_SPnet_aichallenger_224_128_0.54G          | 210               | 787.96                               |
| 17   | Openpose_pruned_0_3            | cf_openpose_aichallenger_368_368_0.3_189.7G  | 210               | 50.78                                |
| 18   | densebox_320_320               | cf_densebox_wider_320_320_0.49G              | 210               | 2448.54                              |
| 19   | densebox_640_360               | cf_densebox_wider_360_640_1.11G              | 210               | 1172.81                              |
| 20   | face_landmark                  | cf_landmark_celeba_96_72_0.14G               | 210               | 6624.48                              |
| 21   | reid                           | cf_reid_market1501_160_80_0.95G              | 210               | 4665.98                              |
| 22   | multi_task                     | cf_multitask_bdd_288_512_14.8G               | 210               | 135.98                               |
| 23   | yolov3_bdd                     | dk_yolov3_bdd_288_512_53.7G                  | 210               | 123.64                               |
| 24   | yolov3_adas_pruned_0_9         | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 210               | 929.92                               |
| 25   | yolov3_voc                     | dk_yolov3_voc_416_416_65.42G                 | 210               | 129.38                               |
| 26   | yolov2_voc                     | dk_yolov2_voc_448_448_34G                    | 210               | 264.76                               |
| 27   | yolov2_voc_pruned_0_66         | dk_yolov2_voc_448_448_0.66_11.56G            | 210               | 591.97                               |
| 28   | yolov2_voc_pruned_0_71         | dk_yolov2_voc_448_448_0.71_9.86G             | 210               | 648.20                               |
| 29   | yolov2_voc_pruned_0_77         | dk_yolov2_voc_448_448_0.77_7.82G             | 210               | 710.40                               |
| 30   | ResNet20-face                  | cf_facerec-resnet20_112_96_3.5G              | 210               | 2181.14                              |
| 31   | ResNet64-face                  | cf_facerec-resnet64_112_96_11G               | 210               | 801.95                               |
| 32   | FPN_Res18_Medical_segmentation | cf_FPN-resnet18_EDD_320_320_45.3G            | 210               | 153.62                               |
| 33   | plate detection                | cf_plate-detection_320_320_0.49G             | 210               | 3817.32                              |
| 34   | Inception_resnet_v2            | tf_inceptionresnetv2_imagenet_299_299_26.35G | 210               | 212.27                               |
| 35   | Inception_v1                   | tf_inceptionv1_imagenet_224_224_3G           | 210               | 1374.46                              |
| 36   | Inception_v3                   | tf_inceptionv3_imagenet_299_299_11.45G       | 210               | 485.56                               |
| 37   | Inception_v4                   | tf_inceptionv4_imagenet_299_299_24.55G       | 210               | 223.57                               |
| 38   | resnet_v1_50                   | tf_resnetv1_50_imagenet_224_224_6.97G        | 210               | 965.20                               |
| 39   | resnet_v1_101                  | tf_resnetv1_101_imagenet_224_224_14.4G       | 210               | 515.58                               |
| 40   | resnet_v1_152                  | tf_resnetv1_152_imagenet_224_224_21.83G      | 210               | 347.65                               |
| 41   | vgg_16                         | tf_vgg16_imagenet_224_224_30.96G             | 210               | 250.81                               |
| 42   | vgg_19                         | tf_vgg19_imagenet_224_224_39.28G             | 210               | 211.20                               |
| 43   | ssd_resnet_50_v1_fpn           | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 210               | 39.73                                |
| 44   | yolov3_voc                     | tf_yolov3_voc_416_416_65.63G                 | 210               | 129.38                               |
| 45   | mlperf_ssd_resnet34            | tf_mlperf_resnet34_coco_1200_1200_433G       | 210               | 11.63                                |
| 46   | torchvision                    | resnet50                                     | 210               | 877.20                               |
| 47   | torchvision                    | inception_v3                                 | 210               | 530.91                               |
| 48   | torchvision                    | squeezenet                                   | 210               | 2150.05                              |


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
| 10   | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | 23\.29                          | 42\.9333                                | 50\.8                                  |
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
