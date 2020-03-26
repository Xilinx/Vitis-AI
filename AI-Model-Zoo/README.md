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

## Model Information
The following table includes comprehensive information about each model, including application, framework, training and validation dataset, backbone, input size, computation as well as float and quantized precision.

<details>
 <summary><b>Click here to view details</b></summary>

| No\. | Application              | Model                          | Name                                                | Framework  | Backbone       | Input Size | OPS per image | Training Set                            | Val Set                 | Float \(Top1, Top5\)/ mAP/mIoU | Quantized \(Top1, Top5\)/mAP/mIoU |
| :--- | :----------------------- | :----------------------------- | :-------------------------------------------------- | :--------- | -------------- | ---------- | ------------- | --------------------------------------- | ----------------------- | ------------------------------ | --------------------------------- |
| 1    | Image Classification     | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G\_1\.1             | caffe      | resnet50       | 224\*224   | 7\.7G         | ImageNet Train                          | ImageNet Validataion    | 0\.7444/0\.9185                | 0\.7334/0\.9131                   |
| 2    | Image Classifiction      | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G\_1\.1            | caffe      | resnet18       | 224\*224   | 3\.65G        | ImageNet Train                          | ImageNet Validataion    | 0\.6832/0\.8848                | 66\.94%/88\.25%                   |
| 3    | Image Classification     | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G\_1\.1         | caffe      | inception\_v1  | 224\*224   | 3\.16G        | ImageNet Train                          | ImageNet Validataion    | 0\.7030/0\.8971                | 0\.6984/0\.8942                   |
| 4    | Image Classification     | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G\_1\.1             | caffe      | bn\-inception  | 224\*224   | 4G            | ImageNet Train                          | ImageNet Validataion    | 0\.7275/0\.9111                | 0\.7168/0\.9029                   |
| 5    | Image Classification     | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G\_1\.1         | caffe      | inception\_v3  | 299\*299   | 11\.4G        | ImageNet Train                          | ImageNet Validataion    | 0\.7701/0\.9329                | 0\.7626/0\.9303                   |
| 6    | Image Classification     | Inception\_v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G\_1\.1         | caffe      | inception\_v3  | 299\*299   | 24\.5G        | ImageNet Train                          | ImageNet Validataion    | 0\.7958/0\.9470                | 0\.7898/0\.9445                   |
| 7    | Image Classification     | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G\_1\.1         | caffe      | MobileNet\_v2  | 224\*224   | 608M          | ImageNet Train                          | ImageNet Validataion    | 0\.6475/0\.8609                | 0\.6354/0\.8506                   |
| 8    | Image Classifiction      | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G\_1\.1             | caffe      | squeezenet     | 227\*227   | 0\.76G        | ImageNet Train                          | ImageNet Validataion    | 0\.5438/0\.7813                | 0\.5026/0\.7658                   |
| 9    | ADAS Pedstrain Detection | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G\_1\.1     | caffe      | VGG\-bn\-16    | 360\*640   | 5\.9G         | coco2014\_train\_person and crowndhuman | coco2014\_val\_person   | 0\.5903                        | 0\.5876                           |
| 10   | Object Detection         | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G\_1\.1            | caffe      | VGG\-bn\-16    | 360\*480   | 25G           | coco2014\_train\_person                  | coco2014\_val\_person   | 0\.6794                        | 0\.6780                           |
| 11   | Object Detection         | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G\_1\.1       | caffe      | VGG\-bn\-16    | 360\*480   | 10\.10G       | coco2014\_train\_person                 | coco2014\_val\_person   | 0\.6489                        | 0\.6486                           |
| 12   | Object Detection         | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G\_1\.1        | caffe      | VGG\-bn\-16    | 360\*480   | 5\.08G        | coco2014\_train\_person                 | coco2014\_val\_person   | 0\.6120                        | 0\.6113                           |
| 13   | ADAS Vehicle Detection   | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G\_1\.1            | caffe      | VGG\-16        | 360\*480   | 6\.3G         | bdd100k \+ private data                 | bdd100k \+ private data | 0\.4207                        | 0\.4200                           |
| 14   | Traffic Detection        | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G\_1\.1              | caffe      | VGG\-16        | 360\*480   | 11\.6G        | private data                            | private data            | 0\.5982                        | 0\.5921                           |
| 15   | ADAS Lane Detection      | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G\_1\.1     | caffe      | VGG            | 480\*640   | 2\.5G         | caltech\-lanes\-train\-dataset          | caltech lane            | 0\.8864\(F1\-score\)            | 0\.8882\(F1\-score\)              |
| 16   | Object Detection         | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G\_1\.1           | caffe      | MobileNet\_v2  | 360\*480   | 6\.57G        | bdd100k train                           | bdd100k val             | 0\.3052                        | 0\.2752                           |
| 17   | ADAS Segmentation        | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G\_1\.1                | caffe      | Google\_v1\_BN | 256\*512   | 8\.9G         | Cityscapes gtFineTrain\(2975\)          | Cityscapes Val\(500\)   | 0\.5669                        | 0\.5662                           |
| 18   | Pose Estimation          | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G\_1\.1           | caffe      | Google\_v1\_BN | 128\*224   | 548\.6M       | ai\_challenger                          | ai\_challenger          | 0\.9000\(PCKh0\.5\)            | 0\.8964\(PCKh0\.5\)               |
| 19   | Pose Estimation          | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G\_1\.1 | caffe      | VGG            | 368\*368   | 49\.88G       | ai\_challenger                          | ai\_challenger          | 0\.4507\(OKs\)                  | 0\.4422\(Oks\)                    |
| 20   | Face Detection           | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G\_1\.1               | caffe      | VGG\-16        | 320\*320   | 0\.49G        | wider\_face                             | FDDB                    | 0\.8833                        | 0\.8791                           |
| 21   | Face Detection           | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G\_1\.1               | caffe      | VGG\-16        | 360\*640   | 1\.11G        | wider\_face                             | FDDB                    | 0\.8931                        | 0\.8925                           |
| 22   | Face Recognition         | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G\_1\.1                | caffe      | lenet          | 96\*72     | 0\.14G        | celebA                                  | processed helen         | 0\.1952\(L2 loss\)              | 0\.1972\(L2 loss\)                |
| 23   | Re\-identification       | reid                           | cf\_reid\_market1501\_160\_80\_0\.95G\_1\.1               | caffe      | resnet18       | 160\*80    | 0\.95G        | Market1501\+CUHK03                      | Market1501              | 0\.7800                        | 0\.7790                           |
| 24   | Detection+Segmentation   | multi-task                     | cf\_multitask\_bdd\_288\_512\_14\.8G\_1\.1                | caffe      | ssd            | 288\*512   | 14\.8G        | BDD100K+Cityscapes                      | BDD100K+Cityscapes      | 0\.2228(Det) 0\.4088(Seg)       | 0\.2202(Det) 0\.4058(Seg)           |
| 25   | Object Detection         | yolov3\_bdd                    | dk\_yolov3\_bdd\_288\_512\_53\.7G\_1\.1                   | darknet    | darknet\-53    | 288\*512   | 53\.7G        | bdd100k                                 | bdd100k                 | 0\.5058                        | 0\.4914                           |
| 26   | Object Detection         | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G\_1\.1      | darknet    | darknet\-53    | 256\*512   | 5\.46G        | cityscape train                         | cityscape val           | 0\.5520                        | 0\.5300                           |
| 27   | Object Detection         | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G\_1\.1                  | darknet    | darknet\-53    | 416\*416   | 65\.42G       | voc07\+12\_trainval                     | voc07\_test             | 0\.8240\(MaxIntegral\)          | 0\.8150\(MaxIntegral\)            |
| 28   | Object Detection         | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G\_1\.1                      | darknet    | darknet\-19    | 448\*448   | 34G           | voc07\+12\_trainval                     | voc07\_test             | 0\.7845\(MaxIntegral\)          | 0\.7739\(MaxIntegral\)            |
| 29   | Object Detection         | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G\_1\.1           | darknet    | darknet\-19    | 448\*448   | 11\.56G       | voc07\+12\_trainval                     | voc07\_test             | 0\.7700\(MaxIntegral\)          | 0\.7600\(MaxIntegral\)            |
| 30   | Object Detection         | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G\_1\.1            | darknet    | darknet\-19    | 448\*448   | 9\.86G        | voc07\+12\_trainval                     | voc07\_test             | 0\.7670\(MaxIntegral\)          | 0\.7530\(MaxIntegral\)            |
| 31   | Object Detection         | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G\_1\.1            | darknet    | darknet\-19    | 448\*448   | 7\.82G        | voc07\+12\_trainval                     | voc07\_test             | 0\.7576\(MaxIntegral\)          | 0\.7460\(MaxIntegral\)            |
| 32   | Image Classifiction      | Inception\_resnet\_v2          | tf\_inceptionresnetv2\_imagenet\_299\_299\_26\.35G\_1\.1  | tensorflow | inception      | 299\*299   | 26\.35G       | ImageNet Train                          | ImageNet Validataion    | 0\.8037                        | 0\.7946                           |
| 33   | Image Classifiction      | Inception\_v1                  | tf\_inceptionv1\_imagenet\_224\_224\_3G\_1\.1             | tensorflow | inception      | 224\*224   | 3G            | ImageNet Train                          | ImageNet Validataion    | 0\.6976                        | 0\.6794                           |
| 34   | Image Classifiction      | Inception\_v3                  | tf\_inceptionv3\_imagenet\_299\_299\_11\.45G\_1\.1        | tensorflow | inception      | 299\*299   | 11\.45G       | ImageNet Train                          | ImageNet Validataion    | 0\.7798                        | 0\.7607                           |
| 35   | Image Classifiction      | Inception\_v4                  | tf\_inceptionv4\_imagenet\_299\_299\_24\.55G\_1\.1        | tensorflow | inception      | 299\*299   | 24\.55G       | ImageNet Train                          | ImageNet Validataion    | 0\.8018                        | 0\.7928                           |
| 36   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_0\.25\_imagenet\_128\_128\_27\.15M\_1\.1 | tensorflow | mobilenet      | 128\*128   | 27\.15M       | ImageNet Train                          | ImageNet Validataion    | 0\.4144                        | 0\.3464                           |
| 37   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_0\.5\_imagenet\_160\_160\_150\.07M\_1\.1 | tensorflow | mobilenet      | 160\*160   | 150\.07M      | ImageNet Train                          | ImageNet Validataion    | 0\.5903                        | 0\.5195                           |
| 38   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_1\.0\_imagenet\_224\_224\_1\.14G\_1\.1   | tensorflow | mobilenet      | 224\*224   | 1\.14G        | ImageNet Train                          | ImageNet Validataion    | 0\.7102                        | 0\.6779                           |
| 39   | Image Classifiction      | Mobilenet\_v2                  | tf\_mobilenetv2\_1\.0\_imagenet\_224\_224\_0\.59G\_1\.1   | tensorflow | mobilenet      | 224\*224   | 0\.59G        | ImageNet Train                          | ImageNet Validataion    | 0\.7013                        | 0\.6767                           |
| 40   | Image Classifiction      | Mobilenet\_v2                  | tf\_mobilenetv2\_1\.4\_imagenet\_224\_224\_1\.16G\_1\.1   | tensorflow | mobilenet      | 224\*224   | 1\.16G        | ImageNet Train                          | ImageNet Validataion    | 0\.7411                        | 0\.7194                           |
| 41   | Image Classifiction      | resnet\_v1\_50                 | tf\_resnetv1\_50\_imagenet\_224\_224\_6\.97G\_1\.1        | tensorflow | resnetv1       | 224\*224   | 6\.97G        | ImageNet Train                          | ImageNet Validataion    | 0\.7520                        | 0\.7423                           |
| 42   | Image Classifiction      | resnet\_v1\_101                | tf\_resnetv1\_101\_imagenet\_224\_224\_14\.4G\_1\.1       | tensorflow | resnetv1       | 224\*224   | 14\.4G        | ImageNet Train                          | ImageNet Validataion    | 0\.7640                        | 0\.7417                           |
| 43   | Image Classifiction      | resnet\_v1\_152                | tf\_resnetv1\_152\_imagenet\_224\_224\_21\.83G\_1\.1      | tensorflow | resnetv1       | 224\*224   | 21\.83G       | ImageNet Train                          | ImageNet Validataion    | 0\.7681                        | 0\.7463                           |
| 44   | Image Classifiction      | vgg\_16                        | tf\_vgg16\_imagenet\_224\_224\_30\.96G\_1\.1              | tensorflow | vgg            | 224\*224   | 30\.96G       | ImageNet Train                          | ImageNet Validataion    | 0\.7089                        | 0\.7069                           |
| 45   | Image Classifiction      | vgg\_19                        | tf\_vgg19\_imagenet\_224\_224\_39\.28G\_1\.1              | tensorflow | vgg            | 224\*224   | 39\.28G       | ImageNet Train                          | ImageNet Validataion    | 0\.7100                        | 0\.7026                           |
| 46   | Object Detection         | ssd\_mobilenet\_v1             | tf\_ssdmobilenetv1\_coco\_300\_300\_2\.47G\_1\.1          | tensorflow | mobilenet      | 300\*300   | 2\.47G        | coco2017                                | coco2014 minival        | 0\.2080                        | 0\.1960                           |
| 47   | Object Detection         | ssd\_mobilenet\_v2             | tf\_ssdmobilenetv2\_coco\_300\_300\_3\.75G\_1\.1          | tensorflow | mobilenet      | 300\*300   | 3\.75G        | coco2017                                | coco2014 minival        | 0\.2150                        | 0\.2030                           |
| 48   | Object Detection         | ssd\_resnet50\_v1\_fpn         | tf\_ssdresnet50v1\_fpn\_coco\_640\_640\_178\.4G\_1\.1     | tensorflow | resnet50       | 300\*300   | 178\.4G       | coco2017                                | coco2014 minival        | 0\.3010                        | 0\.2900                           |
| 49   | Object Detection         | yolov3\_voc                    | tf\_yolov3\_voc\_416\_416\_65\.63G\_1\.1                  | tensorflow | darknet\-53    | 416\*416   | 65\.63G       | voc07\+12\_trainval                     | voc07\_test             | 0\.7846                        | 0\.7744                           |
| 50   | Object Detection         | mlperf\_ssd\_resnet34          | tf\_mlperf_resnet34\_coco\_1200\_1200\_433G\_1\.1         | tensorflow | resnet34       | 1200\*1200 | 433G          | coco2017                                | coco2017                | 0\.2250                        | 0\.2110                           |

</details>

### Naming Rules
Model name: `F_M_D_H_W_(P)_C_V`
* `F` specifies training framework: `cf` is Caffe, `tf` is Tensorflow, `dk` is Darknet, `pt` is PyTorch
* `M` specifies the model
* `D` specifies the dataset
* `H` specifies the height of input data
* `W` specifies the width of input data
* `P` specifies the pruning ratio, it means how much computation is reduced. It is optional depending on whether the model is pruned.
* `C` specifies the computation of the model: how many Gops per image
* `V` specifies the version of Vitis AI


For example, `cf_refinedet_coco_480_360_0.8_25G_1.1` is a `RefineDet` model trained with `Caffe` using `COCO` dataset, input data size is `480*360`, `80%` pruned, the computation per image is `25Gops` and Vitis AI version is `1.1`.


### caffe-xilinx 
This is a custom distribution of caffe. Please use caffe-xilinx to test/finetune the caffe models listed in this page.

**Note:** To download caffe-xlinx, visit [caffe-xilinx.zip](https://www.xilinx.com/bin/public/openDownload?filename=caffe-xilinx-1.1.zip)



## Model Download
The following table lists various models, download link and MD5 checksum for the zip file of each model.

**Note:** To download all the models, visit [all_models_1.1.zip](https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.1.zip).

<details>
 <summary><b>Click here to view details</b></summary>

If you are a:
 - Linux user, use the [`get_model.sh`](reference-files/get_model.sh) script to download all the models.   
 - Windows user, use the download link listed in the following table to download a model.


| No\. | Model                                        | Size      | Download link | Checksum |
| ---- | -------------------------------------------- | --------- | ------------- | -------- |
| 1    | cf_resnet50_imagenet_224_224_7.7G_1.1            |203.99 MB    |https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet50_imagenet_224_224_1.1.zip                  |fe1fcbbdc935dc5cdf75a95780f8983e          |
| 2    | cf_inceptionv1_imagenet_224_224_3.16G_1.1        |79.41 MB     |https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv1_imagenet_224_224_1.1.zip               |17417893abee5c489c25de0420d927b6          |
| 3    | cf_inceptionv2_imagenet_224_224_4G_1.1           |128.63 MB    |https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv2_imagenet_224_224_1.1.zip               |ae52f235af9f1e21aee8aa20d68905f5          |
| 4    | cf_inceptionv3_imagenet_299_299_11.4G_1.1        |190.74 MB    |https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv3_imagenet_299_299_1.1.zip               |b44315d620c4a9e5266763f34f44b7c8          |
| 5    | cf_inceptionv4_imagenet_299_299_24.5G_1.1        |341.64 MB    |https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv4_imagenet_299_299_1.1.zip               |5df3ae9c4daf6f3276612de579239359          |
| 6    | cf_mobilenetv2_imagenet_224_224_0.59G_1.1        |19.67 MB     |https://www.xilinx.com/bin/public/openDownload?filename=cf_mobilenetv2_imagenet_224_224_1.1.zip               |1dbae9a4a8f968ffba665fe67af1ceb6          |
| 7    | cf_squeeze_imagenet_227_227_0.76G_1.1            |10.04 MB     |https://www.xilinx.com/bin/public/openDownload?filename=cf_squeeze_imagenet_227_227_1.1.zip                   |9da53315ea5bfdde385c57e3a451afd5          |
| 8    | cf_resnet18_imagenet_224_224_3.65G_1.1           |130.7MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet18_imagenet_224_224_1.1.zip                  |f8a926550af500b0848db74e8e2e1381          |
| 9    | cf_ssdpedestrian_coco_360_640_0.97_5.9G_1.1      |5.96 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdpedestrian_coco_360_640_0.97_1.1.zip            |0f9ad09d11e250b8a43aa110084090a2          |
| 10   | cf_refinedet_coco_360_480_0.8_25G_1.1            |92.9 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_0.8_1.1.zip                 |a0a5c91b9f641c71727b786ea22ae018          |
| 11   | cf_refinedet_coco_360_480_0.92_10.10G_1.1        |8.12 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_0.92_1.1.zip                |02cbc7ea6b8c2b668af3e6f912303315          |
| 12   | cf_refinedet_coco_360_480_0.96_5.08G_1.1         |4.28 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_0.96_1.1.zip                |1756e7e8c9ec30e5948076acadbed811          |
| 13   | cf_ssdadas_bdd_360_480_0.95_6.3G_1.1             |10.71 MB     |https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdadas_bdd_360_480_0.95_1.1.zip                   |e5a0f9b3b6e2c72aa8961e279a3cef11          |
| 14   | cf_ssdtraffic_360_480_0.9_11.6G_1.1              |15.62 MB     |https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdtraffic_360_480_0.9_1.1.zip                     |a8d6a9db2bb40b16cc1de435709bf570          |
| 15   | cf_VPGnet_caltechlane_480_640_0.99_2.5G_1.1      |5.97 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_VPGnet_caltechlane_480_640_0.99_1.1.zip            |02bd13eaa6d7d4e1b5a4fd0280c8d2e1          |
| 16   | cf_ssdmobilenetv2_bdd_360_480_6.57G_1.1          |77 MB        |https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdmobilenetv2_bdd_360_480_1.1.zip                 |868b430065d39cf60f3ca0662e8e8b9e          |
| 17   | cf_fpn_cityscapes_256_512_8.9G_1.1               |69.27MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_fpn_cityscapes_256_512_1.1.zip                     | 37b49ee32d0b8974ce2787fff6eebca2         |
| 18   | cf_SPnet_aichallenger_224_128_0.54G_1.1          |11 MB        |https://www.xilinx.com/bin/public/openDownload?filename=cf_SPnet_aichallenger_224_128_1.1.zip                 |13f4c7702b94aeb7bcb2dfecc29c5e87          |
| 19   | cf_openpose_aichallenger_368_368_0.3_189.7G_1.1  |408.42 MB    |https://www.xilinx.com/bin/public/openDownload?filename=cf_openpose_aichallenger_368_368_0.3_1.1.zip          |446b135dc57408868f49de7a61142dd7          |
| 20   | cf_densebox_wider_320_320_0.49G_1.1              |4.33 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_densebox_wider_320_320_1.1.zip                     |83ca356836a21d37affd97b129b636ff          |
| 21   | cf_densebox_wider_360_640_1.11G_1.1              |4.33 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_densebox_wider_360_640_1.1.zip                     |7f2e1e2599260a5dd0ab43df6ee1ca61          |
| 22   | cf_landmark_celeba_96_72_0.14G_1.1               |45.25 MB     |https://www.xilinx.com/bin/public/openDownload?filename=cf_landmark_celeba_96_72_1.1.zip                      |8adfd25d1a3225fe0fdd2152c170e4e3          |
| 23   | cf_reid_market1501_160_80_0.95G_1.1              |88.96 MB     |https://www.xilinx.com/bin/public/openDownload?filename=cf_reid_market1501_160_80_1.1.zip                     |4ef83b5e9ef601d3ad99ca71b19bab26          |
| 24   | cf_multitask_bdd_288_512_14.8G_1.1               |108.75 MB    |https://www.xilinx.com/bin/public/openDownload?filename=cf_multitask_bdd_288_512_1.1.zip                      |3d35e68e71cd8911e40c3d57fb7328e5          |
| 25   | dk_yolov3_bdd_288_512_53.7G_1.1                  |695.59 MB    |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_bdd_288_512_1.1.zip                         |346a0810ed3f5d8f004f74934d9a8464          |
| 26   | dk_yolov3_cityscapes_256_512_0.9_5.46G_1.1       |26.96 MB     |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_cityscapes_256_512_0.9_1.1.zip              |63e1a349a35188e9e08136072750690e          |
| 27   | dk_yolov3_voc_416_416_65.42G_1.1                 |689.58 MB    |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_voc_416_416_1.1.zip                         |7bb0afd4766f46d298e07fcc595a6f4a          |
| 28   | dk_yolov2_voc_448_448_34G_1.1                    |410.57 MB    |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_1.1.zip                         |8f7c48f9c5da0d652fef7091c1c11c34          |
| 29   | dk_yolov2_voc_448_448_0.66_11.56G_1.1            |167.56 MB    |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.66_1.1.zip                    |2184ea9b85c94dd04cdd05c4556bab8e          |
| 30   | dk_yolov2_voc_448_448_0.71_9.86G_1.1             |151.83 MB    |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.71_1.1.zip                    |33b830f11b6f89d351b465705cd67e3f          |
| 31   | dk_yolov2_voc_448_448_0.77_7.82G_1.1             |110.02 MB    |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.77_1.1.zip                    |1619814a35f439836ef69a5c34249af1          |
| 32   | tf_inceptionresnetv2_imagenet_299_299_26.35G_1.1 |446.69 MB    |https://www.xilinx.com/bin/public/openDownload?filename=tf_inception_resnet_v2_imagenet_299_299_1.1.zip       |95b6f5bbc7c0b7772fc55963e2d4bb47          |
| 33   | tf_inceptionv1_imagenet_224_224_3G_1.1           |53.43 MB     |https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv1_imagenet_224_224_1.1.zip               |935705a7a371f508953128ba3d07fda5          |
| 34   | tf_inceptionv3_imagenet_299_299_11.45G_1.1       |191.14 MB    |https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv3_imagenet_299_299_1.1.zip               |c12517b59ab5d05c40f501072bfbd270          |
| 35   | tf_inceptionv4_imagenet_299_299_24.55G_1.1       |342.44 MB    |https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv4_imagenet_299_299_1.1.zip               |b8dc266e3261c32ebc18de0be9a84a00          |
| 36   | tf_mobilenetv1_0.25_imagenet_128_128_27.15M_1.1  |3.84 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_0.25_imagenet_128_128_1.1.zip          |90c3f9e0fe33c52f4bd5e3c33e78c22d          |
| 37   | tf_mobilenetv1_0.5_imagenet_160_160_150.07M_1.1  |10.67 MB     |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_0.5_imagenet_160_160_1.1.zip           |132e6ac454918ae82e0024ee976a2c49          |
| 38   | tf_mobilenetv1_1.0_imagenet_224_224_1.14G_1.1    |33.38 MB     |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_1.0_imagenet_224_224_1.1.zip           |077deb4d3fdf2d42765b8fe55e251f76          |
| 39   | tf_mobilenetv2_1.0_imagenet_224_224_0.59G_1.1    |28.31 MB     |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv2_1.0_imagenet_224_224_1.1.zip           |032ca92d995ff4b8dc654566f48024b0          |
| 40   | tf_mobilenetv2_1.4_imagenet_224_224_1.16G_1.1    |49.07 MB     |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv2_1.4_imagenet_224_224_1.1.zip           |80d41f50d67d4674d63852968675d462          |
| 41   | tf_resnetv1_50_imagenet_224_224_6.97G_1.1        |204.47 MB    |https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_50_imagenet_224_224_1.1.zip               |56336c2b37584bac07b0c5aa2b6e408e          |
| 42   | tf_resnetv1_101_imagenet_224_224_14.4G_1.1       |356.08 MB    |https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_101_imagenet_224_224_1.1.zip              |3fccc5a329fd77bf82717454628e78e4          |
| 43   | tf_resnetv1_152_imagenet_224_224_21.83G_1.1      |481.89 MB    |https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_152_imagenet_224_224_1.1.zip              |8350c88145eac9bc82a7f5a1998580a3          |
| 44   | tf_vgg16_imagenet_224_224_30.96G_1.1             |1.1 GB       |https://www.xilinx.com/bin/public/openDownload?filename=tf_vgg16_imagenet_224_224_1.1.zip                     |be883c1bf5be7e5ee889144035e48600          |
| 45   | tf_vgg19_imagenet_224_224_39.28G_1.1             |1.14 GB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_vgg19_imagenet_224_224_1.1.zip                     |885ec350b6d913e5ddd3f1b38190440b          |
| 46   | tf_ssdmobilenetv1_coco_300_300_2.47G_1.1         |53.59 MB     |https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdmobilenetv1_coco_300_300_1.1.zip                |7693181b6a74bfc3eb6edf93d2e484ff          |
| 47   | tf_ssdmobilenetv2_coco_300_300_3.75G_1.1         |129.96 MB    |https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdmobilenetv2_coco_300_300_1.1.zip                |1f2efc1eb61d4bf43ca814944fd20c6f          |
| 48   | tf_ssdresnet50v1_fpn_coco_640_640_178.4G_1.1     |373.07 MB    |https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdresnet50_fpn_coco_640_640_1.1.zip               |f9b4832e9abe06c8dffff2b6d6e12323          |
| 49   | tf_yolov3_voc_416_416_65.63G_1.1                 |500.1 MB     |https://www.xilinx.com/bin/public/openDownload?filename=tf_yolov3_voc_416_416_1.1.zip                         |3c592d349dfeb0c8807409e34cce9145          |
| 50   | tf_mlperf_resnet34_coco_1200_1200_433G_1.1       |236.3 MB     |https://www.xilinx.com/bin/public/openDownload?filename=tf_mlperf_resnet34_coco_1200_1200_1.1.zip             |29a689286036842c34405b42372ad6a3          |
| -    | all_models_1.1                                   |9.86 GB      |https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.1.zip                                    |10bedfa99692c5d0e7f840d23d0cd8d0          |


</details>

### Model Directory Structure
Download and extract the model archive to your working area on the local hard disk. For details on the various models, their download link and MD5 checksum for the zip file of each model, see [Model Download](#model-download).

#### caffe Model Directory Structure
For a caffe model, you should see the following directory structure:

    ├── code                            # Contains code 
    │   ├── test                        # Contains test code which can run demo and evaluate model performance.
    │   └── train                       # Contains training code 
    │                                     
    │                                   
    ├── readme.md                       # Contains the environment requirements, data preprocess and model information.
    │                                     Refer this to know that how to test and train the model with scripts.
    │                                        
    ├── data                            # Contains the dataset that used for model test and training.
    │                                     When test or training script runs successfully, dataset will be automatically placed in it.
    │                                                       
    ├── quantized                             
    │   ├── deploy.caffemodel           # Quantized weights, the output of vai_q_caffe without modification.
    │   ├── deploy.prototxt             # Quantized prototxt, the output of vai_q_caffe without modification.
    │   ├── quantized_test.prototxt     # Used to run evaluation with quantized_train_test.caffemodel on GPU
    │   │                                 using python test code released in near future. Some models don't have this file 
    │   │                                 if they are converted from Darknet (Yolov2, Yolov3),
    │   │                                 Pytorch (ReID) or there is no Caffe Test (Densebox).                                 
    │   ├── quantized_train_test.caffemodel   # Quantized weights can be used for quantizeded-point training and evaluation.    
    │   └── quantized_train_test.prototxt     # Used for quantized-point training and testing with quantized_train_test.caffemodel
    │                                           on GPU when datalayer modified to user's data path.
    └── float                           
        ├── float.caffemodel            # Trained float-point weights.
        ├── float.prototxt              # Modified test.prototxt as the input to vai_q_caffe along 
        │                                 with float.caffemodel. vai_q_caffe is Xilinx quantization tool
        │                                 which quantizes float-point to quantized-point model with minimal
        │                                 accuracy loss.
        ├── test.prototxt               # Used to run evaluation with python test codes released in near future.    
        └── trainval.prorotxt           # Used for training and testing with caffe train/test command
                                          when datalayer modified to user's data path. Some models don't
                                          have this file if they are converted from Darknet (Yolov2, Yolov3),
                                          Pytorch (ReID) or there is no Caffe Test (Densebox).          


#### Tensorflow Model Directory Structure
For a Tensorflow model, you should see the following directory structure:


    ├── code                            # Contains code 
    │   └── test                        # Contains test code which can run demo and evaluate model performance.
    │
    ├── readme.md                       # Contains the environment requirements, data preprocess and model information.
    │                                     Refer this to know that how to test the model with scripts.
    │
    ├── data                            # Contains the dataset that used for model test and training.
    │                                     When test or training script runs successfully, dataset will be automatically placed in it.
    │
    ├── quantized                          
    │   ├── deploy.model.pb             # Quantized model for the compiler (extended Tensorflow format).
    │   └── quantize_eval_model.pb      # Quantized model for evaluation.
    │
    └── float                             
        └── frozen.pb                   # Float-point frozen model, the input to the `vai_q_tensorflow`.


**Note:** For more information on `vai_q_caffe` and `vai_q_tensorflow`, see the [Vitis AI User Guide](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf).


## Model Performance
All the models in the Model Zoo have been deployed on Xilinx hardware with [Vitis AI](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Library). The performance number including end-to-end throughput and latency for each model on various boards with different DPU configurations are listed in the following sections.

For more information about DPU, see [DPU IP Product Guide](https://www.xilinx.com/cgi-bin/docs/ipdoc?c=dpu;v=latest;d=pg338-dpu.pdf).


**Note:** The model performance number listed in the following sections is generated with Vitis AI v1.1 and Vitis AI Lirary v1.1. For each board, a different DPU configuration is used. Vitis AI and Vitis AI Library can be downloaded for free from [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library Github](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Library).
We will continue to improve the performance with Vitis AI. The performance number reported here is subject to change in the near future.

### Performance on ZCU102 (0432055-04)  
This version of ZCU102 is out of stock. The performance number shown below was measured with the previous AI SDK v2.0.4.

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
Measured with Vitis AI 1.1 and Vitis AI Library 1.1  

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-05)` board with a `3 * B4096  @ 281MHz   V1.4.1` DPU configuration:


| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 13.74                            | 72.8                                  | 155.5                                |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 5.72                             | 174.7                                | 461.6                                |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 6.00                             | 166.6                                | 444.0                                |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 7.31                             | 136.7                                | 335.6                                |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 17.84                            | 56.0                                  | 138.5                                |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 35.54                            | 28.1                                  | 71.3                                 |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G        | 4.66                             | 214.7                                | 580.6                                |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 3.71                             | 269.4                                | 1045.9                               |
| 9    | ssd_pedestrain_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 13.19                            | 75.7                                  | 294.4                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G            | 31.57                            | 31.7                                  | 103.8                                 |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G        | 16.71                            | 59.8                                  | 204.0                                |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G         | 12.10                            | 82.6                                  | 290.1                                |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G             | 12.10                            | 82.6                                  | 296.3                                |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G              | 18.39                            | 54.4                                  | 206.7                                |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 9.77                             | 102.3                                | 397.3                                |
| 16   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G          | 26.04                            | 38.4                                  | 112.5                                |
| 17   | FPN                        | cf_fpn_cityscapes_256_512_8.9G               | 16.74                            | 59.7                                  | 185.2                                |
| 18   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G          | 2.03                             | 491.6                                | 1422.8                               |
| 19   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G  | 286.10                           | 3.5                                  | 15.3                                   |
| 20   | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 2.57                             | 388.3                                | 1279                               |
| 21   | densebox_640_360           | cf_densebox_wider_360_640_1.11G              | 5.13                             | 195.0                                | 627.8                                |
| 22   | face_landmark              | cf_landmark_celeba_96_72_0.14G               | 1.18                             | 846.7                                | 1379.9                                 |
| 23   | reid                       | cf_reid_market1501_160_80_0.95G              | 2.76                             | 361.9                                | 672.8                                |
| 24   | multi_task                 | cf_multitask_bdd_288_512_14.8G               | 28.26                            | 35.4                                  | 133.0                                |
| 25   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 77.12                            | 13.0                                  | 37.1                                 |
| 26   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 11.93                            | 83.8                                    | 235.3                                |
| 27   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 73.82                            | 13.5                                  | 38.2                                 |
| 28   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                    | 40.28                            | 24.8                                  | 77.1                                 |
| 29   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G            | 18.78                            | 53.2                                  | 194.2                                |
| 30   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G             | 16.71                            | 59.8                                  | 224.0                                |
| 31   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G             | 14.71                            | 68.0                                  | 266.2                                |
| 32   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G | 42.81                            | 23.3                                  | 51.0                                 |
| 33   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G           | 5.81                             | 172.1                                | 455.6                                |
| 34   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G       | 17.90                            | 55.9                                  | 136.8                                |
| 35   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G       | 35.56                            | 28.1                                  | 71.4                                 |
| 36   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M  | 1.18                             | 848.6                                | 2260.9                               |
| 37   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M  | 1.73                             | 577.2                                | 1913.7                               |
| 38   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G    | 3.83                             | 261.3                                | 788.6                                |
| 39   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G    | 4.57                             | 218.6                                | 598.7                                |
| 40   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G    | 6.15                             | 162.4                                | 412.0                                |
| 41   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G        | 12.76                            | 78.3                                  | 164.7                                |
| 42   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G       | 23.00                            | 43.5                                  | 94.5                                 |
| 43   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G      | 33.43                            | 29.9                                  | 66.1                                 |
| 44   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G             | 50.43                            | 19.8                                  | 44.7                                 |
| 45   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G             | 58.35                            | 17.1                                  | 40.3                                 |
| 46   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G         | 11.46                            | 87.2                                  | 323.5                                |
| 47   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G         | 15.85                            | 63.0                                  | 198.7                                |
| 48   | ssd_resnet_50_v1_fpn        | tf_ssdresnet50v1_fpn_coco_640_640_178.4G    | 745.55                           | 1.3                                  | 5.0                                  |
| 49   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                 | 74.24                            | 13.5                                  | 37.8                                 |
| 50   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G       | 547.28                           | 1.8                                  | 7.4                                    |

</details>


### Performance on ZCU104
Measured with Vitis AI 1.1 and Vitis AI Library 1.1 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 300MHz   V1.4.1` DPU configuration:


| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 12.46                            | 80.2                                   | 146.8                                |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 5.08                             | 196.7                                 | 403.7                                |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 5.27                             | 189.8                                 | 387.0                                |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 6.55                             | 152.7                                 | 298.2                                |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 16.51                            | 60.5                                   | 117.3                                |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 33.10                            | 30.2                                   | 58.6                                 |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G        | 4.01                             | 249.3                                 | 536.6                                |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 3.64                             | 274.4                                 | 941.8                                |
| 9    | ssd_pedestrain_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 12.80                            | 78.1                                   | 221.5                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G            | 30.68                            | 32.6                                   | 76.1                                 |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G        | 16.32                            | 61.3                                   | 154.2                                |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G         | 11.96                            | 83.6                                   | 228.7                                |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G             | 11.80                            | 84.7                                   | 231.9                                |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G              | 17.80                            | 56.1                                   | 153.2                                |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 9.47                             | 105.5                                 | 364.9                                |
| 16   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G          | 39.29                            | 25.4                                   | 101.3                                |
| 17   | FPN                        | cf_fpn_cityscapes_256_512_8.9G               | 16.12                            | 62                                     | 169.9                                |
| 18   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G          | 1.81                             | 552.4                                 | 1245.6                               |
| 19   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G  | 274.46                           | 3.6                                   | 11.0                                 |
| 20   | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 2.52                             | 397.4                                 | 1250.3                               |
| 21   | densebox_640_360           | cf_densebox_wider_360_640_1.11G              | 5.03                             | 198.7                                 | 606.6                                |
| 22   | face_landmark              | cf_landmark_celeba_96_72_0.14G               | 1.12                             | 890.1                                 | 1363.2                               |
| 23   | reid                       | cf_reid_market1501_160_80_0.95G              | 2.59                             | 385.6                                 | 668.8                                |
| 24   | multi_task                 | cf_multitask_bdd_288_512_14.8G               | 27.76                            | 36.0                                     | 108.4                              |
| 25   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 73.62                            | 13.6                                   | 28.7                                 |
| 26   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 11.83                            | 84.5                                   | 218.5                                |
| 27   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 70.27                            | 14.2                                   | 29.5                                 |
| 28   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                    | 38.17                            | 26.2                                   | 59.1                                 |
| 29   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G            | 17.99                            | 55.6                                   | 153.2                                |
| 30   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G             | 16.02                            | 62.4                                   | 180.2                                |
| 31   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G             | 14.17                            | 70.5                                   | 217.4                                |
| 32   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G | 39.35                            | 25.4                                   | 46.1                                 |
| 33   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G           | 5.10                             | 196.1                                 | 401.7                                |
| 34   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G       | 16.57                            | 60.3                                   | 116.4                                |
| 35   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G       | 33.13                            | 30.2                                   | 58.6                                 |
| 36   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M  | 0.79                             | 1263.6                                 | 3957.7                               |
| 37   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M  | 1.31                             | 763.1                                 | 2038.1                               |
| 38   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G    | 3.21                             | 311.8                                 | 731.1                                |
| 39   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G    | 3.98                             | 250.9                                 | 546.6                                |
| 40   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G    | 5.39                             | 185.5                                 | 381.5                                |
| 41   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G        | 11.59                            | 86.3                                   | 157.5                                |
| 42   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G       | 21.15                            | 47.3                                   | 87.2                                 |
| 43   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G      | 30.80                            | 32.5                                   | 60.1                                 |
| 44   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G             | 46.99                            | 21.3                                   | 38.3                                 |
| 45   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G             | 54.41                            | 18.4                                   | 33.8                                 |
| 46   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G         | 10.82                            | 92.4                                   | 330.0                                |
| 47   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G         | 14.99                            | 66.7                                   | 185.0                                |
| 48   | ssd_resnet50_v1_fpn        | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 747.43                           | 1.3                                   | 5.1                                  |
| 49   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                 | 70.64                            | 14.1                                   | 29.3                                 |
| 50   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G       | 626.09                           | 1.6                                   | 5.3                                  |

</details>


### Performance on U50
Measured with Vitis AI 1.1 and Vitis AI Library 1.1 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `Alveo U50` board with 6 DPUv3E kernels running at 250Mhz in Gen3x16:
  

| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 18.00                            | 166.4                                 | 394                                 |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 8.95                             | 334.6                                 | 995.1                               |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 14.13                            | 212.1                                 | 551                                 |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 17.07                            | 175.5                                 | 426.4                               |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 49.55                            | 60.5                                   | 133.3                               |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 101.98                           | 29.4                                   | 61.5                                |
| 7    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 18.01                            | 166.3                                 | 418                                 |
| 8    | ssd_pedestrain_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 88.34                            | 33.9                                   | 83.1                                |
| 9    | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G            | 104.57                           | 28.7                                   | 64.4                                |
| 10   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G        | 72.56                            | 41.3                                   | 97.6                                |
| 11   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G         | 73.04                            | 41                                     | 96.8                                |
| 12   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G             | 64.45                            | 46.5                                   | 118.1                               |
| 13   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G              | 82.49                            | 36.3                                   | 91.1                                |
| 14   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 105.83                           | 28.3                                   | 65.1                                |
| 15   | FPN                        | cf_fpn_cityscapes_256_512_8.9G               | 80.24                            | 37.3                                   | 116.9                               |
| 16   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G          | 7.39                             | 405.5                                 | 1074.5                              |
| 17   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G  | 342.87                           | 8.7                                   | 22.3                                |
| 18   | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 12.57                            | 238.4                                 | 796.2                               |
| 19   | densebox_640_360           | cf_densebox_wider_360_640_1.11G              | 29.00                            | 103.3                                 | 360.6                               |
| 20   | face_landmark              | cf_landmark_celeba_96_72_0.14G               | 1.42                             | 2107.7                                 | 6631.9                              |
| 21   | reid                       | cf_reid_market1501_160_80_0.95G              | 3.99                             | 751.4                                 | 2301                                |
| 22   | multi_task                 | cf_multitask_bdd_288_512_14.8G               | 120.52                           | 24.9                                   | 70.8                                |
| 23   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 155.79                           | 19.2                                   | 42                                  |
| 24   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 90.55                            | 33.1                                   | 75.4                                |
| 25   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 122.24                           | 24.5                                   | 54.9                                |
| 26   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                    | 58.87                            | 50.9                                   | 141.6                               |
| 27   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G            | 47.24                            | 63.4                                   | 187.5                               |
| 28   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G             | 45.20                            | 66.3                                   | 203                                 |
| 29   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G             | 42.27                            | 70.9                                   | 227.9                               |
| 30   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G | 136.51                           | 21.9                                   | 46.5                                |
| 31   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G           | 13.16                            | 227.7                                 | 682                                 |
| 32   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G       | 52.42                            | 57.2                                   | 133.3                               |
| 33   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G       | 105.11                           | 28.5                                   | 61.3                                |
| 34   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G        | 17.63                            | 169.9                                 | 460.7                               |
| 35   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G       | 28.56                            | 104.9                                 | 247                                 |
| 36   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G      | 40.49                            | 74                                     | 165.8                               |
| 37   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G             | 48.11                            | 62.3                                   | 137.9                               |
| 38   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G             | 56.95                            | 52.6                                   | 114.4                               |
| 39   | ssd_resnet50_v1_fpn        | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 525.11                           | 5.7                                   | 15.2                                |
| 40   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                 | 121.91                           | 24.6                                   | 54.7                                |


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
