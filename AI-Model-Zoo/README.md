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
|------|--------------------------|--------------------------------|-----------------------------------------------------|------------|----------------|------------|---------------|-----------------------------------------|-------------------------|--------------------------------|-------------------------------|
| 1    | Image Classification     | resnet50                       | cf\_resnet50\_imagenet\_224\_224\_7\.7G             | caffe      | resnet50       | 224\*224   | 7\.7G         | ImageNet Train                          | ImageNet Validataion    | 0\.74828/0\.92135              | 0\.7338/0\.9130               |
| 2    | Image Classifiction      | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | caffe      | resnet18       | 224\*224   | 3\.65G        | ImageNet Train                          | ImageNet Validataion    | 68\.44%/88\.64%                | 66\.94%/88\.25%               |
| 3    | Image Classification     | Inception\_v1                  | cf\_inceptionv1\_imagenet\_224\_224\_3\.16G         | caffe      | inception\_v1  | 224\*224   | 3\.16G        | ImageNet Train                          | ImageNet Validataion    | 0\.689/0\.897                  | 0\.69882/0\.894122            |
| 4    | Image Classification     | Inception\_v2                  | cf\_inceptionv2\_imagenet\_224\_224\_4G             | caffe      | bn\-inception  | 224\*224   | 4G            | ImageNet Train                          | ImageNet Validataion    | 0\.7283/0\.9109                | 0\.7170/0\.9033               |
| 5    | Image Classification     | Inception\_v3                  | cf\_inceptionv3\_imagenet\_299\_299\_11\.4G         | caffe      | inception\_v3  | 299\*299   | 11\.4G        | ImageNet Train                          | ImageNet Validataion    | 0\.77058/0\.93326              | 0\.76264/0\.930322            |
| 6    | Image Classification     | Inception\_v4                  | cf\_inceptionv4\_imagenet\_299\_299\_24\.5G         | caffe      | inception\_v3  | 299\*299   | 24\.5G        | ImageNet Train                          | ImageNet Validataion    | 0\.7959/0\.9470                | 0\.7899/0\.9445               |
| 7    | Image Classification     | mobileNet\_v2                  | cf\_mobilenetv2\_imagenet\_224\_224\_0\.59G         | caffe      | MobileNet\_v2  | 224\*224   | 608M          | ImageNet Train                          | ImageNet Validataion    | 0\.6649/0\.872362              | 0\.635219/0\.850701           |
| 8    | Image Classifiction      | SqueezeNet                     | cf\_squeeze\_imagenet\_227\_227\_0\.76G             | caffe      | squeezenet     | 227\*227   | 0\.76G        | ImageNet Train                          | ImageNet Validataion    | 54\.64%/78\.20%                | 50\.69%/77\.01%               |
| 9    | ADAS Pedstrain Detection | ssd\_pedestrain\_pruned\_0\.97 | cf\_ssdpedestrian\_coco\_360\_640\_0\.97\_5\.9G     | caffe      | VGG\-bn\-16    | 360\*640   | 5\.9G         | coco2014\_train\_person and crowndhuman | coco2014\_val\_person   | 0\.5899                        | 0\.585                        |
| 10   | Object Detection         | refinedet\_pruned\_0\.8        | cf\_refinedet\_coco\_360\_480\_0\.8\_25G            | caffe      | VGG\-bn\-16    | 360\*480   | 25G           | coco2014\_train\_person                 | coco2014\_val\_person   | 67\.68%                        | 67\.47%                       |
| 11   | Object Detection         | refinedet\_pruned\_0\.92       | cf\_refinedet\_coco\_360\_480\_0\.92\_10\.10G       | caffe      | VGG\-bn\-16    | 360\*480   | 10\.10G       | coco2014\_train\_person                 | coco2014\_val\_person   | 64\.60%                        | 64\.50%                       |
| 12   | Object Detection         | refinedet\_pruned\_0\.96       | cf\_refinedet\_coco\_360\_480\_0\.96\_5\.08G        | caffe      | VGG\-bn\-16    | 360\*480   | 5\.08G        | coco2014\_train\_person                 | coco2014\_val\_person   | 60\.89%                        | 60\.65%                       |
| 13   | ADAS Vehicle Detection   | ssd\_adas\_pruned\_0\.95       | cf\_ssdadas\_bdd\_360\_480\_0\.95\_6\.3G            | caffe      | VGG\-16        | 360\*480   | 6\.3G         | bdd100k \+ private data                 | bdd100k \+ private data | 0\.426                          | 0\.424                        |
| 14   | Traffic Detection        | ssd\_traffic\_pruned\_0\.9     | cf\_ssdtraffic\_360\_480\_0\.9\_11\.6G              | caffe      | VGG\-16        | 360\*480   | 11\.6G        | private data                            | private data            | 0\.602                          | 0\.588                        |
| 15   | ADAS Lane Detection      | VPGnet\_pruned\_0\.99          | cf\_VPGnet\_caltechlane\_480\_640\_0\.99\_2\.5G     | caffe      | VGG            | 480\*640   | 2\.5G         | caltech\-lanes\-train\-dataset          | caltech lane            | 88\.639%\(F1\-score\)          | 87%\(F1\-score\)              |
| 16   | Object Detection         | ssd\_mobilnet\_v2              | cf\_ssdmobilenetv2\_bdd\_360\_480\_6\.57G           | caffe      | MobileNet\_v2  | 360\*480   | 6\.57G        | bdd100k train                           | bdd100k val             | 0\.3186                        | 0\.3019                       |
| 17   | ADAS Segmentation        | FPN                            | cf\_fpn\_cityscapes\_256\_512\_8\.9G                | caffe      | Google\_v1\_BN | 256\*512   | 8\.9G         | Cityscapes gtFineTrain\(2975\)          | Cityscapes Val\(500\)   | 0\.5669                        | 0\.5645                       |
| 18   | Pose Estimation          | SP\-net                        | cf\_SPnet\_aichallenger\_224\_128\_0\.54G           | caffe      | Google\_v1\_BN | 128\*224   | 548\.6M       | ai\_challenger                          | ai\_challenger          | 88\.2%\(PCKh0\.5\)              | 87\.86%\(PCKh0\.5\)           |
| 19   | Pose Estimation          | Openpose\_pruned\_0\.3         | cf\_openpose\_aichallenger\_368\_368\_0\.3\_189\.7G | caffe      | VGG            | 368\*368   | 49\.88G       | ai\_challenger                          | ai\_challenger          | 0\.45067\(OKs\)                | 0\.44287\(Oks\)               |
| 20   | Face Detection           | densebox\_320\_320             | cf\_densebox\_wider\_320\_320\_0\.49G               | caffe      | VGG\-16        | 320\*320   | 0\.49G        | wider\_face                             | FDDB                    | 0\.8818                        | 0\.8768                       |
| 21   | Face Detection           | densebox\_360\_640             | cf\_densebox\_wider\_360\_640\_1\.11G               | caffe      | VGG\-16        | 360\*640   | 1\.11G        | wider\_face                             | FDDB                    | 0\.8909                        | 0\.8909                       |
| 22   | Face Recognition         | face\_landmark                 | cf\_landmark\_celeba\_96\_72\_0\.14G                | caffe      | lenet          | 96\*72     | 0\.14G        | celebA                                  | processed helen         | 0\.03704\(MAE\)                | 0\.03692\(MAE\)               |
| 23   | Re\-identification       | reid                           | cf\_reid\_marketcuhk\_160\_80\_0\.95G               | caffe      | resnet18       | 160\*80    | 0\.95G        | Market1501\+CUHK03                      | Market1501              | 78\.00%                        | 77\.60%                       |
| 24   | Detection+Segmentation   | multi-task                     | cf\_multitask\_bdd\_288\_512\_14\.8G                | caffe      | ssd            | 288\*512   | 14\.8G        | BDD100K+Cityscapes                      | BDD100K+Cityscapes      |41\.0%(Det) 50\.0%(Seg)          | 40\.0%(Det) 47\.8%(Seg)       |
| 25   | Object Detection         | yolov3\_bdd                    | cf\_yolov3\_bdd\_288\_512\_53\.7G                   | darknet    | darknet\-53    | 288\*512   | 53\.7G        | bdd100k                                 | bdd100k                 | 50\.60%                        | 49\.14%                       |
| 26   | Object Detection         | yolov3\_adas\_prune\_0\.9      | dk\_yolov3\_cityscapes\_256\_512\_0\.9\_5\.46G      | darknet    | darknet\-53    | 256\*512   | 5\.46G        | cityscape train                         | cityscape val           | 55\.20%                        | 53\.00%                       |
| 27   | Object Detection         | yolov3\_voc                    | dk\_yolov3\_voc\_416\_416\_65\.42G                  | darknet    | darknet\-53    | 416\*416   | 65\.42G       | voc07\+12\_trainval                     | voc07\_test             | 82\.4%\(MaxIntegral\)          | 81\.5%\(MaxIntegral\)         |
| 28   | Object Detection         | yolov2\_voc                    | dk\_yolov2\_voc\_448\_448\_34G                      | darknet    | darknet\-19    | 448\*448   | 34G           | voc07\+12\_trainval                     | voc07\_test             | 78\.45%\(MaxIntegral\)          | 77\.39%\(MaxIntegral\)        |
| 29   | Object Detection         | yolov2\_voc\_pruned\_0\.66     | dk\_yolov2\_voc\_448\_448\_0\.66\_11\.56G           | darknet    | darknet\-19    | 448\*448   | 11\.56G       | voc07\+12\_trainval                     | voc07\_test             | 77%\(MaxIntegral\)              | 76%\(MaxIntegral\)            |
| 30   | Object Detection         | yolov2\_voc\_pruned\_0\.71     | dk\_yolov2\_voc\_448\_448\_0\.71\_9\.86G            | darknet    | darknet\-19    | 448\*448   | 9\.86G        | voc07\+12\_trainval                     | voc07\_test             | 76\.7%\(MaxIntegral\)          | 75\.3%\(MaxIntegral\)         |
| 31   | Object Detection         | yolov2\_voc\_pruned\_0\.77     | dk\_yolov2\_voc\_448\_448\_0\.77\_7\.82G            | darknet    | darknet\-19    | 448\*448   | 7\.82G        | voc07\+12\_trainval                     | voc07\_test             | 75\.76%\(MaxIntegral\)          | 74\.6%\(MaxIntegral\)         |
| 32   | Image Classifiction      | Inception\_resnet\_v2          | tf\_inceptionresnetv2\_imagenet\_299\_299\_26\.35G  | tensorflow | inception      | 299\*299   | 26\.35G       | ImageNet Train                          | ImageNet Validataion    | 80\.37%                        | 79\.91%                       |
| 33   | Image Classifiction      | Inception\_v1                  | tf\_inceptionv1\_imagenet\_224\_224\_3G             | tensorflow | inception      | 224\*224   | 3G            | ImageNet Train                          | ImageNet Validataion    | 69\.76%                        | 67\.94%                       |
| 34   | Image Classifiction      | Inception\_v3                  | tf\_inceptionv3\_imagenet\_299\_299\_11\.45G        | tensorflow | inception      | 299\*299   | 11\.45G       | ImageNet Train                          | ImageNet Validataion    | 77\.98%                        | 76\.07%                       |
| 35   | Image Classifiction      | Inception\_v4                  | tf\_inceptionv4\_imagenet\_299\_299\_24\.55G        | tensorflow | inception      | 299\*299   | 24\.55G       | ImageNet Train                          | ImageNet Validataion    | 80\.18%                        | 79\.32%                       |
| 36   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_0\.25\_imagenet\_128\_128\_27\.15M | tensorflow | mobilenet      | 128\*128   | 27\.15M       | ImageNet Train                          | ImageNet Validataion    | 41\.44%                        | 34\.64%                       |
| 37   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_0\.5\_imagenet\_160\_160\_150\.07M | tensorflow | mobilenet      | 160\*160   | 150\.07M       | ImageNet Train                          | ImageNet Validataion    | 59\.03%                        | 51\.95%                       |
| 38   | Image Classifiction      | Mobilenet\_v1                  | tf\_mobilenetv1\_1\.0\_imagenet\_224\_224\_1\.14G   | tensorflow | mobilenet      | 224\*224   | 1\.14G       | ImageNet Train                          | ImageNet Validataion    | 71\.02%                        | 66\.10%                       |
| 39   | Image Classifiction      | Mobilenet\_v2                  | tf\_mobilenetv2\_1\.0\_imagenet\_224\_224\_0\.59G   | tensorflow | mobilenet      | 224\*224   | 0\.59G       | ImageNet Train                          | ImageNet Validataion    | 70\.13%                        | 67\.67%                       |
| 40   | Image Classifiction      | Mobilenet\_v2                  | tf\_mobilenetv2\_1\.4\_imagenet\_224\_224\_1\.16G   | tensorflow | mobilenet      | 224\*224   | 1\.16G       | ImageNet Train                          | ImageNet Validataion    | 74\.11%                        | 71\.94%                       |
| 41   | Image Classifiction      | resnet\_v1\_50                 | tf\_resnetv1\_50\_imagenet\_224\_224\_6\.97G        | tensorflow | resnetv1       | 224\*224   | 6\.97G        | ImageNet Train                          | ImageNet Validataion    | 75\.20%                        | 74\.23%                       |
| 42   | Image Classifiction      | resnet\_v1\_101                | tf\_resnetv1\_101\_imagenet\_224\_224\_14\.4G       | tensorflow | resnetv1       | 224\*224   | 14\.4G        | ImageNet Train                          | ImageNet Validataion    | 76\.40%                        | 74\.17%                       |
| 43   | Image Classifiction      | resnet\_v1\_152                | tf\_resnetv1\_152\_imagenet\_224\_224\_21\.83G      | tensorflow | resnetv1       | 224\*224   | 21\.83G       | ImageNet Train                          | ImageNet Validataion    | 76\.81%                        | 74\.69%                       |
| 44   | Image Classifiction      | vgg\_16                        | tf\_vgg16\_imagenet\_224\_224\_30\.96G              | tensorflow | vgg            | 224\*224   | 30\.96G       | ImageNet Train                          | ImageNet Validataion    | 70\.89%                        | 70\.69%                       |
| 45   | Image Classifiction      | vgg\_19                        | tf\_vgg19\_imagenet\_224\_224\_39\.28G              | tensorflow | vgg            | 224\*224   | 39\.28G       | ImageNet Train                          | ImageNet Validataion    | 71\.00%                        | 70\.26%                       |
| 46   | Object Detection         | ssd\_mobilenet\_v1             | tf\_ssdmobilenetv1\_coco\_300\_300\_2\.47G          | tensorflow | mobilenet      | 300\*300   | 2\.47G        | coco2017                                | coco2014 minival        | 20\.80%                        | 19\.60%                       |
| 47   | Object Detection         | ssd\_mobilenet\_v2             | tf\_ssdmobilenetv2\_coco\_300\_300\_3\.75G          | tensorflow | mobilenet      | 300\*300   | 3\.75G        | coco2017                                | coco2014 minival        | 21\.50%                        | 20\.30%                       |
| 48   | Object Detection         | ssd\_resnet50\_v1\_fpn         | tf\_ssdresnet50v1\_fpn\_coco\_640\_640\_178\.4G     | tensorflow | resnet50       | 300\*300   | 178\.4G       | coco2017                                | coco2014 minival        | 30\.10%                        | 29\.00%                       |
| 49   | Object Detection         | yolov3\_voc                    | tf\_yolov3\_voc\_416\_416\_65\.63G                  | tensorflow | darknet\-53    | 416\*416   | 65\.63G       | voc07\+12\_trainval                     | voc07\_test             | 78\.46%                        | 77\.38%                       |
| 50   | Object Detection         | mlperf\_ssd\_resnet34          | tf\_mlperf_resnet34\_coco\_1200\_1200\_433G         | tensorflow | resnet34       | 1200\*1200 | 433G          | coco2017                                | coco2017                | 22\.50%                        | 20\.70%                       |

</details>

### Naming Rules
Model name: `F_M_D_H_W_(P)_C`
* `F` specifies training framework: `cf` is Caffe, `tf` is Tensorflow, `dk` is Darknet, `pt` is PyTorch
* `M` specifies the model
* `D` specifies the dataset
* `H` specifies the height of input data
* `W` specifies the width of input data
* `P` specifies the pruning ratio, it means how much computation is reduced. It is optional depending on whether the model is pruned.
* `C` specifies the computation of the model: how many Gops per image


For example, `cf_refinedet_coco_480_360_0.8_25G_1.0` is a `RefineDet` model trained with `Caffe` using `COCO` dataset, input data size is `480*360`, `80%` pruned, and the computation per image is `25Gops`.


### Caffe_Xilinx 
This is a custom distribution of Caffe. Please use Caffe_Xilinx to test/finetune the caffe models listed in this page.

**Note:** To download Caffe_Xlinx, visit [Caffe_Xilinx.zip](https://www.xilinx.com/bin/public/openDownload?filename=Caffe_Xilinx_1.0.zip)



## Model Download
The following table lists various models, download link and MD5 checksum for the zip file of each model.

**Note:** To download all the models, visit [all_models_1.0.zip](https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.0.zip).

<details>
 <summary><b>Click here to view details</b></summary>

If you are a:
 - Linux user, use the [`get_model.sh`](reference-files/get_model.sh) script to download all the models.   
 - Windows user, use the download link listed in the following table to download a model.


| No\. | Model                                        | Size      | Download link | Checksum |
| ---- | -------------------------------------------- | --------- | ------------- | -------- |
| 1    | cf_resnet50_imagenet_224_224_7.7G            |226.62 MB  |https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet50_imagenet_224_224_1.0.zip               |26a8881c800f6e27888a167947c33559          |
| 2    | cf_inceptionv1_imagenet_224_224_3.16G        |86.47 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv1_imagenet_224_224_1.0.zip               |b3e6f9d61fe25ae4425c8efa24138625          |
| 3    | cf_inceptionv2_imagenet_224_224_4G           |143.38 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv2_imagenet_224_224_1.0.zip               |c8db5d52d6b5fd061c17b5ef116c3f54          |
| 4    | cf_inceptionv3_imagenet_299_299_11.4G        |212.43 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv3_imagenet_299_299_1.0.zip               |ebe9184731d13ce35c567c5f4a200f32          |
| 5    | cf_inceptionv4_imagenet_299_299_24.5G        |380.38 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_inceptionv4_imagenet_299_299_1.0.zip               |cca381dfe5c84e43195aadabe2899622          |
| 6    | cf_mobilenetv2_imagenet_224_224_0.59G        |23.27 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_mobilenetv2_imagenet_224_224_1.0.zip               |fc7de15fbcff8d318327716a7f04b7bd          |
| 7    | cf_squeeze_imagenet_227_227_0.76G            |11.27 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_squeeze_imagenet_227_227_1.0.zip               |efeed69bb60e4807d08a9ed4dee42731          |
| 8    | cf_resnet18_imagenet_224_224_3.65G           |175.28MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet18_imagenet_224_224_1.0.zip               |cc6e2a7d48ddc9c1a68b5d2839fa2b84          |
| 9    | cf_ssdpedestrian_coco_360_640_0.97_5.9G      |7.78 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdpedestrian_coco_360_640_0.97_1.0.zip               |46b992db8718d98dbf212d203b0f1ec6          |
| 10   | cf_refinedet_coco_360_480_0.8_25G            |37.92 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_0.8_1.0.zip               |1bf37b830552b1cc7fbf671414889074          |
| 11   | cf_refinedet_coco_360_480_0.92_10.10G        |10.66 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_0.92_1.0.zip               |dde4c33563eafefbe499bcb4b4cd6d1a          |
| 12   | cf_refinedet_coco_360_480_0.96_5.08G         |5.53 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_refinedet_coco_480_360_0.96_1.0.zip              |0db83cf6ce87325fc34813f1c14ac6df          |
| 13   | cf_ssdadas_bdd_360_480_0.95_6.3G             |11.34 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdadas_bdd_360_480_0.95_1.0.zip               |5becc3e0853612277350d295687fd94e          |
| 14   | cf_ssdtraffic_360_480_0.9_11.6G              |20.13 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdtraffic_360_480_0.9_1.0.zip               |a9b1b10f2f493a34074b70f70ee2dd84          |
| 15   | cf_VPGnet_caltechlane_480_640_0.99_2.5G      |10.39 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_VPGnet_caltechlane_480_640_0.99_1.0.zip               |b4e1091016917b2d0ccaf3e51ecfab3f          |
| 16   | cf_ssdmobilenetv2_bdd_360_480_6.57G          |100.77 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_ssdmobilenetv2_bdd_360_480_1.0.zip               |171746d4c1a2d97408ff8eb9c08a7b6a          |
| 17   | cf_fpn_cityscapes_256_512_8.9G               |58.17MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_fpn_cityscapes_256_512_1.0.zip               | dbae0fba17aaf3c6242d511032efb0fd         |
| 18   | cf_SPnet_aichallenger_224_128_0.54G          |12.06 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_SPnet_aichallenger_224_128_1.0.zip               |a4ef58d3eaec7ff284af2c22f0178d2b          |
| 19   | cf_openpose_aichallenger_368_368_0.3_189.7G  |544.23 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_openpose_aichallenger_368_368_0.3_1.0.zip               |b62adb84d7df0aa976f5485bbec6a375          |
| 20   | cf_densebox_wider_320_320_0.49G              |6.26 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_densebox_wider_320_320_1.0.zip               |15f2e1c780dc8ba72d01491b773c10be          |
| 21   | cf_densebox_wider_360_640_1.11G              |6.26 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_densebox_wider_360_640_1.0.zip               |fd9f136fe664cc4f56b3e0133efcfc49          |
| 22   | cf_landmark_celeba_96_72_0.14G               |50.47 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_landmark_celeba_96_72_1.0.zip               |200993da21ada189110a34ba2f4b65ca          |
| 23   | cf_reid_marketcuhk_160_80_0.95G              |98.36 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_reid_marketcuhk_160_80_1.0.zip               |092c2e42674af381b8a19564077b3c85          |
| 24   | cf_multitask_bdd_288_512_14.8G               |122.37 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_multitask_bdd_288_512_1.0.zip               |d7b1f54bf6a5ecbc91651b50c63bd1cb          |
| 25   | cf_yolov3_bdd_288_512_53.7G                  |948 MB      |https://www.xilinx.com/bin/public/openDownload?filename=cf_yolov3_bdd_288_512_1.0.zip               |83661dba91ac4acf5ddb6db6ee7413c5          |
| 26   | dk_yolov3_cityscapes_256_512_0.9_5.46G       |38.08 MB      |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_cityscapes_256_512_0.9_1.0.zip               |be571f096cf2c52e56293f5a68837a50          |
| 27   | dk_yolov3_voc_416_416_65.42G                 |940.24 MB      |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov3_voc_416_416_1.0.zip               |fc7f103d657a39b9efbe2d675c3de70e          |
| 28   | dk_yolov2_voc_448_448_34G                    |476.55 MB      |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_1.0.zip               |a02a009aed9f36185c5901604ad49c76          |
| 29   | dk_yolov2_voc_448_448_0.66_11.56G            |223.44 MB      |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.66_1.0.zip               |28f7ea8f29c73cc6507c79d86968c2cb          |
| 30   | dk_yolov2_voc_448_448_0.71_9.86G             |202.46 MB      |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.71_1.0.zip               |626971b06f893b24f4a4750fe150101f          |
| 31   | dk_yolov2_voc_448_448_0.77_7.82G             |146.72 MB      |https://www.xilinx.com/bin/public/openDownload?filename=dk_yolov2_voc_448_448_0.77_1.0.zip               |4cb61f9312dc91f7150e599a133059ba          |
| 32   | tf_inceptionresnetv2_imagenet_299_299_26.35G |657.27 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_inception_resnet_v2_imagenet_299_299_1.0.zip               |bf515feaf817b156420c7043aa7ee744          |
| 33   | tf_inceptionv1_imagenet_224_224_3G           |76.95 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv1_imagenet_224_224_1.0.zip               |7df195f8045c5d6d44e56c03c675f8fe          |
| 34   | tf_inceptionv3_imagenet_299_299_11.45G       |287.42 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv3_imagenet_299_299_1.0.zip               |88c5d39491e143e7b10de7718e1e94f1          |
| 35   | tf_inceptionv4_imagenet_299_299_24.55G       |505.64 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_inceptionv4_imagenet_299_299_1.0.zip               |03f69653b71145fa893c66c0fffcf257          |
| 36   | tf_mobilenetv1_0.25_imagenet_128_128_27.15M  |10.74 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_0.25_imagenet_128_128_1.0.zip               |f698cddfe7334c13dd1268d3e1d59b11          |
| 37   | tf_mobilenetv1_0.5_imagenet_160_160_150.07M  |29.72 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_0.5_imagenet_160_160_1.0.zip               |d9cafa9cf361e99e7aabf836982dee3f          |
| 38   | tf_mobilenetv1_1.0_imagenet_224_224_1.14G    |93.97 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv1_1.0_imagenet_224_224_1.0.zip               |7747fad0fd70d7fd5e6688abeeb52817          |
| 39   | tf_mobilenetv2_1.0_imagenet_224_224_0.59G    |78.33 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv2_1.0_imagenet_224_224_1.0.zip               |4fb81d606c2b78fb34e1cc06acc58c00          |
| 40   | tf_mobilenetv2_1.4_imagenet_224_224_1.16G    |135.64 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_mobilenetv2_1.4_imagenet_224_224_1.0.zip               |f31c3cf368c0c04762d3a251b173ed43          |
| 41   | tf_resnetv1_50_imagenet_224_224_6.97G        |295.19 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_50_imagenet_224_224_1.0.zip               |fb6c2b68f6f5dd356100d6f630d21c35          |
| 42   | tf_resnetv1_101_imagenet_224_224_14.4G       |514.12 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_101_imagenet_224_224_1.0.zip               |4d72ed81fbf5ac01de244083f6fcadee          |
| 43   | tf_resnetv1_152_imagenet_224_224_21.83G      |695.86 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1_152_imagenet_224_224_1.0.zip               |5c9321bc0e469f4d4f21cd075e2fadae          |
| 44   | tf_vgg16_imagenet_224_224_30.96G             |1.57 GB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_vgg16_imagenet_224_224_1.0.zip               |b5c5ed1e8bc6d50821e6802c6702da7f          |
| 45   | tf_vgg19_imagenet_224_224_39.28G             |1.63 GB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_vgg19_imagenet_224_224_1.0.zip               |5ca310d0410eb266f4fccf49dd378e23          |
| 46   | tf_ssdmobilenetv1_coco_300_300_2.47G         |135.78 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdmobilenetv1_coco_300_300_1.0.zip               |6c013ef52898b68699c1b3bc5ddc5909          |
| 47   | tf_ssdmobilenetv2_coco_300_300_3.75G         |318.32 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdmobilenetv2_coco_300_300_1.0.zip               |53ace1f075ad0b01b53cca5f6884e0df          |
| 48   | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     |732.25 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_ssdresnet50v1_fpn_coco_640_640_1.0.zip            |fb5f1fbd4dbee9d4e19d4a382ddd893f          |
| 49   | tf_yolov3_voc_416_416_65.63G                 |500.31 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_yolov3_voc_416_416_1.0.zip               |9f134db4acff5f028d822f15ee5da189          |
| 50   | tf_mlperf_resnet34_coco_1200_1200_433G       |508.07 MB      |https://www.xilinx.com/bin/public/openDownload?filename=tf_mlperf_resnet34_coco_1200_1200_1.0.zip               |b6a43644b9ff8d59c7c76201a62d8970          |
| -    | all models                                   |13.87 GB      |https://www.xilinx.com/bin/public/openDownload?filename=all_models_1.0.zip               |ed5509bcd0ce5e3aa2b220145acc17f5          |


</details>

### Model Directory Structure
Download and extract the model archive to your working area on the local hard disk. For details on the various models, their download link and MD5 checksum for the zip file of each model, see [Model Download](#model-download).

#### Caffe Model Directory Structure
For a Caffe model, you should see the following directory structure:

    ├── code                            # Contains code and instructions.
    │   ├── test                        # Contains test code which can run demo and evaluate model performance.
    │   ├── train                       # Contains train code and data preprocess code.
    │   └── readme.md                   # Contains environment requirements and train eval instructions.
    │                                     
    │                                   
    ├── readme.md                       # Contains the environment requirement and data preprocess information.
    │                                     Refer this file to know more about creating `float.prototxt` by adding
    │                                     datalayer to `test.prototxt` in the `float` directory.
    ├── compiler                          
    │   ├── deploy.caffemodel           # Input to the compiler. The same with deploy.caffemodel in the `quantized` directory.
    │   └── deploy.prototxt             # Input to the compiler. The modified prototxt based on deploy.prototxt
    │                                     in the `quantized` directory, which removes unnecessary or unsupported layers
    │                                     for compilation.
    ├── quantized                             
    │   ├── deploy.caffemodel           # Quantized weights, the output of vai_q_caffe without modification.
    │   ├── deploy.prototxt             # Quantized prototxt, the output of vai_q_caffe without modification.
    │   ├── quantized_test.prototxt           # Used to run evaluation with quantized_train_test.caffemodel on GPU
    │   │                                 using python test code released in near future. Some models
    │   │                                 don't have this file if they are converted from Darknet (Yolov2, Yolov3),
    │   │                                 Pytorch (ReID) or there is no Caffe Test (Densebox).
    │   ├── quantized_train_test.caffemodel   # Quantized weights can be used for quantizeded-point training and evaluation.    
    │   └── quantized_train_test.prototxt     # Used for quantized-point training and testing with quantized_train_test.caffemodel
    │                                     on GPU when datalayer modified to user's data path.
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


    ├── test_code                       # Contains code and instructions.
    │   ├── float                       # Test code and instruction for floating model for evaluation.
    │   └── quantized                         # Test code and instruction for quantized model for evaluation.
    │
    ├── readme.md                       # Contains the environment requirement, the input and output nodes as well as
    │                                     the data preprocess and postprocess information.
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


**Note:** The model performance number listed in the following sections is generated with Vitis AI v1.0 and Vitis AI Lirary v1.0. For each board, a different DPU configuration is used. Vitis AI and Vitis AI Library can be downloaded for free from [Vitis AI Github](https://github.com/Xilinx/Vitis-AI) and [Vitis AI Library Github](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Library).
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
| 34   | yolov3\_bdd                    | cf\_yolov3\_bdd\_288\_512\_53\.7G                   | 73\.89                          | 13\.5333                                | 42\.8833                               |
| 35   | tf\_mobilenet\_v1              | tf\_mobilenetv1\_imagenet\_224\_224\_1\.14G         | 3\.2                            | 312\.067                                | 875\.967                               |
| 36   | resnet18                       | cf\_resnet18\_imagenet\_224\_224\_3\.65G            | 5\.1                            | 195\.95                                 | 524\.433                               |
| 37   | resnet18\_wide                 | tf\_resnet18\_imagenet\_224\_224\_28G               | 33\.28                          | 30\.05                                  | 83\.4167                               |
</details>


### Performance on ZCU102 (0432055-05)
Measured with Vitis AI 1.0, Vitis AI Library 1.0 and Vitis DPU 1.0 in Nov 2019  

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU102 (0432055-05)` board with a `3 * B4096  @ 287MHz   V1.4.0` DPU configuration:


| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 14.06                            | 71.1                                  | 150.2                                |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 5.84                             | 171.1                                | 437.6                                |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 6.17                             | 162.2                                | 422.4                                |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 7.52                             | 133                                  | 321.4                                |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 18.25                            | 54.8                                  | 131.2                                |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 36.10                            | 27.7                                  | 67.4                                 |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G        | 4.76                             | 210.1                                | 557.4                                |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 3.78                             | 264.5                                | 1121.6                               |
| 9    | ssd_pedestrain_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 13.16                            | 76                                    | 306.1                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G            | 31.65                            | 31.6                                  | 106                                 |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G        | 16.78                            | 59.6                                  | 206.2                                |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G         | 12.15                            | 82.3                                  | 292.6                                |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G             | 12.11                            | 82.6                                  | 299                                |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G              | 18.05                            | 55.4                                  | 214.2                                |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 9.82                             | 101.8                                | 401.1                                |
| 16   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G          | 25.84                            | 38.7                                  | 117.8                                |
| 17   | FPN                        | cf_fpn_cityscapes_256_512_8.9G               | 17.06                            | 58.6                                  | 186.7                                |
| 18   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G          | 1.95                             | 511.6                                | 1386.4                               |
| 19   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G  | 285.71                           | 3.5                                  | 15.6                                   |
| 20   | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 2.61                             | 383                                  | 1363.7                               |
| 21   | densebox_640_360           | cf_densebox_wider_360_640_1.11G              | 5.24                             | 190.7                                | 637.8                                |
| 22   | face_landmark              | cf_landmark_celeba_96_72_0.14G               | 1.28                             | 779.6                                | 1348                                 |
| 23   | reid                       | cf_reid_marketcuhk_160_80_0.95G              | 2.91                             | 343.3                                | 659.4                                |
| 24   | multi_task                 | cf_multitask_bdd_288_512_14.8G               | 28.17                            | 35.5                                  | 133.2                                |
| 25   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 77.52                            | 12.9                                  | 37.5                                 |
| 26   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 12.20                            | 82                                    | 227.3                                |
| 27   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 74.07                            | 13.5                                  | 38.2                                 |
| 28   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                    | 40.49                            | 24.7                                  | 76.2                                 |
| 29   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G            | 18.83                            | 53.1                                  | 203.7                                |
| 30   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G             | 16.75                            | 59.7                                  | 235.9                                |
| 31   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G             | 14.75                            | 67.8                                  | 281.6                                |
| 32   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G | 43.67                            | 22.9                                  | 49.1                                 |
| 33   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G           | 5.98                             | 167.2                                | 434.8                                |
| 34   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G       | 18.28                            | 54.7                                  | 129.7                                |
| 35   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G       | 36.10                            | 27.7                                  | 67.5                                 |
| 36   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M  | 1.20                             | 836.1                                | 2270.7                               |
| 37   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M  | 1.76                             | 566.7                                | 1816.9                               |
| 38   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G    | 3.90                             | 256.1                                | 763.7                                |
| 39   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G    | 4.68                             | 213.6                                | 575.2                                |
| 40   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G    | 6.30                             | 158.7                                | 395.4                                |
| 41   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G        | 13.09                            | 76.4                                  | 159.4                                |
| 42   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G       | 23.53                            | 42.5                                  | 90.7                                 |
| 43   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G      | 34.13                            | 29.3                                  | 63.3                                 |
| 44   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G             | 52.63                            | 19                                    | 41.8                                 |
| 45   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G             | 60.61                            | 16.5                                  | 37.8                                 |
| 46   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G         | 11.11                            | 90                                  | 320.6                                |
| 47   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G         | 16.18                            | 61.8                                  | 196.6                                |
| 48   | ssd_resnet_50_v1_fpn        | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 769.23                           | 1.3                                  | 5.9                                  |
| 49   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                 | 74.63                            | 13.4                                  | 37.8                                 |
| 50   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G       | 526.32                           | 1.9                                  | 7.7                                    |

</details>


### Performance on ZCU104
Measured with Vitis AI 1.0, Vitis AI Library 1.0 and Vitis DPU 1.0 in Nov 2019 

<details>
 <summary><b>Click here to view details</b></summary>

The following table lists the performance number including end-to-end throughput and latency for each model on the `ZCU104` board with a `2 * B4096  @ 305MHz   V1.4.0` DPU configuration:


| No\. | Model                      | Name                                         | E2E latency \(ms\) Thread num =1 | E2E throughput \-fps\(Single Thread\) | E2E throughput \-fps\(Multi Thread\) |
| ---- | :------------------------- | :------------------------------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ |
| 1    | resnet50                   | cf_resnet50_imagenet_224_224_7.7G            | 12.64                            | 79.1                                   | 142.7                                |
| 2    | resnet18                   | cf_resnet18_imagenet_224_224_3.65G           | 5.18                             | 193                                   | 382.1                                |
| 3    | Inception_v1               | cf_inceptionv1_imagenet_224_224_3.16G        | 5.41                             | 184.7                                 | 371.4                                |
| 4    | Inception_v2               | cf_inceptionv2_imagenet_224_224_4G           | 6.68                             | 149.7                                 | 285                                  |
| 5    | Inception_v3               | cf_inceptionv3_imagenet_299_299_11.4G        | 16.81                            | 59.5                                   | 113.6                                |
| 6    | Inception_v4               | cf_inceptionv4_imagenet_299_299_24.5G        | 33.44                            | 29.9                                   | 57.6                                 |
| 7    | Mobilenet_v2               | cf_mobilenetv2_imagenet_224_224_0.59G        | 4.10                             | 244.2                                 | 510.5                                |
| 8    | SqueezeNet                 | cf_squeeze_imagenet_227_227_0.76G            | 3.70                             | 270.6                                 | 1060.4                               |
| 9    | ssd_pedestrain_pruned_0_97 | cf_ssdpedestrian_coco_360_640_0.97_5.9G      | 12.80                            | 78.1                                   | 192.8                                |
| 10   | refinedet_pruned_0_8       | cf_refinedet_coco_360_480_0.8_25G            | 30.86                            | 32.4                                   | 75                                   |
| 11   | refinedet_pruned_0_92      | cf_refinedet_coco_360_480_0.92_10.10G        | 16.42                            | 60.9                                   | 137.8                                |
| 12   | refinedet_pruned_0_96      | cf_refinedet_coco_360_480_0.96_5.08G         | 12.03                            | 83.1                                   | 193.2                                |
| 13   | ssd_adas_pruned_0_95       | cf_ssdadas_bdd_360_480_0.95_6.3G             | 11.83                            | 84.5                                   | 197.5                                |
| 14   | ssd_traffic_pruned_0_9     | cf_ssdtraffic_360_480_0.9_11.6G              | 17.48                            | 57.2                                   | 133.1                                |
| 15   | VPGnet_pruned_0_99         | cf_VPGnet_caltechlane_480_640_0.99_2.5G      | 9.53                             | 104.9                                 | 351.3                                |
| 16   | ssd_mobilenet_v2           | cf_ssdmobilenetv2_bdd_360_480_6.57G          | 39.53                            | 25.3                                   | 108.4                                |
| 17   | FPN                        | cf_fpn_cityscapes_256_512_8.9G               | 16.39                            | 61                                     | 162.7                                |
| 18   | SP_net                     | cf_SPnet_aichallenger_224_128_0.54G          | 1.87                             | 534.9                                 | 1147.4                               |
| 19   | Openpose_pruned_0_3        | cf_openpose_aichallenger_368_368_0.3_189.7G  | 270.27                           | 3.7                                   | 11.1                                 |
| 20   | densebox_320_320           | cf_densebox_wider_320_320_0.49G              | 2.57                             | 389.5                                 | 1342.9                               |
| 21   | densebox_640_360           | cf_densebox_wider_360_640_1.11G              | 5.08                             | 196.7                                 | 661.5                                |
| 22   | face_landmark              | cf_landmark_celeba_96_72_0.14G               | 1.19                             | 837.2                                 | 1171.7                               |
| 23   | reid                       | cf_reid_marketcuhk_160_80_0.95G              | 2.74                             | 365.3                                 | 619.2                                |
| 24   | multi_task                 | cf_multitask_bdd_288_512_14.8G               | 27.78                            | 36                                     | 107.3                                |
| 25   | yolov3_bdd                 | dk_yolov3_bdd_288_512_53.7G                  | 74.07                            | 13.5                                   | 28.7                                 |
| 26   | yolov3_adas_pruned_0_9     | dk_yolov3_cityscapes_256_512_0.9_5.46G       | 12.02                            | 83.2                                   | 208.8                                |
| 27   | yolov3_voc                 | dk_yolov3_voc_416_416_65.42G                 | 70.42                            | 14.2                                   | 29.6                                 |
| 28   | yolov2_voc                 | dk_yolov2_voc_448_448_34G                    | 38.31                            | 26.1                                   | 58.5                                 |
| 29   | yolov2_voc_pruned_0_66     | dk_yolov2_voc_448_448_0.66_11.56G            | 18.05                            | 55.4                                   | 144.2                                |
| 30   | yolov2_voc_pruned_0_71     | dk_yolov2_voc_448_448_0.71_9.86G             | 16.05                            | 62.3                                   | 169.3                                |
| 31   | yolov2_voc_pruned_0_77     | dk_yolov2_voc_448_448_0.77_7.82G             | 14.20                            | 70.4                                   | 208.7                                |
| 32   | Inception_resnet_v2        | tf_inceptionresnetv2_imagenet_299_299_26.35G | 39.84                            | 25.1                                   | 45.4                                 |
| 33   | Inception_v1               | tf_inceptionv1_imagenet_224_224_3G           | 5.19                             | 192.5                                 | 383.8                                |
| 34   | Inception_v3               | tf_inceptionv3_imagenet_299_299_11.45G       | 16.86                            | 59.3                                   | 112.7                                |
| 35   | Inception_v4               | tf_inceptionv4_imagenet_299_299_24.55G       | 33.44                            | 29.9                                   | 57.7                                 |
| 36   | Mobilenet_v1               | tf_mobilenetv1_0.25_imagenet_128_128_27.15M  | 0.81                             | 1233                                   | 3863.9                               |
| 37   | Mobilenet_v1               | tf_mobilenetv1_0.5_imagenet_160_160_150.07M  | 1.35                             | 739.9                                 | 1929.3                               |
| 38   | Mobilenet_v1               | tf_mobilenetv1_1.0_imagenet_224_224_1.14G    | 3.29                             | 304.4                                 | 672.3                                |
| 39   | Mobilenet_v2               | tf_mobilenetv2_1.0_imagenet_224_224_0.59G    | 4.08                             | 245.3                                 | 519.3                                |
| 40   | Mobilenet_v2               | tf_mobilenetv2_1.4_imagenet_224_224_1.16G    | 5.53                             | 180.8                                 | 369.1                                |
| 41   | resnet_v1_50               | tf_resnetv1_50_imagenet_224_224_6.97G        | 11.78                            | 84.9                                   | 152.2                                |
| 42   | resnet_v1_101              | tf_resnetv1_101_imagenet_224_224_14.4G       | 21.37                            | 46.8                                   | 85.6                                 |
| 43   | resnet_v1_152              | tf_resnetv1_152_imagenet_224_224_21.83G      | 31.06                            | 32.2                                   | 59.2                                 |
| 44   | vgg_16                     | tf_vgg16_imagenet_224_224_30.96G             | 48.08                            | 20.8                                   | 37.1                                 |
| 45   | vgg_19                     | tf_vgg19_imagenet_224_224_39.28G             | 55.25                            | 18.1                                   | 33                                   |
| 46   | ssd_mobilenet_v1           | tf_ssdmobilenetv1_coco_300_300_2.47G         | 10.78                            | 92.8                                   | 315.8                                |
| 47   | ssd_mobilenet_v2           | tf_ssdmobilenetv2_coco_300_300_3.75G         | 15.31                            | 65.3                                   | 177.6                                |
| 48   | ssd_resnet50_v1_fpn        | tf_ssdresnet50v1_fpn_coco_640_640_178.4G     | 714.29                           | 1.4                                   | 6.1                                  |
| 49   | yolov3_voc                 | tf_yolov3_voc_416_416_65.63G                 | 70.92                            | 14.1                                   | 29.3                                 |
| 50   | mlperf_ssd_resnet34        | tf_mlperf_resnet34_coco_1200_1200_433G       | 526.32                           | 1.9                                   | 5.6                                  |

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
| 33   | reid                           | cf\_reid\_marketcuhk\_160\_80\_0\.95G               | 6\.28                           | 159\.15                                 | 166\.633                               |
| 34   | yolov3\_bdd                    | cf\_yolov3\_bdd\_288\_512\_53\.7G                   | 193\.55                         | 5\.16667                                | 5\.31667                               |
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
