<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


# Vitis AI Model Zoo
This repository provides access to optimized deep learning models that can be used to speed up the deployment of deep learning inference on Xilinx&trade; targets. These models cover different applications, including but not limited to ADAS/AD, medical, video surveillance, robotics, data center, etc. You can get started with these free pre-trained models to better understand the benefits of Xilinx accelerated inference.

<p align="left">
  <img width="1264" height="420" src="images/vitis_ai_model_zoo.png">
</p>

As of the 3.0 release of Vitis AI, all Model Zoo documentation has migrated to Github.IO.  You may access the Model Zoo documentation online [here](https://xilinx.github.io/Vitis-AI/3.0/html/docs/workflow-model-zoo) or offline [here](../docs/docs/workflow-model-zoo.html).

Please note that if the models are marked as limited to non-commercial use, then users need compliance with the AMD license agreement for non-commercial models that showed in model.yaml of [model-list](/model_zoo/model-list). 

| #    | Model                        | Platform   | Datatype FP32 | Datatype INT8 | Pruned | Reminder for limited use scope |
| ---- | :--------------------------- | :--------- | :-----------: | :-----------: | :----: | ------------------------------ |
| 1    | inception-resnetv2           | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 2    | inceptionv1                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 3    | inceptionv1 pruned0.09       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 4    | inceptionv1 pruned0.16       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 5    | inceptionv2                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 6    | inceptionv3                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 7    | inceptionv3 pruned0.2        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 8    | inceptionv3 pruned0.4        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 9    | inceptionv4                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 10   | inceptionv4 pruned0.2        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 11   | inceptionv4 pruned0.4        | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 12   | mobilenetv1_0.25             | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 13   | mobilenetv1_0.5              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 14   | mobilenetv1_1.0              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 15   | mobilenetv1_1.0 pruned0.11   | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 16   | mobilenetv1_1.0 pruned0.12   | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 17   | mobilenetv2_1.0              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 18   | mobilenetv2_1.4              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 19   | resnetv1_50                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 20   | resnetv1_50 pruned0.38       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 21   | resnetv1_50 pruned0.65       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 22   | resnetv1_101                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 23   | resnetv1_101 pruned0.35      | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 24   | resnetv1_101 pruned0.57      | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 25   | resnetv1_152                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 26   | resnetv1_152 pruned0.51      | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 27   | resnetv1_152pruned0.60       | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 28   | vgg16                        | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 29   | vgg16 pruned0.43             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 30   | vgg16 pruned0.50             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 31   | vgg19                        | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 32   | vgg19 pruned0.24             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 33   | vgg19 pruned0.39             | TensorFlow |       √       |       √       |   √    | Non-Commercial Use Only        |
| 34   | resnetv2_50                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 35   | resnetv2_101                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 36   | resnetv2_152                 | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 37   | efficientnet-edgetpu-S       | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 38   | efficientnet-edgetpu-M       | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 39   | efficientnet-edgetpu-L       | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 40   | mlperf_resnet50              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 41   | mobilenetEdge1.0             | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 42   | mobilenetEdge0.75            | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 43   | resnet50                     | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 44   | mobilenetv1                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 45   | inceptionv3                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 46   | efficientnet-b0              | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 47   | mobilenetv3                  | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 48   | efficientnet-lite            | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 49   | ViT                          | TensorFlow |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 50   | ssdmobilenetv1               | TensorFlow |       √       |       √       |   ×    |                                |
| 51   | ssdmobilenetv2               | TensorFlow |       √       |       √       |   ×    |                                |
| 52   | ssdresnet50v1_fpn            | TensorFlow |       √       |       √       |   ×    |                                |
| 53   | yolov3                       | TensorFlow |       √       |       √       |   ×    |                                |
| 54   | mlperf_resnet34              | TensorFlow |       √       |       √       |   ×    |                                |
| 55   | ssdlite_mobilenetv2          | TensorFlow |       √       |       √       |   ×    |                                |
| 56   | ssdinceptionv2               | TensorFlow |       √       |       √       |   ×    |                                |
| 57   | refinedet                    | TensorFlow |       √       |       √       |   ×    |                                |
| 58   | efficientdet-d2              | TensorFlow |       √       |       √       |   ×    |                                |
| 59   | yolov3                       | TensorFlow |       √       |       √       |   ×    |                                |
| 60   | yolov4_416                   | TensorFlow |       √       |       √       |   ×    |                                |
| 61   | yolov4_512                   | TensorFlow |       √       |       √       |   ×    |                                |
| 62   | RefineDet-Medical            | TensorFlow |       √       |       √       |   ×    |                                |
| 63   | RefineDet-Medical pruned0.50 | TensorFlow |       √       |       √       |   √    |                                |
| 64   | RefineDet-Medical pruned0.75 | TensorFlow |       √       |       √       |   √    |                                |
| 65   | RefineDet-Medical pruned0.85 | TensorFlow |       √       |       √       |   √    |                                |
| 66   | RefineDet-Medical pruned0.88 | TensorFlow |       √       |       √       |   √    |                                |
| 67   | mobilenetv2 (segmentation)   | TensorFlow |       √       |       √       |   ×    |                                |
| 68   | erfnet                       | TensorFlow |       √       |       √       |   ×    |                                |
| 69   | 2d-unet                      | TensorFlow |       √       |       √       |   ×    |                                |
| 70   | bert-base                    | TensorFlow |       √       |       √       |   ×    |                                |
| 71   | superpoint                   | TensorFlow |       √       |       √       |   ×    |                                |
| 72   | HFNet                        | TensorFlow |       √       |       √       |   ×    |                                |
| 73   | rcan                         | TensorFlow |       √       |       √       |   ×    |                                |
| 74   | inceptionv3                  | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 75   | inceptionv3 pruned0.3        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 76   | inceptionv3 pruned0.4        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 77   | inceptionv3 pruned0.5        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 78   | inceptionv3 pruned0.6        | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 79   | squeezenet                   | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 80   | resnet50_v1.5                | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 81   | resnet50_v1.5 pruned0.3      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 82   | resnet50_v1.5 pruned0.4      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 83   | resnet50_v1.5 pruned0.5      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 84   | resnet50_v1.5 pruned0.6      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 85   | resnet50_v1.5 pruned0.7      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 86   | OFA-resnet50                 | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 87   | OFA-resnet50 pruned0.45      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 88   | OFA-resnet50 pruned0.60      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 89   | OFA-resnet50 pruned0.74      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 90   | OFA-resnet50 pruned0.88      | PyTorch    |       √       |       √       |   √    | Non-Commercial Use Only        |
| 91   | OFA-depthwise-resnet50       | PyTorch    |       √       |       √       |   ×    | Non-Commercial Use Only        |
| 92   | vehicle type classification  | PyTorch    |       √       |       √       |   ×    |                                |
| 93   | vehicle make classification  | PyTorch    |       √       |       √       |   ×    |                                |
| 94   | vehicle color classification | PyTorch    |       √       |       √       |   ×    |                                |
| 95   | OFA-yolo                     | PyTorch    |       √       |       √       |   ×    |                                |
| 96   | OFA-yolo pruned0.3           | PyTorch    |       √       |       √       |   √    |                                |
| 97   | OFA-yolo pruned0.6           | PyTorch    |       √       |       √       |   √    |                                |
| 98   | yolox-nano                   | PyTorch    |       √       |       √       |   ×    |                                |
| 99   | yolov4csp                    | PyTorch    |       √       |       √       |   ×    |                                |
| 100  | yolov5-large                 | PyTorch    |       √       |       √       |   ×    |                                |
| 101  | yolov5-nano                  | PyTorch    |       √       |       √       |   ×    |                                |
| 102  | yolov5s6                     | PyTorch    |       √       |       √       |   ×    |                                |
| 103  | yolov6m                      | PyTorch    |       √       |       √       |   ×    |                                |
| 104  | pointpillars                 | PyTorch    |       √       |       √       |   ×    |                                |
| 105  | CLOCs                        | PyTorch    |       √       |       √       |   ×    |                                |
| 106  | Enet                         | PyTorch    |       √       |       √       |   ×    |                                |
| 107  | SemanticFPN-resnet18         | PyTorch    |       √       |       √       |   ×    |                                |
| 108  | SemanticFPN-mobilenetv2      | PyTorch    |       √       |       √       |   ×    |                                |
| 109  | salsanext pruned0.60         | PyTorch    |       √       |       √       |   √    |                                |
| 110  | salsanextv2 pruned0.75       | PyTorch    |       √       |       √       |   √    |                                |
| 111  | SOLO                         | PyTorch    |       √       |       √       |   ×    |                                |
| 112  | HRNet                        | PyTorch    |       √       |       √       |   ×    |                                |
| 113  | CFLOW                        | PyTorch    |       √       |       √       |   ×    |                                |
| 114  | 3D-UNET                      | PyTorch    |       √       |       √       |   ×    |                                |
| 115  | MaskRCNN                     | PyTorch    |       √       |       √       |   ×    |                                |
| 116  | bert-base                    | PyTorch    |       √       |       √       |   ×    |                                |
| 117  | bert-large                   | PyTorch    |       √       |       √       |   ×    |                                |
| 118  | bert-tiny                    | PyTorch    |       √       |       √       |   ×    |                                |
| 119  | face-mask-detection          | PyTorch    |       √       |       √       |   ×    |                                |
| 120  | movenet                      | PyTorch    |       √       |       √       |   ×    |                                |
| 121  | fadnet                       | PyTorch    |       √       |       √       |   ×    |                                |
| 122  | fadnet pruned0.65            | PyTorch    |       √       |       √       |   √    |                                |
| 123  | fadnetv2                     | PyTorch    |       √       |       √       |   ×    |                                |
| 124  | fadnetv2 pruned0.51          | PyTorch    |       √       |       √       |   √    |                                |
| 125  | psmnet pruned0.68            | PyTorch    |       √       |       √       |   √    |                                |
| 126  | pmg                          | PyTorch    |       √       |       √       |   ×    |                                |
| 127  | SESR-S                       | PyTorch    |       √       |       √       |   ×    |                                |
| 128  | OFA-rcan                     | PyTorch    |       √       |       √       |   ×    |                                |
| 129  | DRUNet                       | PyTorch    |       √       |       √       |   ×    |                                |
| 130  | xilinxSR                     | PyTorch    |       √       |       √       |   ×    |                                |

## Contributing

We welcome community contributions. When contributing to this repository, first discuss the change you wish to make via:

-  [GitHub Issues](https://github.com/Xilinx/Vitis-AI/issues)
-  [Vitis AI Forums](https://forums.xilinx.com/t5/AI-and-Vitis-AI/bd-p/AI)
-  <a href="mailto:xilinx_ai_model_zoo@xilinx.com">Email</a>

You can also submit a pull request with details on how to improve the product. Prior to submitting your pull request, ensure that you can build the product and run all the demos with your patch. In case of a larger feature, provide a relevant demo.
