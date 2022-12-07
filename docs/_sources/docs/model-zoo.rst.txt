.. raw:: html

   <table class="sphinxhide">

.. raw:: html

   <tr>

.. raw:: html

   <td align="center">

.. raw:: html

   <h1>

Vitis AI

.. raw:: html

   </h1>

Adaptable & Real-Time AI Inference Acceleration

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table>

Introduction
============

This repository includes optimized deep learning models to speed up the
deployment of deep learning inference on Xilinx™ platforms. These models
cover different applications, including but not limited to ADAS/AD,
medical, video surveillance, robotics, data center, etc. You can get
started with these free pre-trained models to enjoy the benefits of deep
learning acceleration.

Model Details
-------------

The following tables include comprehensive information about all models,
including application, framework, input size, computation Flops as well
as float and quantized accuracy.

Naming Rules
~~~~~~~~~~~~

Model name: ``F_M_(D)_H_W_(P)_C_V`` \* ``F`` specifies training
framework: ``tf`` is Tensorflow 1.x, ``tf2`` is Tensorflow 2.x, ``pt``
is PyTorch \* ``M`` specifies the model \* ``D`` specifies the dataset.
It is optional depending on whether the dataset is public or private \*
``H`` specifies the height of input data \* ``W`` specifies the width of
input data \* ``P`` specifies the pruning ratio, it means how much
computation is reduced. It is optional depending on whether the model is
pruned or not \* ``C`` specifies the computation of the model: how many
Gops per image \* ``V`` specifies the version of Vitis-AI

For example, ``pt_fadnet_sceneflow_576_960_0.65_154G_2.5`` is ``FADNet``
model trained with ``Pytorch`` using ``SceneFlow`` dataset, input size
is ``576*960``, ``65%`` pruned, the computation per image is
``154 Gops`` and Vitis-AI version is ``2.5``.

Model Index
~~~~~~~~~~~

-  Computation OPS in the table are counted as FLOPs
-  Float & Quantized Accuracy unless otherwise specified, the default
   refers to top1 or top1/top5
-  Models that have AMD EPYC™ CPU version are marked with

   .. raw:: html

      <html>

   ⭐

   .. raw:: html

      </html>

-  For more details and downloading CPU models, please refer to
   `UIF <https://github.com/amd/UIF>`__
-  Supported Tasks

   -  `Classification <#Classification>`__
   -  `Detection <#Detection>`__
   -  `Segmentation <#Segmentation>`__
   -  `NLP <#NLP>`__
   -  `Text-OCR <#Text-OCR>`__
   -  `Surveillance <#Surveillance>`__
   -  `Industrial-Vision-Robotics <#Industrial-Vision-Robotics>`__
   -  `Medical-Image-Enhancement <#Medical-Image-Enhancement>`__

Classification
--------------

+---+----------+-----------+-----------+-----------+-----------+-------+
| N | App      | Name      | Float     | Quantized | Input     | OPS   |
| o | lication |           | Accuracy  | Accuracy  | Size      |       |
| . |          |           |           |           |           |       |
+===+==========+===========+===========+===========+===========+=======+
| 1 | General  | tf_       | 0.8037    | 0.7946    | 299*299   | 2     |
|   |          | inception |           |           |           | 6.35G |
|   |          | resnetv2_ |           |           |           |       |
|   |          | imagenet_ |           |           |           |       |
|   |          | 299_299_2 |           |           |           |       |
|   |          | 6.35G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 2 | General  | tf        | 0.6976    | 0.6794    | 224*224   | 3G    |
|   |          | _inceptio |           |           |           |       |
|   |          | nv1_image |           |           |           |       |
|   |          | net_224_2 |           |           |           |       |
|   |          | 24_3G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 3 | General  | tf_in     | 0.7399    | 07331     | 224*224   | 3.88G |
|   |          | ceptionv2 |           |           |           |       |
|   |          | _imagenet |           |           |           |       |
|   |          | _224_224_ |           |           |           |       |
|   |          | 3.88G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+

\| 4 \| General \| tf_inceptionv3_imagenet_299_299_11.45G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

         |                           0\.7798                            |                           0\.7735                            |                           299\*299                           |                11\.45G                |

\| 5 \| General \| tf_inceptionv3_imagenet_299_299_0.2_9.1G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

             |                            0.7786                            |                            0.7668                            |                           299\*299                           |                 9.1G                  |

\| 6 \| General \| tf_inceptionv3_imagenet_299_299_0.4_6.9G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

             |                            0.7669                            |                            0.7561                            |                           299\*299                           |                 6.9G                  |

\| 7 \| General \| tf_inceptionv4_imagenet_299_299_24.55G_2.5 \| 0.8018
\| 0.7928 \| 299*299 \| 24.55G \| \| 8 \| General \|
tf_mobilenetv1_0.25_imagenet_128_128_27M_2.5 \| 0.4144 \| 0.3464 \|
128*128 \| 27.15M \| \| 9 \| General \|
tf_mobilenetv1_0.5_imagenet_160_160_150M_2.5 \| 0.5903 \| 0.5195 \|
160*160 \| 150M \| \| 10 \| General \|
tf_mobilenetv1_1.0_imagenet_224_224_1.14G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

    |                           0\.7102                            |                           0\.6780                            |                           224\*224                           |                1\.14G                 |

\| 11 \| General \| tf_mobilenetv1_1.0_imagenet_224_224_0.11_1.02G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

       |                            0.7056                            |                            0.6822                            |                           224\*224                           |                 1.02G                 |

\| 12 \| General \| tf_mobilenetv1_1.0_imagenet_224_224_0.12_1G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

          |                            0.7060                            |                            0.6850                            |                           224\*224                           |                  1G                   |

\| 13 \| General \| tf_mobilenetv2_1.0_imagenet_224_224_602M_2.5 \|
0.7013 \| 0.6767 \| 224*224 \| 602M \| \| 14 \| General \|
tf_mobilenetv2_1.4_imagenet_224_224_1.16G_2.5 \| 0.7411 \| 0.7194 \|
224*224 \| 1.16G \| \| 15 \| General \|
tf_resnetv1_50_imagenet_224_224_6.97G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

         |                           0\.7520                            |                           0\.7436                            |                           224\*224                           |                6\.97G                 |

\| 16 \| General \| tf_resnetv1_50_imagenet_224_224_0.38_4.3G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

            |                            0.7442                            |                            0.7375                            |                           224\*224                           |                 4.3G                  |

\| 17 \| General \| tf_resnetv1_50_imagenet_224_224_0.65_2.45G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

           |                            0.7279                            |                            0.7167                            |                           224\*224                           |                 2.45G                 |

\| 18 \| General \| tf_resnetv1_101_imagenet_224_224_14.4G_2.5 \| 0.7640
\| 0.7560 \| 224*224 \| 14.4G \| \| 19 \| General \|
tf_resnetv1_152_imagenet_224_224_21.83G_2.5 \| 0.7681 \| 0.7463 \|
224*224 \| 21.83G \| \| 20 \| General \|
tf_vgg16_imagenet_224_224_30.96G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

               |                           0\.7089                            |                           0\.7069                            |                           224\*224                           |                30\.96G                |

\| 21 \| General \| tf_vgg16_imagenet_224_224_0.43_17.67G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

                |                            0.6929                            |                            0.6823                            |                           224\*224                           |                17.67G                 |

\| 22 \| General \| tf_vgg16_imagenet_224_224_0.5_15.64G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

                 |                            0.6857                            |                            0.6729                            |                           224\*224                           |                15.64G                 |

\| 23 \| General \| tf_vgg19_imagenet_224_224_39.28G_2.5 \| 0.7100 \|
0.7026 \| 224*224 \| 39.28G \| \| 24 \| General \|
tf_resnetv2_50_imagenet_299_299_13.1G_2.5 \| 0.7559 \| 0.7445 \|
299\ *299 \| 13.1G \| \| 25 \| General \|
tf_resnetv2_101_imagenet_299_299_26.78G_2.5 \| 0.7695 \| 0.7506 \|
299*\ 299 \| 26.78G \| \| 26 \| General \|
tf_resnetv2_152_imagenet_299_299_40.47G_2.5 \| 0.7779 \| 0.7432 \|
299\ *299 \| 40.47G \| \| 27 \| General \|
tf_efficientnet-edgetpu-S_imagenet_224_224_4.72G_2.5 \| 0.7702/0.9377 \|
0.7660/0.9337 \| 224*\ 224 \| 4.72G \| \| 28 \| General \|
tf_efficientnet-edgetpu-M_imagenet_240_240_7.34G_2.5 \| 0.7862/0.9440 \|
0.7798/0.9406 \| 240\ *240 \| 7.34G \| \| 29 \| General \|
tf_efficientnet-edgetpu-L_imagenet_300_300_19.36G_2.5 \| 0.8026/0.9514
\| 0.7996/0.9491 \| 300*\ 300 \| 19.36G \| \| 30 \| General \|
tf_mlperf_resnet50_imagenet_224_224_8.19G_2.5 \| 0.7652 \| 0.7606 \|
224\ *224 \| 8.19G \| \| 31 \| General \|
tf_mobilenetEdge1.0_imagenet_224_224_990M_2.5 \| 0.7227 \| 0.6775 \|
224*\ 224 \| 990M \| \| 32 \| General \|
tf_mobilenetEdge0.75_imagenet_224_224_624M_2.5 \| 0.7201 \| 0.6489 \|
224\ *224 \| 624M \| \| 33 \| General \|
tf2_resnet50_imagenet_224_224_7.76G_2.5 \| 0.7513 \| 0.7423 \| 224*\ 224
\| 7.76G \| \| 34 \| General \|
tf2_mobilenetv1_imagenet_224_224_1.15G_2.5 \| 0.7005 \| 0.5603 \|
224\ *224 \| 1.15G \| \| 35 \| General \|
tf2_inceptionv3_imagenet_299_299_11.5G_2.5 \| 0.7753 \| 0.7694 \|
299*\ 299 \| 11.5G \| \| 36 \| General \|
tf2_efficientnet-b0_imagenet_224_224_0.78G_2.5 \| 0.7690/0.9320 \|
0.7515/0.9273 \| 224\ *224 \| 0.36G \| \| 37 \| General \|
tf2_mobilenetv3_imagenet_224_224_132M_2.5 \| 0.6756/0.8728 \|
0.6536/0.8544 \| 224*\ 224 \| 132M \| \| 38 \| General \|
pt_inceptionv3_imagenet_299_299_11.4G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

                |                         0.775/0.936                          |                         0.771/0.935                          |                           299*299                            |                 11.4G                 |

\| 39 \| General \| pt_inceptionv3_imagenet_299_299_0.3_8G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

               |                         0.775/0.936                          |                         0.772/0.935                          |                           299*299                            |                  8G                   |

\| 40 \| General \| pt_inceptionv3_imagenet_299_299_0.4_6.8G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

             |                         0.768/0.931                          |                         0.764/0.929                          |                           299*299                            |                 6.8G                  |

\| 41 \| General \| pt_inceptionv3_imagenet_299_299_0.5_5.7G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

             |                         0.757/0.921                          |                         0.752/0.918                          |                           299*299                            |                 5.7G                  |

\| 42 \| General \| pt_inceptionv3_imagenet_299_299_0.6_4.5G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

             |                         0.739/0.911                          |                         0.732/0.908                          |                           299*299                            |                 4.5G                  |

\| 43 \| General \| pt_squeezenet_imagenet_224_224_703.5M_2.5 \|
0.582/0.806 \| 0.582/0.806 \| 224*224 \| 703.5M \| \| 44 \| General \|
pt_resnet50_imagenet_224_224_8.2G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

                    |                         0.761/0.929                          |                         0.760/0.928                          |                           224*224                            |                 8.2G                  |

\| 45 \| General \| pt_resnet50_imagenet_224_224_0.3_5.8G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

                |                         0.760/0.929                          |                         0.757/0.928                          |                           224*224                            |                 5.8G                  |

\| 46 \| General \| pt_resnet50_imagenet_224_224_0.4_4.9G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

                |                         0.755/0.926                          |                         0.752/0.925                          |                           224*224                            |                 4.9G                  |

\| 47 \| General \| pt_resnet50_imagenet_224_224_0.5_4.1G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

                |                         0.748/0.921                          |                         0.745/0.920                          |                           224*224                            |                 4.1G                  |

\| 48 \| General \| pt_resnet50_imagenet_224_224_0.6_3.3G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

                |                         0.742/0.917                          |                         0.738/0.915                          |                           224*224                            |                 3.3G                  |

\| 49 \| General \| pt_resnet50_imagenet_224_224_0.7_2.5G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

                |                         0.726/0.908                          |                         0.720/0.906                          |                           224*224                            |                 2.5G                  |

\| 50 \| General \| pt_OFA-resnet50_imagenet_224_224_15.0G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

               |                         0.799/0.948                          |                         0.789/0.944                          |                           224*224                            |                 15.0G                 |

\| 51 \| General \| pt_OFA-resnet50_imagenet_224_224_0.45_8.2G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

           |                         0.795/0.945                          |                         0.784/0.941                          |                           224*224                            |                 8.2G                  |

\| 52 \| General \| pt_OFA-resnet50_imagenet_224_224_0.60_6.0G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

           |                         0.791/0.943                          |                         0.780/0.939                          |                           224*224                            |                 6.0G                  |

\| 53 \| General \| pt_OFA-resnet50_imagenet_192_192_0.74_3.6G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

           |                         0.777/0.937                          |                         0.770/0.933                          |                           192*192                            |                 3.6G                  |

\| 54 \| General \| pt_OFA-resnet50_imagenet_160_160_0.88_1.8G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

           |                         0.752/0.918                          |                         0.744/0.918                          |                           160*160                            |                 1.8G                  |

| 55 \| General \| pt_OFA-depthwise-res50_imagenet_176_176_2.49G_2.5 \|
  0.7633/0.9292 \| 0.7629/0.9306 \| 176*176 \| 2.49G \|
| 56 \| General \| tf_ViT_imagenet_352_352_21.3G_2.5 \| 0.8282 \| 0.8254
  \| 352*352 \| 21.3G \|
| 57 \| Car type classification \|
  pt_vehicle-type-classification_CompCars_224_224_3.63G_2.5 \| 0.9025 \|
  0.9011 \| 224*224 \| 3.63G \|
| 58 \| Car make classification \|
  pt_vehicle-make-classification_CompCars_224_224_3.63G_2.5 \| 0.8991 \|
  0.8939 \| 224*224 \| 3.63G \|
| 59 \| Car color classification \|
  pt_vehicle-color-classification_color_224_224_3.63G_2.5 \| 0.9549 \|
  0.9549 \| 224*224 \| 3.63G \|

Note: #30 & #44-#49 are resnet_v1.5, #33 is resnet_v1, #50-#54 are
resnet-D.

Detection
---------

+---+----------+-----------+-----------+-----------+-----------+-------+
| N | App      | Name      | Float     | Quantized | Input     | OPS   |
| o | lication |           | Accuracy  | Accuracy  | Size      |       |
| . |          |           |           |           |           |       |
+===+==========+===========+===========+===========+===========+=======+
| 1 | General  | tf_s      | 0.2080    | 0.2100    | 300*300   | 2.47G |
|   |          | sdmobilen |           |           |           |       |
|   |          | etv1_coco |           |           |           |       |
|   |          | _300_300_ |           |           |           |       |
|   |          | 2.47G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 2 | General  | tf_s      | 0.2150    | 0.2110    | 300*300   | 3.75G |
|   |          | sdmobilen |           |           |           |       |
|   |          | etv2_coco |           |           |           |       |
|   |          | _300_300_ |           |           |           |       |
|   |          | 3.75G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 3 | General  | tf_ssdre  | 0.3010    | 0.2900    | 640*640   | 1     |
|   |          | snet50v1_ |           |           |           | 78.4G |
|   |          | fpn_coco_ |           |           |           |       |
|   |          | 640_640_1 |           |           |           |       |
|   |          | 78.4G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 4 | General  | tf_yo     | 0.7846    | 0.7744    | 416*416   | 6     |
|   |          | lov3_voc_ |           |           |           | 5.63G |
|   |          | 416_416_6 |           |           |           |       |
|   |          | 5.63G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 5 | General  | tf_mlp    | 0.2250    | 0.2150    | 1200*1200 | 433G  |
|   |          | erf_resne |           |           |           |       |
|   |          | t34_coco_ |           |           |           |       |
|   |          | 1200_1200 |           |           |           |       |
|   |          | _433G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 6 | General  | tf_ssdli  | 0.2170    | 0.2090    | 300*300   | 1.5G  |
|   |          | te_mobile |           |           |           |       |
|   |          | netv2_coc |           |           |           |       |
|   |          | o_300_300 |           |           |           |       |
|   |          | _1.5G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 7 | General  | tf_s      | 0.2390    | 0.2360    | 300*300   | 9.62G |
|   |          | sdincepti |           |           |           |       |
|   |          | onv2_coco |           |           |           |       |
|   |          | _300_300_ |           |           |           |       |
|   |          | 9.62G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 8 | General  | tf_refi   | 0.8015    | 0.7999    | 320*320   | 81.9G |
|   |          | nedet_VOC |           |           |           |       |
|   |          | _320_320_ |           |           |           |       |
|   |          | 81.9G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 9 | General  | tf_eff    | 0.4130    | 0.3270    | 768*768   | 1     |
|   |          | icientdet |           |           |           | 1.06G |
|   |          | -d2_coco_ |           |           |           |       |
|   |          | 768_768_1 |           |           |           |       |
|   |          | 1.06G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | General  | tf2_yo    | 0.377     | 0.331     | 416*416   | 65.9G |
| 0 |          | lov3_coco |           |           |           |       |
|   |          | _416_416_ |           |           |           |       |
|   |          | 65.9G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | General  | tf_yo     | 0.477     | 0.393     | 416*416   | 60.3G |
| 1 |          | lov4_coco |           |           |           |       |
|   |          | _416_416_ |           |           |           |       |
|   |          | 60.3G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | General  | tf_yo     | 0.487     | 0.412     | 512*512   | 91.2G |
| 2 |          | lov4_coco |           |           |           |       |
|   |          | _512_512_ |           |           |           |       |
|   |          | 91.2G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | General  | pt_OFA-y  | 0.436     | 0.421     | 640*640   | 4     |
| 3 |          | olo_coco_ |           |           |           | 8.88G |
|   |          | 640_640_4 |           |           |           |       |
|   |          | 8.88G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | General  | pt_       | 0.420     | 0.401     | 640*640   | 3     |
| 4 |          | OFA-yolo_ |           |           |           | 4.72G |
|   |          | coco_640_ |           |           |           |       |
|   |          | 640_0.3_3 |           |           |           |       |
|   |          | 4.72G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | General  | pt_       | 0.392     | 0.378     | 640*640   | 2     |
| 5 |          | OFA-yolo_ |           |           |           | 4.62G |
|   |          | coco_640_ |           |           |           |       |
|   |          | 640_0.5_2 |           |           |           |       |
|   |          | 4.62G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+

\| 16 \| Medical Detection \|
tf_RefineDet-Medical_EDD_320_320_81.28G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

              |                            0.7866                            |                            0.7857                            |                           320*320                            |                81.28G                 |

\| 17 \| Medical Detection \|
tf_RefineDet-Medical_EDD_320_320_0.5_41.42G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

          |                            0.7798                            |                            0.7772                            |                           320*320                            |                41.42G                 |

\| 18 \| Medical Detection \|
tf_RefineDet-Medical_EDD_320_320_0.75_20.54G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

         |                            0.7885                            |                            0.7826                            |                           320*320                            |                20.54G                 |

\| 19 \| Medical Detection \|
tf_RefineDet-Medical_EDD_320_320_0.85_12.32G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

         |                            0.7898                            |                            0.7877                            |                           320*320                            |                12.32G                 |

\| 20 \| Medical Detection \|
tf_RefineDet-Medical_EDD_320_320_0.88_9.83G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

          |                            0.7839                            |                            0.8002                            |                           320*320                            |                 9.83G                 |

| 21 \| ADAS Traffic sign Detection \| pt_yolox_TT100K_640_640_73G_2.5
  \| 0.623 \| 0.621 \| 640*640 \| 73G \|
| 22 \| ADAS Lane Detection \| pt_ultrafast_CULane_288_800_8.4G_2.5 \|
  0.6988 \| 0.6922 \| 288*800 \| 8.4G \|
| 23 \| ADAS 3D Detection \| pt_pointpillars_kitti_12000_100_10.8G_2.5
  \| Car 3D AP@0.5(easy, moderate, hard) 90.79, 89.66, 88.78 \| Car 3D
  AP@0.5(easy, moderate, hard) 90.75, 87.04, 83.44 \| 12000*100*4 \|
  10.8G \|
| 24 \| ADAS Surround-view 3D Detection \|
  pt_pointpillars_nuscenes_40000_64_108G_2.5 \| mAP: 42.2NDS: 55.1 \|
  mAP: 40.5NDS: 53.0 \| 40000*64*5 \| 108G \|
| 25 \| ADAS 4D radar based 3D Detection \|
  pt_centerpoint_astyx_2560_40_54G_2.5 \| BEV AP@0.5: 32.843D AP@0.5:
  28.27 \| BEV AP@0.5: 33.823D AP@0.5: 18.54(QAT) \| 2560*40*4 \| 54G \|
| 26 \| ADAS Image-lidar fusion based 3D Detection \| pt_CLOCs_kitti_2.5
  \| 2d detection: Mod Car bbox AP@0.70: 89.403d detection: Mod Car
  bev@0.7 :85.50Mod Car 3d@0.7 :70.01Mod Car bev@0.5 :89.69Mod Car
  3d@0.5 :89.48fusionnet: Mod Car bev@0.7 :87.58Mod Car 3d@0.7 :73.04Mod
  Car bev@0.5 :93.98Mod Car 3d@0.5 :93.56 \| 2d detection: Mod Car bbox
  AP@0.70: 89.503d detection: Mod Car bev@0.7 :85.50Mod Car 3d@0.7
  :70.01Mod Car bev@0.5 :89.69Mod Car 3d@0.5 :89.48fusionnet: Mod Car
  bev@0.7 :87.58Mod Car 3d@0.7 :73.04Mod Car bev@0.5 :93.98Mod Car
  3d@0.5 :93.56 \| 2d detection (YOLOX): 384*12483d detection
  (PointPillars): 12000*100*4fusionnet: 800*1000*4 \| 41G \|
| 27 \| ADAS Image-lidar sensor fusion Detection & Segmentation \|
  pt_pointpainting_nuscenes_126G_2.5 \| mIoU: 69.1mAP: 51.8NDS: 58.7 \|
  mIoU: 68.6mAP: 50.4NDS: 56.4 \|
  semanticfpn:320*576pointpillars:40000*64*16 \|
  semanticfpn:14Gpointpillars:112G \|
| 28 \| ADAS Multi Task \| pt_MT-resnet18_mixed_320_512_13.65G_2.5 \|
  mAP:39.51mIOU:44.03 \| mAP:38.41mIOU:42.71 \| 320*512 \| 13.65G \|
| 29 \| ADAS Multi Task \| pt_multitaskv3_mixed_320_512_25.44G_2.5 \|
  mAP:51.2mIOU:58.14Drivable mIOU: 82.57Lane IOU:43.71Silog: 8.78 \|
  mAP:50.9mIOU:57.52Drivable mIOU: 82.30Lane IOU:44.01Silog: 9.32 \|
  320*512 \| 25.44G \|

| 
| 

Segmentation
------------

+---+----------+-----------+-----------+-----------+-----------+-------+
| N | App      | Name      | Float     | Quantized | Input     | OPS   |
| o | lication |           | Accuracy  | Accuracy  | Size      |       |
| . |          |           |           |           |           |       |
+===+==========+===========+===========+===========+===========+=======+
| 1 | ADAS 2D  | pt_ENet_c | 0.6442    | 0.6315    | 512*1024  | 8.6G  |
|   | Segm     | ityscapes |           |           |           |       |
|   | entation | _512_1024 |           |           |           |       |
|   |          | _8.6G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 2 | ADAS 2D  | pt_Se     | 0.6290    | 0.6230    | 256*512   | 10G   |
|   | Segm     | manticFPN |           |           |           |       |
|   | entation | -resnet18 |           |           |           |       |
|   |          | _cityscap |           |           |           |       |
|   |          | es_256_51 |           |           |           |       |
|   |          | 2_10G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 3 | ADAS 2D  | p         | 0.6870    | 0.6820    | 512*1024  | 5.4G  |
|   | Segm     | t_Semanti |           |           |           |       |
|   | entation | cFPN-mobi |           |           |           |       |
|   |          | lenetv2_c |           |           |           |       |
|   |          | ityscapes |           |           |           |       |
|   |          | _512_1024 |           |           |           |       |
|   |          | _5.4G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 4 | ADAS 3D  | pt_s      | Acc avg   | Acc avg   | 64*2048   | 20.4G |
|   | Point    | alsanext_ | 0.8860IoU | 0.8350IoU |           |       |
|   | Cloud    | semantic- | avg       | avg       |           |       |
|   | Segm     | kitti_64_ | 0.5100    | 0.4540    |           |       |
|   | entation | 2048_0.6_ |           |           |           |       |
|   |          | 20.4G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 5 | ADAS 3D  | pt_sa     | mIou:     | mIou:     | 64*2048*5 | 32G   |
|   | Point    | lsanextv2 | 54.2%     | 54.2%     |           |       |
|   | Cloud    | _semantic |           |           |           |       |
|   | Segm     | -kitti_64 |           |           |           |       |
|   | entation | _2048_0.7 |           |           |           |       |
|   |          | 5_32G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 6 | ADAS     | pt        | 0.242     | 0.212     | 640*640   | 107G  |
|   | Instance | _SOLO_coc |           |           |           |       |
|   | Segm     | o_640_640 |           |           |           |       |
|   | entation | _107G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 7 | ADAS 2D  | tf        | 0.6263    | 0.4578    | 1024*2048 | 13    |
|   | Segm     | _mobilene |           |           |           | 2.74G |
|   | entation | tv2_citys |           |           |           |       |
|   |          | capes_102 |           |           |           |       |
|   |          | 4_2048_13 |           |           |           |       |
|   |          | 2.74G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 8 | ADAS 2D  | tf        | 0.5298    | 0.5167    | 512*1024  | 54G   |
|   | Segm     | 2_erfnet_ |           |           |           |       |
|   | entation | cityscape |           |           |           |       |
|   |          | s_512_102 |           |           |           |       |
|   |          | 4_54G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 9 | Medical  | tf2_2d-un | 0.3968    | 0.3968    | 128*128   | 5.31G |
|   | Cell     | et_nuclei |           |           |           |       |
|   | Nuclear  | _128_128_ |           |           |           |       |
|   | Segm     | 5.31G_2.5 |           |           |           |       |
|   | entation |           |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | Medical  | pt_FPN-re | 2-classes | 2-classes | 352*352   | 22.7G |
| 0 | Covid-19 | snet18_co | Di        | Di        |           |       |
|   | Segm     | vid19-seg | ce:0.8588 | ce:0.8547 |           |       |
|   | entation | _352_352_ | 3-classes | 3-classes |           |       |
|   |          | 22.7G_2.5 | mI        | mI        |           |       |
|   |          |           | oU:0.5989 | oU:0.5957 |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | Medical  | pt_unet   | Di        | Di        | 512*512   | 23.3G |
| 1 | CT lung  | _chaos-CT | ce:0.9758 | ce:0.9747 |           |       |
|   | Segm     | _512_512_ |           |           |           |       |
|   | entation | 23.3G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | Medical  | pt_HardN  | mDi       | mDi       | 352*352   | 2     |
| 2 | Polyp    | et_mixed_ | ce=0.9142 | ce=0.9136 |           | 2.78G |
|   | Segm     | 352_352_2 |           |           |           |       |
|   | entation | 2.78G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 1 | RGB-D    | pt_sa-    | miou:     | miou:     | 360*360   | 178G  |
| 3 | Segm     | gate_NYUv | 47.58%    | 46.80%    |           |       |
|   | entation | 2_360_360 |           |           |           |       |
|   |          | _178G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+

NLP
---

+---+----------+-----------+-----------+-----------+-----------+-------+
| N | App      | Name      | Float     | Quantized | Input     | OPS   |
| o | lication |           | Accuracy  | Accuracy  | Size      |       |
| . |          |           |           |           |           |       |
+===+==========+===========+===========+===========+===========+=======+
| 1 | Question | tf_ber    | 0.8694    | 0.8656    | 128       | 1     |
|   | and      | t-base_SQ |           |           |           | 1.17G |
|   | A        | uAD_128_2 |           |           |           |       |
|   | nswering | 2.34G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 2 | S        | tf2_senti | 0.8708    | 0.8695    | 500*32    | 53.3M |
|   | entiment | ment-dete |           |           |           |       |
|   | D        | ction_IMD |           |           |           |       |
|   | etection | B_500_32_ |           |           |           |       |
|   |          | 53.3M_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 3 | Customer | tf        | 0.9565    | 0.9565    | 25*32     | 2.7M  |
|   | Sati     | 2_custome |           |           |           |       |
|   | sfaction | r-satisfa |           |           |           |       |
|   | As       | ction_Car |           |           |           |       |
|   | sessment | s4U_25_32 |           |           |           |       |
|   |          | _2.7M_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 4 | Open     | pt_open-  | Acc/F1-s  | Acc/F1-s  | 100*200   | 1.5G  |
|   | Inf      | informati | core:58.7 | core:58.7 |           |       |
|   | ormation | on-extrac | 0%/77.12% | 0%/77.19% |           |       |
|   | Ex       | tion_qasr |           |           |           |       |
|   | traction | l_100_200 |           |           |           |       |
|   |          | _1.5G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+

Text-OCR
--------

+---+----------+-----------+-----------+-----------+-----------+-------+
| N | App      | Name      | Float     | Quantized | Input     | OPS   |
| o | lication |           | Accuracy  | Accuracy  | Size      |       |
| . |          |           |           |           |           |       |
+===+==========+===========+===========+===========+===========+=======+
| 1 | Text     | pt_t      | 0.8863    | 0.8851    | 960*960   | 5     |
|   | D        | extmounta |           |           |           | 75.2G |
|   | etection | in_ICDAR_ |           |           |           |       |
|   |          | 960_960_5 |           |           |           |       |
|   |          | 75.2G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 2 | E2E OCR  | pt_OCR    | 0.6758    | 0.6776    | 960*960   | 875G  |
|   |          | _ICDAR201 |           |           |           |       |
|   |          | 5_960_960 |           |           |           |       |
|   |          | _875G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+

Surveillance
------------

+---+----------+-----------+-----------+-----------+-----------+-------+
| N | App      | Name      | Float     | Quantized | Input     | OPS   |
| o | lication |           | Accuracy  | Accuracy  | Size      |       |
| . |          |           |           |           |           |       |
+===+==========+===========+===========+===========+===========+=======+
| 1 | Face     | pt_fa     | 0.9955    | 0.9947    | 112*96    | 3.5G  |
|   | Rec      | cerec-res |           |           |           |       |
|   | ognition | net20_mix |           |           |           |       |
|   |          | ed_112_96 |           |           |           |       |
|   |          | _3.5G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 2 | Face     | pt_fa     | 0.1233    | 0.1258    | 80*60     | 6     |
|   | Quality  | ce-qualit |           |           |           | 1.68M |
|   |          | y_80_60_6 |           |           |           |       |
|   |          | 1.68M_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 3 | Face     | pt_fa     | mAP:0.794 | mAP:0.790 | 96*96     | 515M  |
|   | ReID     | cereid-la | Ra        | Ra        |           |       |
|   |          | rge_96_96 | nk1:0.955 | nk1:0.953 |           |       |
|   |          | _515M_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 4 | Face     | pt_f      | mAP:0.560 | mAP:0.559 | 80*80     | 90M5  |
|   | ReID     | acereid-s | Ra        | Ra        |           |       |
|   |          | mall_80_8 | nk1:0.865 | nk1:0.865 |           |       |
|   |          | 0_90M_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 5 | ReID     | p         | mAP:0.753 | mAP:0.746 | 176*80    | 1.1G  |
|   |          | t_personr | Ra        | Ra        |           |       |
|   |          | eid-res18 | nk1:0.898 | nk1:0.893 |           |       |
|   |          | _market15 |           |           |           |       |
|   |          | 01_176_80 |           |           |           |       |
|   |          | _1.1G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+

\| 6 \| ReID \| pt_personreid-res50_market1501_256_128_5.3G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

          |                    mAP:0.866  Rank1:0.951                    |                    mAP:0.869  Rank1:0.948                    |                           256*128                            |                 5.3G                  |

\| 7 \| ReID \| pt_personreid-res50_market1501_256_128_0.4_3.3G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

      |                    mAP:0.869  Rank1:0.948                    |                    mAP:0.869  Rank1:0.948                    |                           256*128                            |                 3.3G                  |

\| 8 \| ReID \| pt_personreid-res50_market1501_256_128_0.5_2.7G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

      |                    mAP:0.864  Rank1:0.944                    |                    mAP:0.864  Rank1:0.944                    |                           256*128                            |                 2.7G                  |

\| 9 \| ReID \| pt_personreid-res50_market1501_256_128_0.6_2.1G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

      |                    mAP:0.863  Rank1:0.946                    |                    mAP:0.859  Rank1:0.942                    |                           256*128                            |                 2.1G                  |

\| 10 \| ReID \| pt_personreid-res50_market1501_256_128_0.7_1.6G_2.5

.. raw:: html

   <html>

⭐

.. raw:: html

   </html>

::

      |                    mAP:0.850  Rank1:0.940                    |                    mAP:0.848  Rank1:0.938                    |                           256*128                            |                 1.6G                  |

| 11 \| Person orientation estimation \|
  pt_person-orientation_224_112_558M_2.5 \| 0.930 \| 0.929 \| 224*112 \|
  558M \|
| 12 \| Joint detection and Tracking \|
  pt_FairMOT_mixed_640_480_0.5_36G_2.5 \| MOTA 59.1%IDF1 62.5% \| MOTA
  58.1%IDF1 60.5% \| 640*480 \| 36G \|
| 13 \| Crowd Counting \| pt_BCC_shanghaitech_800_1000_268.9G_2.5 \|
  MAE: 65.83MSE: 111.75 \| MAE: 67.60MSE: 117.36 \| 800*1000 \| 268.9G
  \|
| 14 \| Face Mask Detection \| pt_face-mask-detection_512_512_0.59G_2.5
  \| 0.8860 \| 0.8810 \| 512*512 \| 0.59G \|
| 15 \| Pose Estimation \| pt_movenet_coco_192_192_0.5G_2.5 \| 0.7972 \|
  0.7984 \| 192*192 \| 0.5G \|

Industrial-Vision-Robotics
--------------------------

+---+----------+-----------+-----------+-----------+-----------+-------+
| N | App      | Name      | Float     | Quantized | Input     | OPS   |
| o | lication |           | Accuracy  | Accuracy  | Size      |       |
| . |          |           |           |           |           |       |
+===+==========+===========+===========+===========+===========+=======+
| 1 | Depth    | pt_fadnet | EPE:      | EPE:      | 576*960   | 359G  |
|   | Es       | _sceneflo | 0.926     | 1.169     |           |       |
|   | timation | w_576_960 |           |           |           |       |
|   |          | _441G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 2 | B        | pt_fa     | EPE:      | EPE:      | 576*960   | 154G  |
|   | inocular | dnet_scen | 0.823     | 1.158     |           |       |
|   | depth    | eflow_576 |           |           |           |       |
|   | es       | _960_0.65 |           |           |           |       |
|   | timation | _154G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 3 | B        | pt_ps     | EPE:      | EPE:      | 576*960   | 696G  |
|   | inocular | mnet_scen | 0.961     | 1.022     |           |       |
|   | depth    | eflow_576 |           |           |           |       |
|   | es       | _960_0.68 |           |           |           |       |
|   | timation | _696G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 4 | Pr       | pt        | 0.9640    | 0.9618    | 224*224   | 2.28G |
|   | oduction | _pmg_rp2k |           |           |           |       |
|   | Rec      | _224_224_ |           |           |           |       |
|   | ognition | 2.28G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 5 | Interest | t         | 83.4      | 84.3      | 480*640   | 52.4G |
|   | Point    | f_superpo | (thr=3)   | (thr=3)   |           |       |
|   | D        | int_mixed |           |           |           |       |
|   | etection | _480_640_ |           |           |           |       |
|   | and      | 52.4G_2.5 |           |           |           |       |
|   | Des      |           |           |           |           |       |
|   | cription |           |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 6 | Hier     | tf_HFN    | Day:      | Day:      | 960*960   | 2     |
|   | archical | et_mixed_ | 76        | 74        |           | 0.09G |
|   | Loca     | 960_960_2 | .2/83.6/9 | .2/82.4/8 |           |       |
|   | lization | 0.09G_2.5 | 0.0Night: | 9.2Night: |           |       |
|   |          |           | 58.2/     | 54.1/     |           |       |
|   |          |           | 68.4/80.6 | 66.3/73.5 |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+

Medical-Image-Enhancement
-------------------------

+---+----------+-----------+-----------+-----------+-----------+-------+
| N | App      | Name      | Float     | Quantized | Input     | OPS   |
| o | lication |           | Accuracy  | Accuracy  | Size      |       |
| . |          |           |           |           |           |       |
+===+==========+===========+===========+===========+===========+=======+
| 1 | Super    | t         | Set5      | Set5      | 360*640   | 8     |
|   | Re       | f_rcan_DI | Y_PSNR :  | Y_PSNR :  |           | 6.95G |
|   | solution | V2K_360_6 | 3         | 37        |           |       |
|   |          | 40_0.98_8 | 7.640SSIM | .2495SSIM |           |       |
|   |          | 6.95G_2.5 | : 0.959   | : 0.9556  |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 2 | Super    | pt_SES    | (Set5)    | (Set5)    | 360*640   | 7.48G |
|   | Re       | R-S_DIV2K | P         | P         |           |       |
|   | solution | _360_640_ | SNR/SSIM= | SNR/SSIM= |           |       |
|   |          | 7.48G_2.5 | 3         | 3         |           |       |
|   |          |           | 7.309/0.9 | 6.813/0.9 |           |       |
|   |          |           | 58(Set14) | 54(Set14) |           |       |
|   |          |           | P         | P         |           |       |
|   |          |           | SNR/SSIM= | SNR/SSIM= |           |       |
|   |          |           | 32.894/   | 32.607/   |           |       |
|   |          |           | 0.        | 0.        |           |       |
|   |          |           | 911(B100) | 906(B100) |           |       |
|   |          |           | P         | P         |           |       |
|   |          |           | SNR/SSIM= | SNR/SSIM= |           |       |
|   |          |           | 31.663/   | 31.443/   |           |       |
|   |          |           | 0.893(    | 0.889(    |           |       |
|   |          |           | Urban100) | Urban100) |           |       |
|   |          |           | PSNR/SSIM | PSNR/SSIM |           |       |
|   |          |           | =         | =         |           |       |
|   |          |           | 30.       | 29.       |           |       |
|   |          |           | 276/0.908 | 901/0.899 |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 3 | Super    | pt_OFA-r  | (Set5)    | (Set5)    | 360*640   | 45.7G |
|   | Re       | can_DIV2K | P         | P         |           |       |
|   | solution | _360_640_ | SNR/SSIM= | SNR/SSIM= |           |       |
|   |          | 45.7G_2.5 | 3         | 3         |           |       |
|   |          |           | 7.654/0.9 | 7.384/0.9 |           |       |
|   |          |           | 59(Set14) | 56(Set14) |           |       |
|   |          |           | P         | P         |           |       |
|   |          |           | SNR/SSIM= | SNR/SSIM= |           |       |
|   |          |           | 33.169/   | 33.012/   |           |       |
|   |          |           | 0.        | 0.        |           |       |
|   |          |           | 914(B100) | 911(B100) |           |       |
|   |          |           | P         | P         |           |       |
|   |          |           | SNR/SSIM= | SNR/SSIM= |           |       |
|   |          |           | 31.891/   | 31.785/   |           |       |
|   |          |           | 0.897(    | 0.894(    |           |       |
|   |          |           | Urban100) | Urban100) |           |       |
|   |          |           | PSNR/SSIM | PSNR/SSIM |           |       |
|   |          |           | =         | =         |           |       |
|   |          |           | 30.       | 30.       |           |       |
|   |          |           | 978/0.917 | 839/0.913 |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 4 | Image    | pt_DRU    | PSNR =    | PSNR =    | 528*608   | 0.4G  |
|   | D        | Net_Kvasi | 34.57     | 34.06     |           |       |
|   | enoising | r_528_608 |           |           |           |       |
|   |          | _0.4G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 5 | Spectral | pt        | Visu      | Visu      | 256*256   | 3     |
|   | Remove   | _SSR_CVC_ | alization | alization |           | 9.72G |
|   |          | 256_256_3 |           |           |           |       |
|   |          | 9.72G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+
| 6 | Coverage | pt_C2D2   | M         | MA        | 512*512   | 6.86G |
|   | Pr       | lite_CC20 | AE=7.566% | E=10.399% |           |       |
|   | ediction | _512_512_ |           |           |           |       |
|   |          | 6.86G_2.5 |           |           |           |       |
+---+----------+-----------+-----------+-----------+-----------+-------+

Model Download
--------------

Please visit the ‘/model_zoo/model-list’ subdirectory in the Github
repository. You will get download link and MD5 of all released models,
including pre-compiled models that running on different platforms.

.. raw:: html

   </details>

Automated Download Script
~~~~~~~~~~~~~~~~~~~~~~~~~

With downloader.py, you could quickly find the model you are interested
in and specify a version to download it immediately. Please make sure
that downloader.py and ‘/model_zoo/model-list’ folder are at the same
level directory.

::

   python3  downloader.py  

Step1: You need input framework and model name keyword. Use space
divide. If input ``all`` you will get list of all models.

tf: tensorflow1.x, tf2: tensorflow2.x, pt: pytorch, cf: caffe, dk:
darknet, all: list all model

Step2: Select the specified model based on standard name.

Step3: Select the specified hardware platform for your slected model.

For example, after running downloader.py and input ``tf resnet`` then
you will see the alternatives such as:

::

   0:  all
   1:  tf_resnetv1_50_imagenet_224_224_6.97G_2.5
   2:  tf_resnetv1_101_imagenet_224_224_14.4G_2.5
   3:  tf_resnetv1_152_imagenet_224_224_21.83G_2.5
   ......

After you input the num: 1, you will see the alternatives such as:

::

   0:  all
   1:  tf_resnetv1_50_imagenet_224_224_6.97G_2.5    GPU
   2:  resnet_v1_50_tf    ZCU102 & ZCU104 & KV260
   3:  resnet_v1_50_tf    VCK190
   4:  resnet_v1_50_tf    vck50006pe-DPUCVDX8H
   5:  resnet_v1_50_tf    vck50008pe-DPUCVDX8H-DWC
   6:  resnet_v1_50_tf    u50lv-DPUCAHX8H
   ......

Then you could choose it and input the number, the specified version of
model will be automatically downloaded to the current directory.

In addition, if you need download all models on all platforms at once,
you just need enter number in the order indicated by the tips of Step
1/2/3 (select: all -> 0 -> 0).

Model Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

Download and extract the model archive to your working area on the local
hard disk. For details on the various models, download link and MD5
checksum for the zip file of each model, refer to the subdirectory
‘/model_zoo/model-list’ within the Github repository.

Tensorflow Model Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a Tensorflow model, you should see the following directory
structure:

::

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

Pytorch Model Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a Pytorch model, you should see the following directory structure:

::

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
       

**Note:** For more information on Vitis-AI Quantizer such as
``vai_q_tensorflow`` and ``vai_q_pytorch``, please see the `Vitis AI
User Guide <https://docs.xilinx.com/r/en-US/ug1414-vitis-ai>`__.

Model Performance
-----------------

All the models in the Model Zoo have been deployed on Xilinx hardware
with `Vitis AI <https://github.com/Xilinx/Vitis-AI>`__ and `Vitis AI
Library <https://github.com/Xilinx/Vitis-AI/tree/master/examples/Vitis-AI-Library>`__.
The performance number including end-to-end throughput and latency for
each model on various boards with different DPU configurations are
listed in the following sections.

For more information about DPU, see `DPU IP Product
Guide <https://www.xilinx.com/cgi-bin/docs/ipdoc?c=dpu;v=latest;d=pg338-dpu.pdf>`__.

For RNN models such as NLP, please refer to
`DPU-for-RNN <https://github.com/Xilinx/Vitis-AI/blob/master/demo/DPU-for-RNN>`__
for dpu specification information.

Besides, for Transformer demos such as ViT, Bert-base you could refer to
`Transformer <https://github.com/Xilinx/Vitis-AI/tree/master/examples/Transformer>`__.

**Note:** The model performance number listed in the following sections
is generated with Vitis AI v2.5 and Vitis AI Lirary v2.5. For different
platforms, the different DPU configurations are used. Vitis AI and Vitis
AI Library can be downloaded for free from `Vitis AI
Github <https://github.com/Xilinx/Vitis-AI>`__ and `Vitis AI Library
Github <https://github.com/Xilinx/Vitis-AI/tree/master/examples/Vitis-AI-Library>`__.
We will continue to improve the performance with Vitis AI. The
performance number reported here is subject to change in the near
future.

Performance on ZCU102 (0432055-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measured with Vitis AI 2.5 and Vitis AI Library 2.5

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``ZCU102 (0432055-05)``
board with a ``3 * B4096  @ 281MHz`` DPU configuration:

+---+--------------+---------------------+---------------+---------------+
| N | Model        | GPU Model Standard  | E2E           | E2E           |
| o |              | Name                | throughput    | throughput    |
| . |              |                     | (fps) Single  | (fps) Multi   |
|   |              |                     | Thread        | Thread        |
+===+==============+=====================+===============+===============+
| 1 | bcc_pt       | pt_BCC_shanghait    | 3.31          | 10.86         |
|   |              | ech_800_1000_268.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | c2d2_lite_pt | pt_C2D2lite         | 2.84          | 5.24          |
|   |              | _CC20_512_512_6.86G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | centerpoint  | pt_centerpoin       | 16.02         | 47.9          |
|   |              | t_astyx_2560_40_54G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | CLOCs        | pt_CLOCs_kitti_2.0  | 2.85          | 10.18         |
+---+--------------+---------------------+---------------+---------------+
| 5 | drunet_pt    | pt_DRUNet_          | 60.77         | 189.53        |
|   |              | Kvasir_528_608_0.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | effici       | tf_efficientdet-d2_ | 3.10          | 6.05          |
|   | entdet_d2_tf | coco_768_768_11.06G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | efficie      | tf2_                | 78.02         | 152.61        |
|   | ntnet-b0_tf2 | efficientnet-b0_ima |               |               |
|   |              | genet_224_224_0.36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | e            | tf_efficien         | 34.99         | 90.14         |
|   | fficientNet- | tnet-edgetpu-L_imag |               |               |
|   | edgetpu-L_tf | enet_300_300_19.36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | e            | tf_efficie          | 79.88         | 205.22        |
|   | fficientNet- | ntnet-edgetpu-M_ima |               |               |
|   | edgetpu-M_tf | genet_240_240_7.34G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | e            | tf_efficie          | 115.05        | 308.15        |
| 0 | fficientNet- | ntnet-edgetpu-S_ima |               |               |
|   | edgetpu-S_tf | genet_224_224_4.72G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ENet_c       | pt_ENet_citys       | 10.10         | 36.43         |
| 1 | ityscapes_pt | capes_512_1024_8.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | face_mask_   | pt_face-mask-dete   | 115.80        | 401.08        |
| 2 | detection_pt | ction_512_512_0.59G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | fac          | pt_face-q           | 2988.30       | 8791.90       |
| 3 | e-quality_pt | uality_80_60_61.68M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | f            | pt_facerec-resnet2  | 168.33        | 337.68        |
| 4 | acerec-resne | 0_mixed_112_96_3.5G |               |               |
|   | t20_mixed_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | facer        | pt_facere           | 941.81        | 2275.70       |
| 5 | eid-large_pt | id-large_96_96_515M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | facer        | pt_facer            | 2240.00       | 6336.00       |
| 6 | eid-small_pt | eid-small_80_80_90M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | fadnet       | pt_fadnet_sce       | 1.17          | 1.59          |
| 7 |              | neflow_576_960_441G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | f            | pt_fadnet_sceneflo  | 1.76          | 2.67          |
| 8 | adnet_pruned | w_576_960_0.65_154G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | FairMot_pt   | pt_FairMOT_mi       | 22.59         | 66.03         |
| 9 |              | xed_640_480_0.5_36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | FPN          | pt_                 | 37.08         | 107.18        |
| 0 | -resnet18_co | FPN-resnet18_covid1 |               |               |
|   | vid19-seg_pt | 9-seg_352_352_22.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Har          | pt_HardNet_m        | 24.71         | 56.77         |
| 1 | dNet_MSeg_pt | ixed_352_352_22.78G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | HFnet_tf     | tf_HFNet_m          | 3.57          | 15.90         |
| 2 |              | ixed_960_960_20.09G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inception_   | tf_inc              | 23.89         | 51.93         |
| 3 | resnet_v2_tf | eptionresnetv2_imag |               |               |
|   |              | enet_299_299_26.35G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inc          | tf_inceptionv1_     | 191.26        | 470.88        |
| 4 | eption_v1_tf | imagenet_224_224_3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | inc          | tf_inceptionv2_ima  | 93.04         | 229.67        |
| 5 | eption_v2_tf | genet_224_224_3.88G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inc          | tf_inceptionv3_imag | 59.75         | 135.30        |
| 6 | eption_v3_tf | enet_299_299_11.45G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | in           | tf                  | 68.43         | 158.60        |
| 7 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_2_tf | et_299_299_0.2_9.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | in           | tf                  | 85.04         | 203.97        |
| 8 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_4_tf | et_299_299_0.4_6.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Ince         | tf2_inceptionv3_ima | 59.18         | 136.24        |
| 9 | ption_v3_tf2 | genet_299_299_11.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | Inc          | pt_inceptionv3_ima  | 59.83         | 136.02        |
| 0 | eption_v3_pt | genet_299_299_11.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt_inceptionv3_imag | 75.76         | 175.51        |
| 1 | ception_v3_p | enet_299_299_0.3_8G |               |               |
|   | runed_0_3_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 85.52         | 203.83        |
| 2 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_4_pt | et_299_299_0.4_6.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 96.39         | 234.92        |
| 3 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_5_pt | et_299_299_0.5_5.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 114.10        | 283.68        |
| 4 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_6_pt | et_299_299_0.6_4.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | Inc          | tf_inceptionv4_imag | 28.83         | 68.57         |
| 5 | eption_v4_tf | enet_299_299_24.55G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | medical_     | tf2_2d-unet_n       | 154.32        | 393.59        |
| 6 | seg_cell_tf2 | uclei_128_128_5.31G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mlperf_ssd   | tf_mlperf_resnet34_ | 1.87          | 7.12          |
| 7 | _resnet34_tf | coco_1200_1200_433G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mlperf       | tf_                 | 78.85         | 173.51        |
| 8 | _resnet50_tf | mlperf_resnet50_ima |               |               |
|   |              | genet_224_224_8.19G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mobilenet_   | tf_m                | 262.66        | 720.24        |
| 9 | edge_0_75_tf | obilenetEdge0.75_im |               |               |
|   |              | agenet_224_224_624M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet    | tf_                 | 214.78        | 554.55        |
| 0 | _edge_1_0_tf | mobilenetEdge1.0_im |               |               |
|   |              | agenet_224_224_990M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet    | tf2_mobilenetv1_ima | 321.47        | 939.61        |
| 1 | _1_0_224_tf2 | genet_224_224_1.15G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v1 | tf                  | 1332.90       | 4754.20       |
| 2 | _0_25_128_tf | _mobilenetv1_0.25_i |               |               |
|   |              | magenet_128_128_27M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf                  | 914.84        | 3116.80       |
| 3 | 1_0_5_160_tf | _mobilenetv1_0.5_im |               |               |
|   |              | agenet_160_160_150M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_                 | 326.62        | 951.00        |
| 4 | 1_1_0_224_tf | mobilenetv1_1.0_ima |               |               |
|   |              | genet_224_224_1.14G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_mobil            | 329.61        | 879.01        |
| 5 | 1_1_0_224_pr | enetv1_1.0_imagenet |               |               |
|   | uned_0_11_tf | _224_224_0.11_1.02G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_mo               | 335.60        | 934.39        |
| 6 | 1_1_0_224_pr | bilenetv1_1.0_image |               |               |
|   | uned_0_12_tf | net_224_224_0.12_1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf                  | 268.07        | 707.03        |
| 7 | 2_1_0_224_tf | _mobilenetv2_1.0_im |               |               |
|   |              | agenet_224_224_602M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_                 | 191.08        | 473.15        |
| 8 | 2_1_4_224_tf | mobilenetv2_1.4_ima |               |               |
|   |              | genet_224_224_1.16G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mo           | tf_mo               | 1.75          | 5.28          |
| 9 | bilenet_v2_c | bilenetv2_cityscape |               |               |
|   | ityscapes_tf | s_1024_2048_132.74G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mo           | tf2_mobilenetv3_im  | 336.62        | 965.76        |
| 0 | bilenet_v3_s | agenet_224_224_132M |               |               |
|   | mall_1_0_tf2 |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mo           | pt_movene           | 94.72         | 390.02        |
| 1 | venet_ntd_pt | t_coco_192_192_0.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | MT-resne     | pt_MT-resnet18_m    | 32.79         | 102.96        |
| 2 | t18_mixed_pt | ixed_320_512_13.65G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mult         | pt_multitaskv3_m    | 17.10         | 61.35         |
| 3 | i_task_v3_pt | ixed_320_512_25.44G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ocr_pt       | pt_OCR_ICD          | 1.10          | 3.41          |
| 4 |              | AR2015_960_960_875G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_depthw   | pt_OFA-             | 106.03        | 370.18        |
| 5 | ise_res50_pt | depthwise-res50_ima |               |               |
|   |              | genet_176_176_2.49G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_rca      | pt_OFA-rcan_        | 17.00         | 28.03         |
| 6 | n_latency_pt | DIV2K_360_640_45.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_resnet50 | pt_OFA-resnet50_ima | 47.64         | 93.90         |
| 7 | _baseline_pt | genet_224_224_15.0G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa          | pt_O                | 72.22         | 144.17        |
| 8 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_45_pt | t_224_224_0.45_8.2G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa          | pt_O                | 88.77         | 162.01        |
| 9 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_60_pt | t_224_224_0.60_6.0G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa          | pt_O                | 129.44        | 258.29        |
| 0 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_74_pt | t_192_192_0.74_3.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_resn     | pt_O                | 183.81        | 354.73        |
| 1 | et50_0_9B_pt | FA-resnet50_imagene |               |               |
|   |              | t_160_160_0.88_1.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pt  | pt_OFA-yolo_        | 16.87         | 42.75         |
| 2 |              | coco_640_640_48.88G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pr  | pt_OFA-yolo_coco    | 21.53         | 54.59         |
| 3 | uned_0_30_pt | _640_640_0.3_34.72G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pr  | pt_OFA-yolo_coco    | 27.71         | 71.35         |
| 4 | uned_0_50_pt | _640_640_0.6_24.62G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | pers         | pt_person-orien     | 661.42        | 1428.2        |
| 5 | on-orientati | tation_224_112_558M |               |               |
|   | on_pruned_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | personr      | pt_pe               | 107.54        | 237.66        |
| 6 | eid-res50_pt | rsonreid-res50_mark |               |               |
|   |              | et1501_256_128_5.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 118.42        | 300.87        |
| 7 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_4_pt | 01_256_128_0.4_3.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 126.01        | 320.82        |
| 8 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_5_pt | 01_256_128_0.5_2.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 136.24        | 362.63        |
| 9 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_6_pt | 01_256_128_0.6_2.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | person       | pt_person           | 144.01        | 376.18        |
| 0 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_7_pt | 01_256_128_0.7_1.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | personr      | pt_p                | 370.36        | 690.85        |
| 1 | eid-res18_pt | ersonreid-res18_mar |               |               |
|   |              | ket1501_176_80_1.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pmg_pt       | pt_pmg              | 151.69        | 366.62        |
| 2 |              | _rp2k_224_224_2.28G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointpaint   | pt_pointpaintin     | 1.28          | 4.25          |
| 3 | ing_nuscenes | g_nuscenes_126G_2.5 |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointp       | pt_pointpillars_ki  | 19.63         | 49.20         |
| 4 | illars_kitti | tti_12000_100_10.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointpill    | pt_pointpillars_nus | 2.23          | 9.61          |
| 5 | ars_nuscenes | cenes_40000_64_108G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | rc           | tf_rcan_DIV2K_      | 8.61          | 18.05         |
| 6 | an_pruned_tf | 360_640_0.98_86.95G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | refi         | tf_refinede         | 11.31         | 34.50         |
| 7 | nedet_VOC_tf | t_VOC_320_320_81.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | RefineDet    | t                   | 12.04         | 34.90         |
| 8 | -Medical_EDD | f_RefineDet-Medical |               |               |
|   | _baseline_tf | _EDD_320_320_81.28G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | RefineDet-M  | tf_Re               | 22.47         | 68.28         |
| 9 | edical_EDD_p | fineDet-Medical_EDD |               |               |
|   | runed_0_5_tf | _320_320_0.5_41.42G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Ref              | 39.53         | 125.60        |
| 0 | dical_EDD_pr | ineDet-Medical_EDD_ |               |               |
|   | uned_0_75_tf | 320_320_0.75_20.54G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Ref              | 59.57         | 197.08        |
| 1 | dical_EDD_pr | ineDet-Medical_EDD_ |               |               |
|   | uned_0_85_tf | 320_320_0.85_12.32G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Re               | 67.62         | 229.37        |
| 2 | dical_EDD_tf | fineDet-Medical_EDD |               |               |
|   |              | _320_320_0.88_9.83G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_resnetv1_50_ima  | 87.95         | 191.05        |
| 3 | net_v1_50_tf | genet_224_224_6.97G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_                 | 109.45        | 213.71        |
| 4 | net_v1_50_pr | resnetv1_50_imagene |               |               |
|   | uned_0_38_tf | t_224_224_0.38_4.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_r                | 138.81        | 276.33        |
| 5 | net_v1_50_pr | esnetv1_50_imagenet |               |               |
|   | uned_0_65_tf | _224_224_0.65_2.45G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | tf_resnetv1_101_ima | 46.38         | 110.72        |
| 6 | et_v1_101_tf | genet_224_224_14.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | t                   | 31.64         | 77.23         |
| 7 | et_v1_152_tf | f_resnetv1_152_imag |               |               |
|   |              | enet_224_224_21.83G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_resnetv2_50_ima  | 45.05         | 99.64         |
| 8 | net_v2_50_tf | genet_299_299_13.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | t                   | 23.60         | 55.60         |
| 9 | et_v2_101_tf | f_resnetv2_101_imag |               |               |
|   |              | enet_299_299_26.78G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resn         | t                   | 16.07         | 38.18         |
| 0 | et_v2_152_tf | f_resnetv2_152_imag |               |               |
|   |              | enet_299_299_40.47G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_tf2 | tf2_resnet50_ima    | 87.33         | 192.94        |
| 1 |              | genet_224_224_7.76G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_pt  | pt_resnet50_im      | 77.58         | 173.16        |
| 2 |              | agenet_224_224_8.2G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 90.97         | 179.67        |
| 3 | runed_0_3_pt | et_224_224_0.3_5.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 96.45         | 190.87        |
| 4 | runed_0_4_pt | et_224_224_0.4_4.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 105.13        | 203.14        |
| 5 | runed_0_5_pt | et_224_224_0.5_4.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 118.47        | 228.31        |
| 6 | runed_0_6_pt | et_224_224_0.6_3.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 129.01        | 251.31        |
| 7 | runed_0_7_pt | et_224_224_0.7_2.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | SA_          | pt_sa-gate          | 3.29          | 9.45          |
| 8 | gate_base_pt | _NYUv2_360_360_178G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | salsanext_pt | pt_sals             | 5.61          | 21.28         |
| 9 |              | anext_semantic-kitt |               |               |
|   |              | i_64_2048_0.6_20.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | sal          | pt_salsa            | 4.15          | 10.99         |
| 0 | sanext_v2_pt | nextv2_semantic-kit |               |               |
| 0 |              | ti_64_2048_0.75_32G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | semantic_s   | tf2_erfnet_city     | 7.44          | 23.88         |
| 0 | eg_citys_tf2 | scapes_512_1024_54G |               |               |
| 1 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | S            | pt_SemanticFPN_cit  | 35.49         | 162.79        |
| 0 | emanticFPN_c | yscapes_256_512_10G |               |               |
| 2 | ityscapes_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | Se           | pt_SemanticFP       | 10.52         | 52.46         |
| 0 | manticFPN_Mo | N-mobilenetv2_citys |               |               |
| 2 | bilenetv2_pt | capes_512_1024_5.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | SESR_S_pt    | pt_SESR-S_          | 88.30         | 140.63        |
| 0 |              | DIV2K_360_640_7.48G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | solo_pt      | pt_SOL              | 1.45          | 4.84          |
| 0 |              | O_coco_640_640_107G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | S            | pt_squeezenet_imag  | 575.41        | 1499.60       |
| 0 | queezeNet_pt | enet_224_224_351.7M |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_inceptio | tf_ssdinceptionv2   | 39.41         | 102.23        |
| 0 | n_v2_coco_tf | _coco_300_300_9.62G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_mobilene | tf_ssdmobilenetv1   | 111.06        | 332.23        |
| 0 | t_v1_coco_tf | _coco_300_300_2.47G |               |               |
| 8 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_mobilene | tf_ssdmobilenetv2   | 81.77         | 213.25        |
| 0 | t_v2_coco_tf | _coco_300_300_3.75G |               |               |
| 9 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | s            | tf                  | 2.92          | 5.17          |
| 1 | sd_resnet_50 | _ssdresnet50v1_fpn_ |               |               |
| 0 | _fpn_coco_tf | coco_640_640_178.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssdl         | tf                  | 106.06        | 304.19        |
| 1 | ite_mobilene | _ssdlite_mobilenetv |               |               |
| 1 | t_v2_coco_tf | 2_coco_300_300_1.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssr_pt       | pt_SSR              | 6.03          | 14.42         |
| 1 |              | _CVC_256_256_39.72G |               |               |
| 2 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | s            | tf_superpoint_      | 12.61         | 53.79         |
| 1 | uperpoint_tf | mixed_480_640_52.4G |               |               |
| 3 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | tex          | pt_textmountain_I   | 1.67          | 4.68          |
| 1 | tmountain_pt | CDAR_960_960_575.2G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | tsd_yolox_pt | pt_yolox            | 13.13         | 33.70         |
| 1 |              | _TT100K_640_640_73G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ultrafast_pt | pt_ultrafast_       | 35.08         | 95.82         |
| 1 |              | CULane_288_800_8.4G |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | unet         | pt_unet_cha         | 22.80         | 69.63         |
| 1 | _chaos-CT_pt | os-CT_512_512_23.3G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | chen_color   | pt_vehicle-co       | 204.70        | 499.15        |
| 1 | _resnet18_pt | lor-classification_ |               |               |
| 8 |              | color_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vehicle_make | pt_vehicle-make     | 203.44        | 498.51        |
| 1 | _resnet18_pt | -classification_Com |               |               |
| 9 |              | pCars_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vehicle_type | pt_vehicle-type     | 205.03        | 499.64        |
| 2 | _resnet18_pt | -classification_Com |               |               |
| 0 |              | pCars_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_tf    | tf_vgg16_imag       | 20.15         | 40.94         |
| 2 |              | enet_224_224_30.96G |               |               |
| 1 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_pr    | tf_vgg16_imagenet_  | 43.51         | 105.48        |
| 2 | uned_0_43_tf | 224_224_0.43_17.67G |               |               |
| 2 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_p     | tf_vgg16_imagenet   | 47.23         | 116.24        |
| 2 | runed_0_5_tf | _224_224_0.5_15.64G |               |               |
| 3 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_19_tf    | tf_vgg19_imag       | 17.37         | 36.47         |
| 2 |              | enet_224_224_39.28G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov3           | 13.55         | 35.12         |
| 2 | olov3_voc_tf | _voc_416_416_65.63G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | yol          | tf2_yolov3          | 13.24         | 34.87         |
| 2 | ov3_coco_tf2 | _coco_416_416_65.9G |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov4           | 13.52         | 33.96         |
| 2 | olov4_416_tf | _coco_416_416_60.3G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov4           | 10.25         | 25.24         |
| 2 | olov4_512_tf | _coco_512_512_91.2G |               |               |
| 8 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+

.. raw:: html

   </details>

Performance on ZCU104
~~~~~~~~~~~~~~~~~~~~~

Measured with Vitis AI 2.5 and Vitis AI Library 2.5

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``ZCU104`` board with a
``2 * B4096  @ 300MHz`` DPU configuration:

+---+--------------+---------------------+---------------+---------------+
| N | Model        | GPU Model Standard  | E2E           | E2E           |
| o |              | Name                | throughput    | throughput    |
| . |              |                     | (fps) Single  | (fps) Multi   |
|   |              |                     | Thread        | Thread        |
+===+==============+=====================+===============+===============+
| 1 | bcc_pt       | pt_BCC_shanghait    | 3.51          | 7.98          |
|   |              | ech_800_1000_268.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | c2d2_lite_pt | pt_C2D2lite         | 3.13          | 3.55          |
|   |              | _CC20_512_512_6.86G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | centerpoint  | pt_centerpoin       | 16.84         | 20.72         |
|   |              | t_astyx_2560_40_54G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | CLOCs        | pt_CLOCs_kitti_2.0  | 2.89          | 10.24         |
+---+--------------+---------------------+---------------+---------------+
| 5 | drunet_pt    | pt_DRUNet_          | 63.75         | 152.73        |
|   |              | Kvasir_528_608_0.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | effici       | tf_efficientdet-d2_ | 63.75         | 152.73        |
|   | entdet_d2_tf | coco_768_768_11.06G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | efficie      | tf2_                | 83.77         | 146.26        |
|   | ntnet-b0_tf2 | efficientnet-b0_ima |               |               |
|   |              | genet_224_224_0.36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | e            | tf_efficien         | 37.33         | 73.44         |
|   | fficientNet- | tnet-edgetpu-L_imag |               |               |
|   | edgetpu-L_tf | enet_300_300_19.36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | e            | tf_efficie          | 85.42         | 168.85        |
|   | fficientNet- | ntnet-edgetpu-M_ima |               |               |
|   | edgetpu-M_tf | genet_240_240_7.34G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | e            | tf_efficie          | 122.86        | 248.87        |
| 0 | fficientNet- | ntnet-edgetpu-S_ima |               |               |
|   | edgetpu-S_tf | genet_224_224_4.72G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ENet_c       | pt_ENet_citys       | 10.44         | 39.45         |
| 1 | ityscapes_pt | capes_512_1024_8.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | face_mask_   | pt_face-mask-dete   | 120.72        | 326.94        |
| 2 | detection_pt | ction_512_512_0.59G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | fac          | pt_face-q           | 3050.46       | 8471.64       |
| 3 | e-quality_pt | uality_80_60_61.68M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | f            | pt_facerec-resnet2  | 179.71        | 314.42        |
| 4 | acerec-resne | 0_mixed_112_96_3.5G |               |               |
|   | t20_mixed_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | facer        | pt_facere           | 992.52        | 2122.75       |
| 5 | eid-large_pt | id-large_96_96_515M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | facer        | pt_facer            | 2304.48       | 5927.11       |
| 6 | eid-small_pt | eid-small_80_80_90M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | fadnet       | pt_fadnet_sce       | 1.66          | 3.46          |
| 7 |              | neflow_576_960_441G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | f            | pt_fadnet_sceneflo  | 2.65          | 5.28          |
| 8 | adnet_pruned | w_576_960_0.65_154G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | FairMot_pt   | pt_FairMOT_mi       | 23.83         | 52.39         |
| 9 |              | xed_640_480_0.5_36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | FPN          | pt_                 | 39.44         | 81.02         |
| 0 | -resnet18_co | FPN-resnet18_covid1 |               |               |
|   | vid19-seg_pt | 9-seg_352_352_22.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Har          | pt_HardNet_m        | 26.42         | 49.36         |
| 1 | dNet_MSeg_pt | ixed_352_352_22.78G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | HFnet_tf     | tf_HFNet_m          | 3.21          | 14.39         |
| 2 |              | ixed_960_960_20.09G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inception_   | tf_inc              | 25.54         | 47.05         |
| 3 | resnet_v2_tf | eptionresnetv2_imag |               |               |
|   |              | enet_299_299_26.35G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inc          | tf_inceptionv1_     | 202.49        | 412.79        |
| 4 | eption_v1_tf | imagenet_224_224_3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | inc          | tf_inceptionv2_ima  | 98.99         | 195.15        |
| 5 | eption_v2_tf | genet_224_224_3.88G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inc          | tf_inceptionv3_imag | 63.43         | 121.49        |
| 6 | eption_v3_tf | enet_299_299_11.45G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | in           | tf                  | 72.73         | 141.54        |
| 7 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_2_tf | et_299_299_0.2_9.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | in           | tf                  | 90.22         | 180.10        |
| 8 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_4_tf | et_299_299_0.4_6.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Ince         | tf2_inceptionv3_ima | 62.84         | 121.59        |
| 9 | ption_v3_tf2 | genet_299_299_11.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | Inc          | pt_inceptionv3_ima  | 63.53         | 122.11        |
| 0 | eption_v3_pt | genet_299_299_11.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt_inceptionv3_imag | 80.35         | 157.27        |
| 1 | ception_v3_p | enet_299_299_0.3_8G |               |               |
|   | runed_0_3_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 90.53         | 180.34        |
| 2 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_4_pt | et_299_299_0.4_6.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 101.89        | 206.08        |
| 3 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_5_pt | et_299_299_0.5_5.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 120.56        | 248.18        |
| 4 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_6_pt | et_299_299_0.6_4.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | Inc          | tf_inceptionv4_imag | 30.72         | 59.03         |
| 5 | eption_v4_tf | enet_299_299_24.55G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | medical_     | tf2_2d-unet_n       | 163.55        | 338.55        |
| 6 | seg_cell_tf2 | uclei_128_128_5.31G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mlperf_ssd   | tf_mlperf_resnet34_ | 1.87          | 5.22          |
| 7 | _resnet34_tf | coco_1200_1200_433G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mlperf       | tf_                 | 84.21         | 157.65        |
| 8 | _resnet50_tf | mlperf_resnet50_ima |               |               |
|   |              | genet_224_224_8.19G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mobilenet_   | tf_m                | 278.38        | 604.87        |
| 9 | edge_0_75_tf | obilenetEdge0.75_im |               |               |
|   |              | agenet_224_224_624M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet    | tf_                 | 228.03        | 476.52        |
| 0 | _edge_1_0_tf | mobilenetEdge1.0_im |               |               |
|   |              | agenet_224_224_990M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet    | tf2_mobilenetv1_ima | 339.56        | 770.94        |
| 1 | _1_0_224_tf2 | genet_224_224_1.15G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v1 | tf                  | 1375.03       | 4263.73       |
| 2 | _0_25_128_tf | _mobilenetv1_0.25_i |               |               |
|   |              | magenet_128_128_27M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf                  | 950.02        | 2614.83       |
| 3 | 1_0_5_160_tf | _mobilenetv1_0.5_im |               |               |
|   |              | agenet_160_160_150M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_                 | 344.85        | 783.96        |
| 4 | 1_1_0_224_tf | mobilenetv1_1.0_ima |               |               |
|   |              | genet_224_224_1.14G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_mobil            | 351.57        | 775.26        |
| 5 | 1_1_0_224_pr | enetv1_1.0_imagenet |               |               |
|   | uned_0_11_tf | _224_224_0.11_1.02G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_mo               | 356.01        | 801.88        |
| 6 | 1_1_0_224_pr | bilenetv1_1.0_image |               |               |
|   | uned_0_12_tf | net_224_224_0.12_1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf                  | 284.18        | 609.18        |
| 7 | 2_1_0_224_tf | _mobilenetv2_1.0_im |               |               |
|   |              | agenet_224_224_602M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_                 | 203.20        | 413.59        |
| 8 | 2_1_4_224_tf | mobilenetv2_1.4_ima |               |               |
|   |              | genet_224_224_1.16G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mo           | tf_mo               | 1.81          | 5.53          |
| 9 | bilenet_v2_c | bilenetv2_cityscape |               |               |
|   | ityscapes_tf | s_1024_2048_132.74G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mo           | tf2_mobilenetv3_im  | 363.63        | 819.63        |
| 0 | bilenet_v3_s | agenet_224_224_132M |               |               |
|   | mall_1_0_tf2 |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mo           | pt_movene           | 95.38         | 362.92        |
| 1 | venet_ntd_pt | t_coco_192_192_0.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | MT-resne     | pt_MT-resnet18_m    | 34.31         | 93.07         |
| 2 | t18_mixed_pt | ixed_320_512_13.65G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mult         | pt_multitaskv3_m    | 17.76         | 55.32         |
| 3 | i_task_v3_pt | ixed_320_512_25.44G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ocr_pt       | pt_OCR_ICD          | 1.04          | 2.56          |
| 4 |              | AR2015_960_960_875G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_depthw   | pt_OFA-             | 107.8         | 343.72        |
| 5 | ise_res50_pt | depthwise-res50_ima |               |               |
|   |              | genet_176_176_2.49G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_rca      | pt_OFA-rcan_        | 17.93         | 28.87         |
| 6 | n_latency_pt | DIV2K_360_640_45.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_resnet50 | pt_OFA-resnet50_ima | 50.90         | 90.48         |
| 7 | _baseline_pt | genet_224_224_15.0G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa          | pt_O                | 77.23         | 138.28        |
| 8 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_45_pt | t_224_224_0.45_8.2G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa          | pt_O                | 95.29         | 163.25        |
| 9 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_60_pt | t_224_224_0.60_6.0G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa          | pt_O                | 138.44        | 248.57        |
| 0 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_74_pt | t_192_192_0.74_3.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_resn     | pt_O                | 198.24        | 354.63        |
| 1 | et50_0_9B_pt | FA-resnet50_imagene |               |               |
|   |              | t_160_160_0.88_1.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pt  | pt_OFA-yolo_        | 17.89         | 37.46         |
| 2 |              | coco_640_640_48.88G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pr  | pt_OFA-yolo_coco    | 22.75         | 48.69         |
| 3 | uned_0_30_pt | _640_640_0.3_34.72G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pr  | pt_OFA-yolo_coco    | 29.17         | 64.39         |
| 4 | uned_0_50_pt | _640_640_0.6_24.62G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | pers         | pt_person-orien     | 700.79        | 1369.03       |
| 5 | on-orientati | tation_224_112_558M |               |               |
|   | on_pruned_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | personr      | pt_pe               | 114.82        | 216.34        |
| 6 | eid-res50_pt | rsonreid-res50_mark |               |               |
|   |              | et1501_256_128_5.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 124.41        | 277.83        |
| 7 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_4_pt | 01_256_128_0.4_3.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 132.06        | 298.51        |
| 8 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_5_pt | 01_256_128_0.5_2.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 142.58        | 333.84        |
| 9 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_6_pt | 01_256_128_0.6_2.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | person       | pt_person           | 150.03        | 352.19        |
| 0 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_7_pt | 01_256_128_0.7_1.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | personr      | pt_p                | 395.46        | 700.55        |
| 1 | eid-res18_pt | ersonreid-res18_mar |               |               |
|   |              | ket1501_176_80_1.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pmg_pt       | pt_pmg              | 161.53        | 319.21        |
| 2 |              | _rp2k_224_224_2.28G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointpaint   | pt_pointpaintin     | 1.31          | 4.51          |
| 3 | ing_nuscenes | g_nuscenes_126G_2.5 |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointp       | pt_pointpillars_ki  | 20.13         | 49.83         |
| 4 | illars_kitti | tti_12000_100_10.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointpill    | pt_pointpillars_nus | 2.28          | 8.91          |
| 5 | ars_nuscenes | cenes_40000_64_108G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | rc           | tf_rcan_DIV2K_      | 9.17          | 17.15         |
| 6 | an_pruned_tf | 360_640_0.98_86.95G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | refi         | tf_refinede         | 10.8          | 25.87         |
| 7 | nedet_VOC_tf | t_VOC_320_320_81.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | RefineDet    | t                   | 12.82         | 26.09         |
| 8 | -Medical_EDD | f_RefineDet-Medical |               |               |
|   | _baseline_tf | _EDD_320_320_81.28G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | RefineDet-M  | tf_Re               | 23.90         | 50.30         |
| 9 | edical_EDD_p | fineDet-Medical_EDD |               |               |
|   | runed_0_5_tf | _320_320_0.5_41.42G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Ref              | 41.89         | 92.30         |
| 0 | dical_EDD_pr | ineDet-Medical_EDD_ |               |               |
|   | uned_0_75_tf | 320_320_0.75_20.54G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Ref              | 62.86         | 145.99        |
| 1 | dical_EDD_pr | ineDet-Medical_EDD_ |               |               |
|   | uned_0_85_tf | 320_320_0.85_12.32G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Re               | 71.3          | 169.42        |
| 2 | dical_EDD_tf | fineDet-Medical_EDD |               |               |
|   |              | _320_320_0.88_9.83G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_resnetv1_50_ima  | 93.84         | 175           |
| 3 | net_v1_50_tf | genet_224_224_6.97G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_                 | 117.80        | 209.94        |
| 4 | net_v1_50_pr | resnetv1_50_imagene |               |               |
|   | uned_0_38_tf | t_224_224_0.38_4.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_r                | 149.77        | 270.73        |
| 5 | net_v1_50_pr | esnetv1_50_imagenet |               |               |
|   | uned_0_65_tf | _224_224_0.65_2.45G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | tf_resnetv1_101_ima | 49.54         | 94.71         |
| 6 | et_v1_101_tf | genet_224_224_14.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | t                   | 33.81         | 64.91         |
| 7 | et_v1_152_tf | f_resnetv1_152_imag |               |               |
|   |              | enet_224_224_21.83G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_resnetv2_50_ima  | 48.10         | 90.57         |
| 8 | net_v2_50_tf | genet_299_299_13.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | t                   | 25.23         | 47.89         |
| 9 | et_v2_101_tf | f_resnetv2_101_imag |               |               |
|   |              | enet_299_299_26.78G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resn         | t                   | 17.18         | 32.61         |
| 0 | et_v2_152_tf | f_resnetv2_152_imag |               |               |
|   |              | enet_299_299_40.47G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_tf2 | tf2_resnet50_ima    | 93.23         | 175.39        |
| 1 |              | genet_224_224_7.76G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_pt  | pt_resnet50_im      | 83.56         | 157.2         |
| 2 |              | agenet_224_224_8.2G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 97.59         | 175.01        |
| 3 | runed_0_3_pt | et_224_224_0.3_5.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 103.34        | 185.46        |
| 4 | runed_0_4_pt | et_224_224_0.4_4.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 112.27        | 199.57        |
| 5 | runed_0_5_pt | et_224_224_0.5_4.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 126.82        | 225.88        |
| 6 | runed_0_6_pt | et_224_224_0.6_3.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 137.70        | 247.08        |
| 7 | runed_0_7_pt | et_224_224_0.7_2.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | SA_          | pt_sa-gate          | 3.46          | 8.52          |
| 8 | gate_base_pt | _NYUv2_360_360_178G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | salsanext_pt | pt_sals             | 5.68          | 21.45         |
| 9 |              | anext_semantic-kitt |               |               |
|   |              | i_64_2048_0.6_20.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | sal          | pt_salsa            | 4.30          | 11.74         |
| 0 | sanext_v2_pt | nextv2_semantic-kit |               |               |
| 0 |              | ti_64_2048_0.75_32G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | semantic_s   | tf2_erfnet_city     | 7.68          | 25.54         |
| 0 | eg_citys_tf2 | scapes_512_1024_54G |               |               |
| 1 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | S            | pt_SemanticFPN_cit  | 36.37         | 149.18        |
| 0 | emanticFPN_c | yscapes_256_512_10G |               |               |
| 2 | ityscapes_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | Se           | pt_SemanticFP       | 10.73         | 51.97         |
| 0 | manticFPN_Mo | N-mobilenetv2_citys |               |               |
| 3 | bilenetv2_pt | capes_512_1024_5.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | SESR_S_pt    | pt_SESR-S_          | 94.66         | 146.02        |
| 0 |              | DIV2K_360_640_7.48G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | solo_pt      | pt_SOL              | 1.47          | 4.82          |
| 0 |              | O_coco_640_640_107G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | S            | pt_squeezenet_imag  | 599.88        | 1315.93       |
| 0 | queezeNet_pt | enet_224_224_351.7M |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_inceptio | tf_ssdinceptionv2   | 41.64         | 87.86         |
| 0 | n_v2_coco_tf | _coco_300_300_9.62G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_mobilene | tf_ssdmobilenetv1   | 115.34        | 307.17        |
| 0 | t_v1_coco_tf | _coco_300_300_2.47G |               |               |
| 8 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_mobilene | tf_ssdmobilenetv2   | 85.56         | 196.25        |
| 0 | t_v2_coco_tf | _coco_300_300_3.75G |               |               |
| 9 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | s            | tf                  | 2.91          | 5.22          |
| 1 | sd_resnet_50 | _ssdresnet50v1_fpn_ |               |               |
| 0 | _fpn_coco_tf | coco_640_640_178.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssdl         | tf                  | 110.49        | 280.15        |
| 1 | ite_mobilene | _ssdlite_mobilenetv |               |               |
| 1 | t_v2_coco_tf | 2_coco_300_300_1.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssr_pt       | pt_SSR              | 6.46          | 12.37         |
| 1 |              | _CVC_256_256_39.72G |               |               |
| 2 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | s            | tf_superpoint_      | 11.11         | 40.32         |
| 1 | uperpoint_tf | mixed_480_640_52.4G |               |               |
| 3 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | tex          | pt_textmountain_I   | 1.77          | 3.68          |
| 1 | tmountain_pt | CDAR_960_960_575.2G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | tsd_yolox_pt | pt_yolox            | 13.99         | 27.78         |
| 1 |              | _TT100K_640_640_73G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ultrafast_pt | pt_ultrafast_       | 37.25         | 78.39         |
| 1 |              | CULane_288_800_8.4G |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | unet         | pt_unet_cha         | 23.63         | 59.35         |
| 1 | _chaos-CT_pt | os-CT_512_512_23.3G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | chen_color   | pt_vehicle-co       | 217.68        | 437.02        |
| 1 | _resnet18_pt | lor-classification_ |               |               |
| 8 |              | color_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vehicle_make | pt_vehicle-make     | 216.22        | 435.90        |
| 1 | _resnet18_pt | -classification_Com |               |               |
| 9 |              | pCars_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vehicle_type | pt_vehicle-type     | 218.02        | 437.32        |
| 2 | _resnet18_pt | -classification_Com |               |               |
| 0 |              | pCars_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_tf    | tf_vgg16_imag       | 21.49         | 37.10         |
| 2 |              | enet_224_224_30.96G |               |               |
| 1 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_pr    | tf_vgg16_imagenet_  | 46.41         | 88.40         |
| 2 | uned_0_43_tf | 224_224_0.43_17.67G |               |               |
| 2 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_p     | tf_vgg16_imagenet   | 50.44         | 96.87         |
| 2 | runed_0_5_tf | _224_224_0.5_15.64G |               |               |
| 3 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_19_tf    | tf_vgg19_imag       | 18.54         | 32.69         |
| 2 |              | enet_224_224_39.28G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov3           | 14.44         | 28.82         |
| 2 | olov3_voc_tf | _voc_416_416_65.63G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | yol          | tf2_yolov3          | 14.09         | 28.64         |
| 2 | ov3_coco_tf2 | _coco_416_416_65.9G |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov4           | 14.36         | 28.89         |
| 2 | olov4_416_tf | _coco_416_416_60.3G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov4           | 10.93         | 21.76         |
| 2 | olov4_512_tf | _coco_512_512_91.2G |               |               |
| 8 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+

.. raw:: html

   </details>

Performance on VCK190
~~~~~~~~~~~~~~~~~~~~~

Measured with Vitis AI 2.5 and Vitis AI Library 2.5

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``Versal`` board with 192
AIEs running at 1250 MHz:

+---+--------------+---------------------+---------------+---------------+
| N | Model        | GPU Model Standard  | E2E           | E2E           |
| o |              | Name                | throughput    | throughput    |
| . |              |                     | (fps) Single  | (fps) Multi   |
|   |              |                     | Thread        | Thread        |
+===+==============+=====================+===============+===============+
| 1 | bcc_pt       | pt_BCC_shanghait    | 37.01         | 72.51         |
|   |              | ech_800_1000_268.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | c2d2_lite_pt | pt_C2D2lite         | 20.57         | 26.49         |
|   |              | _CC20_512_512_6.86G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | centerpoint  | pt_centerpoin       | 128.12        | 236.99        |
|   |              | t_astyx_2560_40_54G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | CLOCs        | pt_CLOCs_kitti_2.0  | 8.11          | 14.63         |
+---+--------------+---------------------+---------------+---------------+
| 5 | drunet_pt    | pt_DRUNet_          | 256.13        | 459.87        |
|   |              | Kvasir_528_608_0.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | effici       | tf_efficientdet-d2_ | 18.1          | 40.38         |
|   | entdet_d2_tf | coco_768_768_11.06G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | efficie      | tf2_                | 1090.29       | 1811.8        |
|   | ntnet-b0_tf2 | efficientnet-b0_ima |               |               |
|   |              | genet_224_224_0.36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | e            | tf_efficien         | 397.35        | 513.32        |
|   | fficientNet- | tnet-edgetpu-L_imag |               |               |
|   | edgetpu-L_tf | enet_300_300_19.36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | e            | tf_efficie          | 889.46        | 1394.48       |
|   | fficientNet- | ntnet-edgetpu-M_ima |               |               |
|   | edgetpu-M_tf | genet_240_240_7.34G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | e            | tf_efficie          | 1209.7        | 2192.61       |
| 0 | fficientNet- | ntnet-edgetpu-S_ima |               |               |
|   | edgetpu-S_tf | genet_224_224_4.72G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ENet_c       | pt_ENet_citys       | 25.64         | 54.58         |
| 1 | ityscapes_pt | capes_512_1024_8.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | face_mask_   | pt_face-mask-dete   | 451.11        | 914.73        |
| 2 | detection_pt | ction_512_512_0.59G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | fac          | pt_face-q           | 13746.3       | 29425.4       |
| 3 | e-quality_pt | uality_80_60_61.68M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | f            | pt_facerec-resnet2  | 3849.25       | 5691.74       |
| 4 | acerec-resne | 0_mixed_112_96_3.5G |               |               |
|   | t20_mixed_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | facer        | pt_facere           | 7574.57       | 19419.9       |
| 5 | eid-large_pt | id-large_96_96_515M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | facer        | pt_facer            | 11369.1       | 26161.2       |
| 6 | eid-small_pt | eid-small_80_80_90M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | fadnet       | pt_fadnet_sce       | 8.02          | 13.46         |
| 7 |              | neflow_576_960_441G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | f            | pt_fadnet_sceneflo  | 9.01          | 16.24         |
| 8 | adnet_pruned | w_576_960_0.65_154G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | FairMot_pt   | pt_FairMOT_mi       | 194.87        | 374.48        |
| 9 |              | xed_640_480_0.5_36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | FPN          | pt_                 | 481.49        | 788.38        |
| 0 | -resnet18_co | FPN-resnet18_covid1 |               |               |
|   | vid19-seg_pt | 9-seg_352_352_22.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Har          | pt_HardNet_m        | 205.76        | 274.23        |
| 1 | dNet_MSeg_pt | ixed_352_352_22.78G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | HFnet_tf     | tf_HFNet_m          | 9.37          | 22.34         |
| 2 |              | ixed_960_960_20.09G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inception_   | tf_inc              | 387.99        | 499.77        |
| 3 | resnet_v2_tf | eptionresnetv2_imag |               |               |
|   |              | enet_299_299_26.35G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inc          | tf_inceptionv1_     | 1303.24       | 2458.23       |
| 4 | eption_v1_tf | imagenet_224_224_3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | inc          | tf_inceptionv2_ima  | 827.14        | 1190.76       |
| 5 | eption_v2_tf | genet_224_224_3.88G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inc          | tf_inceptionv3_imag | 595.41        | 903.47        |
| 6 | eption_v3_tf | enet_299_299_11.45G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | in           | tf                  | 629.90        | 985.47        |
| 7 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_2_tf | et_299_299_0.2_9.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | in           | tf                  | 687.25        | 1130.21       |
| 8 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_4_tf | et_299_299_0.4_6.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Ince         | tf2_inceptionv3_ima | 657.37        | 1061.04       |
| 9 | ption_v3_tf2 | genet_299_299_11.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | Inc          | pt_inceptionv3_ima  | 592.62        | 896.70        |
| 0 | eption_v3_pt | genet_299_299_11.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt_inceptionv3_imag | 653.31        | 1052.53       |
| 1 | ception_v3_p | enet_299_299_0.3_8G |               |               |
|   | runed_0_3_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 683.24        | 1125.47       |
| 2 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_4_pt | et_299_299_0.4_6.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 726.47        | 1242.57       |
| 3 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_5_pt | et_299_299_0.5_5.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 787.76        | 1430.62       |
| 4 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_6_pt | et_299_299_0.6_4.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | Inc          | tf_inceptionv4_imag | 341.84        | 424.03        |
| 5 | eption_v4_tf | enet_299_299_24.55G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | medical_     | tf2_2d-unet_n       | 1528.81       | 3034.36       |
| 6 | seg_cell_tf2 | uclei_128_128_5.31G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mlperf_ssd   | tf_mlperf_resnet34_ | 11.94         | 20.92         |
| 7 | _resnet34_tf | coco_1200_1200_433G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mlperf       | tf_                 | 1367.1        | 2744.24       |
| 8 | _resnet50_tf | mlperf_resnet50_ima |               |               |
|   |              | genet_224_224_8.19G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mobilenet_   | tf_m                | 1909.12       | 4995.3        |
| 9 | edge_0_75_tf | obilenetEdge0.75_im |               |               |
|   |              | agenet_224_224_624M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet    | tf_                 | 1839.19       | 4879.14       |
| 0 | _edge_1_0_tf | mobilenetEdge1.0_im |               |               |
|   |              | agenet_224_224_990M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet    | tf2_mobilenetv1_ima | 2026.52       | 5049.85       |
| 1 | _1_0_224_tf2 | genet_224_224_1.15G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v1 | tf                  | 5019.74       | 10348.4       |
| 2 | _0_25_128_tf | _mobilenetv1_0.25_i |               |               |
|   |              | magenet_128_128_27M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf                  | 3604.67       | 7997.22       |
| 3 | 1_0_5_160_tf | _mobilenetv1_0.5_im |               |               |
|   |              | agenet_160_160_150M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_                 | 2022.61       | 5015.41       |
| 4 | 1_1_0_224_tf | mobilenetv1_1.0_ima |               |               |
|   |              | genet_224_224_1.14G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_mobil            | 2034.76       | 5013.31       |
| 5 | 1_1_0_224_pr | enetv1_1.0_imagenet |               |               |
|   | uned_0_11_tf | _224_224_0.11_1.02G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_mo               | 2025.73       | 4957.45       |
| 6 | 1_1_0_224_pr | bilenetv1_1.0_image |               |               |
|   | uned_0_12_tf | net_224_224_0.12_1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf                  | 1891.80       | 4927.03       |
| 7 | 2_1_0_224_tf | _mobilenetv2_1.0_im |               |               |
|   |              | agenet_224_224_602M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_                 | 1640.06       | 4189.87       |
| 8 | 2_1_4_224_tf | mobilenetv2_1.4_ima |               |               |
|   |              | genet_224_224_1.16G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mo           | tf_mo               | 5.33          | 12.03         |
| 9 | bilenet_v2_c | bilenetv2_cityscape |               |               |
|   | ityscapes_tf | s_1024_2048_132.74G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mo           | tf2_mobilenetv3_im  | 2052.58       | 4977.33       |
| 0 | bilenet_v3_s | agenet_224_224_132M |               |               |
|   | mall_1_0_tf2 |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mo           | pt_movene           | 245.96        | 445.52        |
| 1 | venet_ntd_pt | t_coco_192_192_0.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | MT-resne     | pt_MT-resnet18_m    | 144.05        | 261.02        |
| 2 | t18_mixed_pt | ixed_320_512_13.65G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mult         | pt_multitaskv3_m    | 77.73         | 173.24        |
| 3 | i_task_v3_pt | ixed_320_512_25.44G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ocr_pt       | pt_OCR_ICD          | 8.67          | 18.67         |
| 4 |              | AR2015_960_960_875G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_depthw   | pt_OFA-             | 306.23        | 454.3         |
| 5 | ise_res50_pt | depthwise-res50_ima |               |               |
|   |              | genet_176_176_2.49G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_rca      | pt_OFA-rcan_        | 60.15         | 81.08         |
| 6 | n_latency_pt | DIV2K_360_640_45.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_resnet50 | pt_OFA-resnet50_ima | 845.17        | 1225.92       |
| 7 | _baseline_pt | genet_224_224_15.0G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa          | pt_O                | 1007.54       | 1602.95       |
| 8 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_45_pt | t_224_224_0.45_8.2G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa          | pt_O                | 1146.16       | 1982.49       |
| 9 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_60_pt | t_224_224_0.60_6.0G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa          | pt_O                | 1548.71       | 2848.63       |
| 0 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_74_pt | t_192_192_0.74_3.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_resn     | pt_O                | 2082.78       | 4027.03       |
| 1 | et50_0_9B_pt | FA-resnet50_imagene |               |               |
|   |              | t_160_160_0.88_1.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pt  | pt_OFA-yolo_        | 128.26        | 223.24        |
| 2 |              | coco_640_640_48.88G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pr  | pt_OFA-yolo_coco    | 146.2         | 255.8         |
| 3 | uned_0_30_pt | _640_640_0.3_34.72G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pr  | pt_OFA-yolo_coco    | 167.96        | 297.74        |
| 4 | uned_0_50_pt | _640_640_0.6_24.62G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | pers         | pt_person-orien     | 5617.05       | 12583.8       |
| 5 | on-orientati | tation_224_112_558M |               |               |
|   | on_pruned_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | personr      | pt_pe               | 1822.71       | 3816.15       |
| 6 | eid-res50_pt | rsonreid-res50_mark |               |               |
|   |              | et1501_256_128_5.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 1157.63       | 2238.36       |
| 7 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_4_pt | 01_256_128_0.4_3.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 1178.59       | 2239.07       |
| 8 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_5_pt | 01_256_128_0.5_2.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 1211.16       | 2234.36       |
| 9 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_6_pt | 01_256_128_0.6_2.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | person       | pt_person           | 1240.68       | 2228.85       |
| 0 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_7_pt | 01_256_128_0.7_1.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | personr      | pt_p                | 4173.59       | 8691.82       |
| 1 | eid-res18_pt | ersonreid-res18_mar |               |               |
|   |              | ket1501_176_80_1.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pmg_pt       | pt_pmg              | 1779.41       | 3635.07       |
| 2 |              | _rp2k_224_224_2.28G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointpaint   | pt_pointpaintin     | 3.8           | 6.75          |
| 3 | ing_nuscenes | g_nuscenes_126G_2.5 |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointp       | pt_pointpillars_ki  | 24.6          | 35.29         |
| 4 | illars_kitti | tti_12000_100_10.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointpill    | pt_pointpillars_nus | 7.65          | 15.97         |
| 5 | ars_nuscenes | cenes_40000_64_108G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | psmnet       | pt_psmnet_sceneflo  | 0.36          | 0.71          |
| 6 |              | w_576_960_0.68_696G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | rc           | tf_rcan_DIV2K_      | 45.29         | 56.99         |
| 7 | an_pruned_tf | 360_640_0.98_86.95G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | refi         | tf_refinede         | 101.95        | 224.59        |
| 8 | nedet_VOC_tf | t_VOC_320_320_81.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | RefineDet    | t                   | 232.24        | 315.20        |
| 9 | -Medical_EDD | f_RefineDet-Medical |               |               |
|   | _baseline_tf | _EDD_320_320_81.28G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-M  | tf_Re               | 342.68        | 563.85        |
| 0 | edical_EDD_p | fineDet-Medical_EDD |               |               |
|   | runed_0_5_tf | _320_320_0.5_41.42G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Ref              | 444.29        | 898.26        |
| 1 | dical_EDD_pr | ineDet-Medical_EDD_ |               |               |
|   | uned_0_75_tf | 320_320_0.75_20.54G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Ref              | 532.85        | 1259.42       |
| 2 | dical_EDD_pr | ineDet-Medical_EDD_ |               |               |
|   | uned_0_85_tf | 320_320_0.85_12.32G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Re               | 502.46        | 1282.02       |
| 3 | dical_EDD_tf | fineDet-Medical_EDD |               |               |
|   |              | _320_320_0.88_9.83G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_resnetv1_50_ima  | 1452.5        | 3054.88       |
| 4 | net_v1_50_tf | genet_224_224_6.97G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_                 | 1531.45       | 3239.42       |
| 5 | net_v1_50_pr | resnetv1_50_imagene |               |               |
|   | uned_0_38_tf | t_224_224_0.38_4.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_r                | 1744.88       | 4601.93       |
| 6 | net_v1_50_pr | esnetv1_50_imagenet |               |               |
|   | uned_0_65_tf | _224_224_0.65_2.45G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | tf_resnetv1_101_ima | 1063.41       | 1756.03       |
| 7 | et_v1_101_tf | genet_224_224_14.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | t                   | 838.21        | 1215.98       |
| 8 | et_v1_152_tf | f_resnetv1_152_imag |               |               |
|   |              | enet_224_224_21.83G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_resnetv2_50_ima  | 549.19        | 794.83        |
| 9 | net_v2_50_tf | genet_299_299_13.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resn         | t                   | 413.38        | 538.87        |
| 0 | et_v2_101_tf | f_resnetv2_101_imag |               |               |
|   |              | enet_299_299_26.78G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resn         | t                   | 331.27        | 407.21        |
| 1 | et_v2_152_tf | f_resnetv2_152_imag |               |               |
|   |              | enet_299_299_40.47G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_tf2 | tf2_resnet50_ima    | 1462.41       | 3095.02       |
| 2 |              | genet_224_224_7.76G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_pt  | pt_resnet50_im      | 1374.31       | 2773.11       |
| 3 |              | agenet_224_224_8.2G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 1414.68       | 2942.34       |
| 4 | runed_0_3_pt | et_224_224_0.3_5.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 1470.40       | 3173.66       |
| 5 | runed_0_4_pt | et_224_224_0.4_4.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 1533.74       | 3343.16       |
| 6 | runed_0_5_pt | et_224_224_0.5_4.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 1612.50       | 3883.51       |
| 7 | runed_0_6_pt | et_224_224_0.6_3.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 1698.97       | 4525.27       |
| 8 | runed_0_7_pt | et_224_224_0.7_2.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | SA_          | pt_sa-gate          | 7.61          | 9.62          |
| 9 | gate_base_pt | _NYUv2_360_360_178G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | salsanext_pt | pt_sals             | 12.27         | 24.06         |
| 0 |              | anext_semantic-kitt |               |               |
| 0 |              | i_64_2048_0.6_20.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | sal          | pt_salsa            | 10.61         | 22.3          |
| 0 | sanext_v2_pt | nextv2_semantic-kit |               |               |
| 1 |              | ti_64_2048_0.75_32G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | semantic_s   | tf2_erfnet_city     | 20.36         | 47.78         |
| 0 | eg_citys_tf2 | scapes_512_1024_54G |               |               |
| 2 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | S            | pt_SemanticFPN_cit  | 110.51        | 224.58        |
| 0 | emanticFPN_c | yscapes_256_512_10G |               |               |
| 3 | ityscapes_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | Se           | pt_SemanticFP       | 27.15         | 55.92         |
| 0 | manticFPN_Mo | N-mobilenetv2_citys |               |               |
| 4 | bilenetv2_pt | capes_512_1024_5.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | SESR_S_pt    | pt_SESR-S_          | 343.09        | 639.38        |
| 0 |              | DIV2K_360_640_7.48G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | solo_pt      | pt_SOL              | 4.36          | 7.47          |
| 0 |              | O_coco_640_640_107G |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | S            | pt_squeezenet_imag  | 3143.04       | 5797.62       |
| 0 | queezeNet_pt | enet_224_224_351.7M |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_inceptio | tf_ssdinceptionv2   | 273.59        | 381.22        |
| 0 | n_v2_coco_tf | _coco_300_300_9.62G |               |               |
| 8 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_mobilene | tf_ssdmobilenetv1   | 433.55        | 517.66        |
| 0 | t_v1_coco_tf | _coco_300_300_2.47G |               |               |
| 9 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_mobilene | tf_ssdmobilenetv2   | 387.64        | 508.52        |
| 1 | t_v2_coco_tf | _coco_300_300_3.75G |               |               |
| 0 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | s            | tf                  | 11.2          | 12.19         |
| 1 | sd_resnet_50 | _ssdresnet50v1_fpn_ |               |               |
| 1 | _fpn_coco_tf | coco_640_640_178.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssdl         | tf                  | 408.27        | 523.56        |
| 1 | ite_mobilene | _ssdlite_mobilenetv |               |               |
| 2 | t_v2_coco_tf | 2_coco_300_300_1.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssr_pt       | pt_SSR              | 74.48         | 78.02         |
| 1 |              | _CVC_256_256_39.72G |               |               |
| 3 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | s            | tf_superpoint_      | 49.86         | 106.38        |
| 1 | uperpoint_tf | mixed_480_640_52.4G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | tex          | pt_textmountain_I   | 20.92         | 29.6          |
| 1 | tmountain_pt | CDAR_960_960_575.2G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | tsd_yolox_pt | pt_yolox            | 132.05        | 193.27        |
| 1 |              | _TT100K_640_640_73G |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ultrafast_pt | pt_ultrafast_       | 392.04        | 962.59        |
| 1 |              | CULane_288_800_8.4G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | unet         | pt_unet_cha         | 80.52         | 242.28        |
| 1 | _chaos-CT_pt | os-CT_512_512_23.3G |               |               |
| 8 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | chen_color   | pt_vehicle-co       | 2036.82       | 5191.25       |
| 1 | _resnet18_pt | lor-classification_ |               |               |
| 9 |              | color_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vehicle_make | pt_vehicle-make     | 1986.11       | 5191.13       |
| 2 | _resnet18_pt | -classification_Com |               |               |
| 0 |              | pCars_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vehicle_type | pt_vehicle-type     | 2053.59       | 5251.65       |
| 2 | _resnet18_pt | -classification_Com |               |               |
| 1 |              | pCars_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_tf    | tf_vgg16_imag       | 530.06        | 655.44        |
| 2 |              | enet_224_224_30.96G |               |               |
| 2 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_pr    | tf_vgg16_imagenet_  | 863.92        | 1261.41       |
| 2 | uned_0_43_tf | 224_224_0.43_17.67G |               |               |
| 3 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_p     | tf_vgg16_imagenet   | 904.06        | 1355.58       |
| 2 | runed_0_5_tf | _224_224_0.5_15.64G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_19_tf    | tf_vgg19_imag       | 480.75        | 582.57        |
| 2 |              | enet_224_224_39.28G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov3           | 192.68        | 292.35        |
| 2 | olov3_voc_tf | _voc_416_416_65.63G |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | yol          | tf2_yolov3          | 218.53        | 291.4         |
| 2 | ov3_coco_tf2 | _coco_416_416_65.9G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov4           | 141.81        | 214.67        |
| 2 | olov4_416_tf | _coco_416_416_60.3G |               |               |
| 8 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov4           | 105.09        | 154.65        |
| 2 | olov4_512_tf | _coco_512_512_91.2G |               |               |
| 9 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+

.. raw:: html

   </details>

Performance on Kria KV260 SOM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measured with Vitis AI 2.5 and Vitis AI Library 2.5

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``Kria KV260`` board with a
``1 * B4096F  @ 300MHz`` DPU configuration:

+---+--------------+---------------------+---------------+---------------+
| N | Model        | GPU Model Standard  | E2E           | E2E           |
| o |              | Name                | throughput    | throughput    |
| . |              |                     | (fps) Single  | (fps) Multi   |
|   |              |                     | Thread        | Thread        |
+===+==============+=====================+===============+===============+
| 1 | bcc_pt       | pt_BCC_shanghait    | 3.55          | 4.01          |
|   |              | ech_800_1000_268.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | c2d2_lite_pt | pt_C2D2lite         | 3.24          | 3.46          |
|   |              | _CC20_512_512_6.86G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | centerpoint  | pt_centerpoin       | 17.21         | 20.2          |
|   |              | t_astyx_2560_40_54G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | CLOCs        | pt_CLOCs_kitti_2.0  | 3.08          | 9.05          |
+---+--------------+---------------------+---------------+---------------+
| 5 | drunet_pt    | pt_DRUNet_          | 65.17         | 78.01         |
|   |              | Kvasir_528_608_0.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | effici       | tf_efficientdet-d2_ | 3.29          | 4.24          |
|   | entdet_d2_tf | coco_768_768_11.06G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | efficie      | tf2_                | 84.26         | 87.73         |
|   | ntnet-b0_tf2 | efficientnet-b0_ima |               |               |
|   |              | genet_224_224_0.36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | e            | tf_efficien         | 37.52         | 38.64         |
|   | fficientNet- | tnet-edgetpu-L_imag |               |               |
|   | edgetpu-L_tf | enet_300_300_19.36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | e            | tf_efficie          | 86.03         | 90.05         |
|   | fficientNet- | ntnet-edgetpu-M_ima |               |               |
|   | edgetpu-M_tf | genet_240_240_7.34G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | e            | tf_efficie          | 124.06        | 131.85        |
| 0 | fficientNet- | ntnet-edgetpu-S_ima |               |               |
|   | edgetpu-S_tf | genet_224_224_4.72G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ENet_c       | pt_ENet_citys       | 11.18         | 30.00         |
| 1 | ityscapes_pt | capes_512_1024_8.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | face_mask_   | pt_face-mask-dete   | 124.23        | 170.38        |
| 2 | detection_pt | ction_512_512_0.59G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | fac          | pt_face-q           | 3316.38       | 5494.16       |
| 3 | e-quality_pt | uality_80_60_61.68M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | f            | pt_facerec-resnet2  | 179.9         | 184.56        |
| 4 | acerec-resne | 0_mixed_112_96_3.5G |               |               |
|   | t20_mixed_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | facer        | pt_facere           | 1021.38       | 1134.19       |
| 5 | eid-large_pt | id-large_96_96_515M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | facer        | pt_facer            | 2472.62       | 3560.3        |
| 6 | eid-small_pt | eid-small_80_80_90M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | fadnet       | pt_fadnet_sce       | 1.67          | 2.26          |
| 7 |              | neflow_576_960_441G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | f            | pt_fadnet_sceneflo  | 2.78          | 4.29          |
| 8 | adnet_pruned | w_576_960_0.65_154G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | FairMot_pt   | pt_FairMOT_mi       | 24.17         | 27.04         |
| 9 |              | xed_640_480_0.5_36G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | FPN          | pt_                 | 39.64         | 41.25         |
| 0 | -resnet18_co | FPN-resnet18_covid1 |               |               |
|   | vid19-seg_pt | 9-seg_352_352_22.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Har          | pt_HardNet_m        | 26.66         | 27.63         |
| 1 | dNet_MSeg_pt | ixed_352_352_22.78G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | HFnet_tf     | tf_HFNet_m          | 3.39          | 11.24         |
| 2 |              | ixed_960_960_20.09G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inception_   | tf_inc              | 25.65         | 26.19         |
| 3 | resnet_v2_tf | eptionresnetv2_imag |               |               |
|   |              | enet_299_299_26.35G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inc          | tf_inceptionv1_     | 205.17        | 227.17        |
| 4 | eption_v1_tf | imagenet_224_224_3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | inc          | tf_inceptionv2_ima  | 99.05         | 104.45        |
| 5 | eption_v2_tf | genet_224_224_3.88G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Inc          | tf_inceptionv3_imag | 63.91         | 67.38         |
| 6 | eption_v3_tf | enet_299_299_11.45G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | in           | tf                  | 73.34         | 77.93         |
| 7 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_2_tf | et_299_299_0.2_9.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | in           | tf                  | 91.19         | 98.33         |
| 8 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_4_tf | et_299_299_0.4_6.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 2 | Ince         | tf2_inceptionv3_ima | 63.34         | 66.68         |
| 9 | ption_v3_tf2 | genet_299_299_11.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | Inc          | pt_inceptionv3_ima  | 64.03         | 67.46         |
| 0 | eption_v3_pt | genet_299_299_11.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt_inceptionv3_imag | 80.72         | 86.79         |
| 1 | ception_v3_p | enet_299_299_0.3_8G |               |               |
|   | runed_0_3_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 91.55         | 98.71         |
| 2 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_4_pt | et_299_299_0.4_6.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 103.26        | 112.6         |
| 3 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_5_pt | et_299_299_0.5_5.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | in           | pt                  | 121.92        | 135.54        |
| 4 | ception_v3_p | _inceptionv3_imagen |               |               |
|   | runed_0_6_pt | et_299_299_0.6_4.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | Inc          | tf_inceptionv4_imag | 30.83         | 31.6          |
| 5 | eption_v4_tf | enet_299_299_24.55G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | medical_     | tf2_2d-unet_n       | 165.01        | 179.4         |
| 6 | seg_cell_tf2 | uclei_128_128_5.31G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mlperf_ssd   | tf_mlperf_resnet34_ | 1.91          | 2.64          |
| 7 | _resnet34_tf | coco_1200_1200_433G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mlperf       | tf_                 | 84.83         | 88.69         |
| 8 | _resnet50_tf | mlperf_resnet50_ima |               |               |
|   |              | genet_224_224_8.19G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 3 | mobilenet_   | tf_m                | 282.1         | 328.78        |
| 9 | edge_0_75_tf | obilenetEdge0.75_im |               |               |
|   |              | agenet_224_224_624M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet    | tf_                 | 232.17        | 260.7         |
| 0 | _edge_1_0_tf | mobilenetEdge1.0_im |               |               |
|   |              | agenet_224_224_990M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet    | tf2_mobilenetv1_ima | 347.97        | 416.6         |
| 1 | _1_0_224_tf2 | genet_224_224_1.15G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v1 | tf                  | 1470.42       | 2428.88       |
| 2 | _0_25_128_tf | _mobilenetv1_0.25_i |               |               |
|   |              | magenet_128_128_27M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf                  | 1002.62       | 1492.12       |
| 3 | 1_0_5_160_tf | _mobilenetv1_0.5_im |               |               |
|   |              | agenet_160_160_150M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_                 | 354.19        | 425.57        |
| 4 | 1_1_0_224_tf | mobilenetv1_1.0_ima |               |               |
|   |              | genet_224_224_1.14G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_mobil            | 361.27        | 437.31        |
| 5 | 1_1_0_224_pr | enetv1_1.0_imagenet |               |               |
|   | uned_0_11_tf | _224_224_0.11_1.02G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_mo               | 364.45        | 441.98        |
| 6 | 1_1_0_224_pr | bilenetv1_1.0_image |               |               |
|   | uned_0_12_tf | net_224_224_0.12_1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf                  | 290.62        | 336.82        |
| 7 | 2_1_0_224_tf | _mobilenetv2_1.0_im |               |               |
|   |              | agenet_224_224_602M |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mobilenet_v  | tf_                 | 206.04        | 228.2         |
| 8 | 2_1_4_224_tf | mobilenetv2_1.4_ima |               |               |
|   |              | genet_224_224_1.16G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 4 | mo           | tf_mo               | 1.90          | 3.32          |
| 9 | bilenet_v2_c | bilenetv2_cityscape |               |               |
|   | ityscapes_tf | s_1024_2048_132.74G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mo           | tf2_mobilenetv3_im  | 374.47        | 453.97        |
| 0 | bilenet_v3_s | agenet_224_224_132M |               |               |
|   | mall_1_0_tf2 |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mo           | pt_movene           | 102.8         | 351.28        |
| 1 | venet_ntd_pt | t_coco_192_192_0.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | MT-resne     | pt_MT-resnet18_m    | 35.45         | 50.52         |
| 2 | t18_mixed_pt | ixed_320_512_13.65G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | mult         | pt_multitaskv3_m    | 18.57         | 29.95         |
| 3 | i_task_v3_pt | ixed_320_512_25.44G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ocr_pt       | pt_OCR_ICD          | 1.06          | 1.29          |
| 4 |              | AR2015_960_960_875G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_depthw   | pt_OFA-             | 116.42        | 264.3         |
| 5 | ise_res50_pt | depthwise-res50_ima |               |               |
|   |              | genet_176_176_2.49G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_rca      | pt_OFA-rcan_        | 18.10         | 19.27         |
| 6 | n_latency_pt | DIV2K_360_640_45.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa_resnet50 | pt_OFA-resnet50_ima | 51.2          | 52.49         |
| 7 | _baseline_pt | genet_224_224_15.0G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa          | pt_O                | 77.6          | 80.94         |
| 8 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_45_pt | t_224_224_0.45_8.2G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 5 | ofa          | pt_O                | 95.68         | 100.35        |
| 9 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_60_pt | t_224_224_0.60_6.0G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa          | pt_O                | 139.85        | 148.19        |
| 0 | _resnet50_pr | FA-resnet50_imagene |               |               |
|   | uned_0_74_pt | t_192_192_0.74_3.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_resn     | pt_O                | 199.09        | 214.92        |
| 1 | et50_0_9B_pt | FA-resnet50_imagene |               |               |
|   |              | t_160_160_0.88_1.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pt  | pt_OFA-yolo_        | 17.95         | 18.89         |
| 2 |              | coco_640_640_48.88G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pr  | pt_OFA-yolo_coco    | 22.98         | 26.44         |
| 3 | uned_0_30_pt | _640_640_0.3_34.72G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | ofa_yolo_pr  | pt_OFA-yolo_coco    | 29.19         | 35.48         |
| 4 | uned_0_50_pt | _640_640_0.6_24.62G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | pers         | pt_person-orien     | 712.72        | 804.15        |
| 5 | on-orientati | tation_224_112_558M |               |               |
|   | on_pruned_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | personr      | pt_pe               | 115.82        | 122.43        |
| 6 | eid-res50_pt | rsonreid-res50_mark |               |               |
|   |              | et1501_256_128_5.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 127.16        | 159.8         |
| 7 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_4_pt | 01_256_128_0.4_3.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 135.88        | 173.97        |
| 8 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_5_pt | 01_256_128_0.5_2.7G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 6 | person       | pt_person           | 145.46        | 190.93        |
| 9 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_6_pt | 01_256_128_0.6_2.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | person       | pt_person           | 154.51        | 206.66        |
| 0 | reid_res50_p | reid-res50_market15 |               |               |
|   | runed_0_7_pt | 01_256_128_0.7_1.6G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | personr      | pt_p                | 399.80        | 436.59        |
| 1 | eid-res18_pt | ersonreid-res18_mar |               |               |
|   |              | ket1501_176_80_1.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pmg_pt       | pt_pmg              | 162.79        | 173.08        |
| 2 |              | _rp2k_224_224_2.28G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointpaint   | pt_pointpaintin     | 1.34          | 2.78          |
| 3 | ing_nuscenes | g_nuscenes_126G_2.5 |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointp       | pt_pointpillars_ki  | 20.81         | 32.24         |
| 4 | illars_kitti | tti_12000_100_10.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | pointpill    | pt_pointpillars_nus | 2.30          | 5.46          |
| 5 | ars_nuscenes | cenes_40000_64_108G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | rc           | tf_rcan_DIV2K_      | 9.21          | 9.5           |
| 6 | an_pruned_tf | 360_640_0.98_86.95G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | refi         | tf_refinede         | 10.90         | 13.11         |
| 7 | nedet_VOC_tf | t_VOC_320_320_81.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | RefineDet    | t                   | 12.86         | 13.21         |
| 8 | -Medical_EDD | f_RefineDet-Medical |               |               |
|   | _baseline_tf | _EDD_320_320_81.28G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 7 | RefineDet-M  | tf_Re               | 24.04         | 25.31         |
| 9 | edical_EDD_p | fineDet-Medical_EDD |               |               |
|   | runed_0_5_tf | _320_320_0.5_41.42G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Ref              | 42.23         | 46.49         |
| 0 | dical_EDD_pr | ineDet-Medical_EDD_ |               |               |
|   | uned_0_75_tf | 320_320_0.75_20.54G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Ref              | 63.87         | 73.62         |
| 1 | dical_EDD_pr | ineDet-Medical_EDD_ |               |               |
|   | uned_0_85_tf | 320_320_0.85_12.32G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | RefineDet-Me | tf_Re               | 71.75         | 85.60         |
| 2 | dical_EDD_tf | fineDet-Medical_EDD |               |               |
|   |              | _320_320_0.88_9.83G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_resnetv1_50_ima  | 94.21         | 99.01         |
| 3 | net_v1_50_tf | genet_224_224_6.97G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_                 | 119.45        | 126.55        |
| 4 | net_v1_50_pr | resnetv1_50_imagene |               |               |
|   | uned_0_38_tf | t_224_224_0.38_4.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_r                | 151.27        | 162.96        |
| 5 | net_v1_50_pr | esnetv1_50_imagenet |               |               |
|   | uned_0_65_tf | _224_224_0.65_2.45G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | tf_resnetv1_101_ima | 49.86         | 51.07         |
| 6 | et_v1_101_tf | genet_224_224_14.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | t                   | 33.98         | 34.54         |
| 7 | et_v1_152_tf | f_resnetv1_152_imag |               |               |
|   |              | enet_224_224_21.83G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | res          | tf_resnetv2_50_ima  | 48.42         | 50.46         |
| 8 | net_v2_50_tf | genet_299_299_13.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 8 | resn         | t                   | 25.09         | 25.87         |
| 9 | et_v2_101_tf | f_resnetv2_101_imag |               |               |
|   |              | enet_299_299_26.78G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resn         | t                   | 17.25         | 17.48         |
| 0 | et_v2_152_tf | f_resnetv2_152_imag |               |               |
|   |              | enet_299_299_40.47G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_tf2 | tf2_resnet50_ima    | 93.80         | 98.21         |
| 1 |              | genet_224_224_7.76G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_pt  | pt_resnet50_im      | 84.44         | 87.96         |
| 2 |              | agenet_224_224_8.2G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 98.92         | 103.79        |
| 3 | runed_0_3_pt | et_224_224_0.3_5.8G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 104.15        | 109.58        |
| 4 | runed_0_4_pt | et_224_224_0.4_4.9G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 113.71        | 120.24        |
| 5 | runed_0_5_pt | et_224_224_0.5_4.1G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 127.95        | 136.11        |
| 6 | runed_0_6_pt | et_224_224_0.6_3.3G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | resnet50_p   | pt_resnet50_imagen  | 139.55        | 149.48        |
| 7 | runed_0_7_pt | et_224_224_0.7_2.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | SA_          | pt_sa-gate          | 3.55          | 4.64          |
| 8 | gate_base_pt | _NYUv2_360_360_178G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 9 | salsanext_pt | pt_sals             | 6.11          | 19.55         |
| 9 |              | anext_semantic-kitt |               |               |
|   |              | i_64_2048_0.6_20.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | sal          | pt_salsa            | 4.55          | 11.46         |
| 0 | sanext_v2_pt | nextv2_semantic-kit |               |               |
| 0 |              | ti_64_2048_0.75_32G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | semantic_s   | tf2_erfnet_city     | 8.11          | 15.38         |
| 0 | eg_citys_tf2 | scapes_512_1024_54G |               |               |
| 1 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | S            | pt_SemanticFPN_cit  | 37.36         | 86.73         |
| 0 | emanticFPN_c | yscapes_256_512_10G |               |               |
| 2 | ityscapes_pt |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | Se           | pt_SemanticFP       | 11.52         | 33.01         |
| 0 | manticFPN_Mo | N-mobilenetv2_citys |               |               |
| 3 | bilenetv2_pt | capes_512_1024_5.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | SESR_S_pt    | pt_SESR-S_          | 95.84         | 105.55        |
| 0 |              | DIV2K_360_640_7.48G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | solo_pt      | pt_SOL              | 1.46          | 4.44          |
| 0 |              | O_coco_640_640_107G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | S            | pt_squeezenet_imag  | 619.72        | 762.62        |
| 0 | queezeNet_pt | enet_224_224_351.7M |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_inceptio | tf_ssdinceptionv2   | 42.06         | 47.23         |
| 0 | n_v2_coco_tf | _coco_300_300_9.62G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_mobilene | tf_ssdmobilenetv1   | 119.11        | 172.23        |
| 0 | t_v1_coco_tf | _coco_300_300_2.47G |               |               |
| 8 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssd_mobilene | tf_ssdmobilenetv2   | 87.71         | 112.72        |
| 0 | t_v2_coco_tf | _coco_300_300_3.75G |               |               |
| 9 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | s            | tf                  | 2.96          | 5.26          |
| 1 | sd_resnet_50 | _ssdresnet50v1_fpn_ |               |               |
| 0 | _fpn_coco_tf | coco_640_640_178.4G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssdl         | tf                  | 111.50        | 161.41        |
| 1 | ite_mobilene | _ssdlite_mobilenetv |               |               |
| 1 | t_v2_coco_tf | 2_coco_300_300_1.5G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ssr_pt       | pt_SSR              | 6.47          | 6.50          |
| 1 |              | _CVC_256_256_39.72G |               |               |
| 2 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | s            | tf_superpoint_      | 11.46         | 20.57         |
| 1 | uperpoint_tf | mixed_480_640_52.4G |               |               |
| 3 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | tex          | pt_textmountain_I   | 1.78          | 1.89          |
| 1 | tmountain_pt | CDAR_960_960_575.2G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | tsd_yolox_pt | pt_yolox            | 13.99         | 14.62         |
| 1 |              | _TT100K_640_640_73G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | ultrafast_pt | pt_ultrafast_       | 37.58         | 41.32         |
| 1 |              | CULane_288_800_8.4G |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | unet         | pt_unet_cha         | 23.88         | 30.82         |
| 1 | _chaos-CT_pt | os-CT_512_512_23.3G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | chen_color   | pt_vehicle-co       | 219.52        | 239.61        |
| 1 | _resnet18_pt | lor-classification_ |               |               |
| 8 |              | color_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vehicle_make | pt_vehicle-make     | 217.70        | 239.21        |
| 1 | _resnet18_pt | -classification_Com |               |               |
| 9 |              | pCars_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vehicle_type | pt_vehicle-type     | 220.38        | 239.78        |
| 2 | _resnet18_pt | -classification_Com |               |               |
| 0 |              | pCars_224_224_3.63G |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_tf    | tf_vgg16_imag       | 21.50         | 21.72         |
| 2 |              | enet_224_224_30.96G |               |               |
| 1 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_pr    | tf_vgg16_imagenet_  | 46.43         | 47.56         |
| 2 | uned_0_43_tf | 224_224_0.43_17.67G |               |               |
| 2 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_16_p     | tf_vgg16_imagenet   | 50.52         | 51.76         |
| 2 | runed_0_5_tf | _224_224_0.5_15.64G |               |               |
| 3 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | vgg_19_tf    | tf_vgg19_imag       | 18.55         | 18.71         |
| 2 |              | enet_224_224_39.28G |               |               |
| 4 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov3           | 14.48         | 14.86         |
| 2 | olov3_voc_tf | _voc_416_416_65.63G |               |               |
| 5 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | yol          | tf2_yolov3          | 14.09         | 14.76         |
| 2 | ov3_coco_tf2 | _coco_416_416_65.9G |               |               |
| 6 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov4           | 14.43         | 15.35         |
| 2 | olov4_416_tf | _coco_416_416_60.3G |               |               |
| 7 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+
| 1 | y            | tf_yolov4           | 10.98         | 11.65         |
| 2 | olov4_512_tf | _coco_512_512_91.2G |               |               |
| 8 |              |                     |               |               |
+---+--------------+---------------------+---------------+---------------+

.. raw:: html

   </details>

Performance on VCK5000
~~~~~~~~~~~~~~~~~~~~~~

Measured with Vitis AI 2.5 and Vitis AI Library 2.5

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput for models on the ``Versal ACAP VCK5000`` board with
DPUCVDX8H running at 4PE@350 MHz in Gen3x16:

+---+----------------+------------------------+-------+-----------------+
| N | Model          | GPU Model Standard     | DPU   | E2E throughput  |
| o |                | Name                   | Freq  | (fps) Multi     |
| . |                |                        | uency | Thread          |
|   |                |                        | (MHz) |                 |
+===+================+========================+=======+=================+
| 1 | bcc_pt         | pt_BCC_shangh          | 350   | 49.02           |
|   |                | aitech_800_1000_268.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | c2d2_lite_pt   | pt_C2D2l               | 350   | 22.48           |
|   |                | ite_CC20_512_512_6.86G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | centerpoint    | pt_centerp             | 350   | 146.98          |
|   |                | oint_astyx_2560_40_54G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | CLOCs          | pt_CLOCs_kitti_2.0     | 350   | 11.71           |
+---+----------------+------------------------+-------+-----------------+
| 5 | drunet_pt      | pt_DRUN                | 350   | 94.14           |
|   |                | et_Kvasir_528_608_0.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | effic          | tf2_efficientnet-b0_   | 350   | 426.84          |
|   | ientnet-b0_tf2 | imagenet_224_224_0.36G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | efficientNe    | tf_ef                  | 350   | 393.64          |
|   | t-edgetpu-L_tf | ficientnet-edgetpu-L_i |       |                 |
|   |                | magenet_300_300_19.36G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | efficientNe    | tf_e                   | 350   | 974.60          |
|   | t-edgetpu-M_tf | fficientnet-edgetpu-M_ |       |                 |
|   |                | imagenet_240_240_7.34G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | efficientNe    | tf_e                   | 350   | 1663.81         |
|   | t-edgetpu-S_tf | fficientnet-edgetpu-S_ |       |                 |
|   |                | imagenet_224_224_4.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ENet           | pt_ENet_ci             | 350   | 67.35           |
| 0 | _cityscapes_pt | tyscapes_512_1024_8.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | face_mas       | pt_face-mask-d         | 350   | 770.84          |
| 1 | k_detection_pt | etection_512_512_0.59G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | f              | pt_fac                 | 350   | 16649.70        |
| 2 | ace-quality_pt | e-quality_80_60_61.68M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | facerec-res    | pt_facerec-resn        | 350   | 2954.23         |
| 3 | net20_mixed_pt | et20_mixed_112_96_3.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fac            | pt_fac                 | 350   | 11815.20        |
| 4 | ereid-large_pt | ereid-large_96_96_515M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fac            | pt_fa                  | 350   | 17826.50        |
| 5 | ereid-small_pt | cereid-small_80_80_90M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fadnet         | pt_fadnet_             | 350   | 9.60            |
| 6 |                | sceneflow_576_960_441G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fadnet_pruned  | pt_fadnet_scene        | 350   | 10.05           |
| 7 |                | flow_576_960_0.65_154G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FairMot_pt     | pt_FairMOT             | 350   | 277.87          |
| 8 |                | _mixed_640_480_0.5_36G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FPN-resnet18_  | pt_FPN-resnet18_cov    | 350   | 612.50          |
| 9 | covid19-seg_pt | id19-seg_352_352_22.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | H              | pt_HardNe              | 350   | 158.09          |
| 0 | ardNet_MSeg_pt | t_mixed_352_352_22.78G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | Inceptio       | tf_inceptionresnetv2_i | 350   | 307.80          |
| 1 | n_resnet_v2_tf | magenet_299_299_26.35G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | tf_inception           | 350   | 1585.16         |
| 2 | nception_v1_tf | v1_imagenet_224_224_3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | i              | tf_inceptionv2_        | 350   | 210.04          |
| 3 | nception_v2_tf | imagenet_224_224_3.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | tf_inceptionv3_i       | 350   | 496.16          |
| 4 | nception_v3_tf | magenet_299_299_11.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | tf_inceptionv3_ima     | 350   | 541.24          |
| 5 | _pruned_0_2_tf | genet_299_299_0.2_9.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | tf_inceptionv3_ima     | 350   | 591.99          |
| 6 | _pruned_0_4_tf | genet_299_299_0.4_6.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | In             | tf2_inceptionv3_       | 350   | 490.31          |
| 7 | ception_v3_tf2 | imagenet_299_299_11.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | pt_inceptionv3_        | 350   | 494.73          |
| 8 | nception_v3_pt | imagenet_299_299_11.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_i       | 350   | 564.87          |
| 9 | _pruned_0_3_pt | magenet_299_299_0.3_8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | inception_v3   | pt_inceptionv3_ima     | 350   | 586.50          |
| 0 | _pruned_0_4_pt | genet_299_299_0.4_6.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | inception_v3   | pt_inceptionv3_ima     | 350   | 667.43          |
| 1 | _pruned_0_5_pt | genet_299_299_0.5_5.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | inception_v3   | pt_inceptionv3_ima     | 350   | 749.79          |
| 2 | _pruned_0_6_pt | genet_299_299_0.6_4.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | I              | tf_inceptionv4_i       | 350   | 280.19          |
| 3 | nception_v4_tf | magenet_299_299_24.55G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | medica         | tf2_2d-une             | 350   | 914.61          |
| 4 | l_seg_cell_tf2 | t_nuclei_128_128_5.31G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mlperf_s       | tf_mlperf_resnet       | 350   | 49.42           |
| 5 | sd_resnet34_tf | 34_coco_1200_1200_433G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mlpe           | tf_mlperf_resnet50_    | 350   | 1638.84         |
| 6 | rf_resnet50_tf | imagenet_224_224_8.19G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilene       | tf_mobilenetEdge0.75   | 350   | 3344.23         |
| 7 | t_edge_0_75_tf | _imagenet_224_224_624M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf_mobilenetEdge1.0    | 350   | 3101.21         |
| 8 | et_edge_1_0_tf | _imagenet_224_224_990M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf2_mobilenetv1_       | 350   | 5864.58         |
| 9 | et_1_0_224_tf2 | imagenet_224_224_1.15G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet_     | tf_mobilenetv1_0.2     | 350   | 19353.80        |
| 0 | v1_0_25_128_tf | 5_imagenet_128_128_27M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet      | tf_mobilenetv1_0.5     | 350   | 13349.20        |
| 1 | _v1_0_5_160_tf | _imagenet_160_160_150M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet      | tf_mobilenetv1_1.0_    | 350   | 6561.03         |
| 2 | _v1_1_0_224_tf | imagenet_224_224_1.14G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilen        | tf                     | 350   | 6959.29         |
| 3 | et_v1_1_0_224_ | _mobilenetv1_1.0_image |       |                 |
|   | pruned_0_11_tf | net_224_224_0.11_1.02G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilen        | tf_mobilenetv1_1.0_im  | 350   | 6934.40         |
| 4 | et_v1_1_0_224_ | agenet_224_224_0.12_1G |       |                 |
|   | pruned_0_12_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet      | tf_mobilenetv2_1.0     | 350   | 4058.70         |
| 5 | _v2_1_0_224_tf | _imagenet_224_224_602M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet      | tf_mobilenetv2_1.4_    | 350   | 3137.54         |
| 6 | _v2_1_4_224_tf | imagenet_224_224_1.16G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet_v2   | tf_mobilenetv2_citysc  | 350   | 16.29           |
| 7 | _cityscapes_tf | apes_1024_2048_132.74G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet_v3   | tf2_mobilenetv3        | 350   | 1963.18         |
| 8 | _small_1_0_tf2 | _imagenet_224_224_132M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | movenet_ntd_pt | pt_mov                 | 350   | 2750.94         |
| 9 |                | enet_coco_192_192_0.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | MT-res         | pt_MT-resnet1          | 350   | 279.93          |
| 0 | net18_mixed_pt | 8_mixed_320_512_13.65G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | mu             | pt_multitaskv          | 350   | 136.76          |
| 1 | lti_task_v3_pt | 3_mixed_320_512_25.44G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ocr_pt         | pt_OCR_                | 350   | 24.20           |
| 2 |                | ICDAR2015_960_960_875G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_dept       | p                      | 350   | 2948.67         |
| 3 | hwise_res50_pt | t_OFA-depthwise-res50_ |       |                 |
|   |                | imagenet_176_176_2.49G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_r          | pt_OFA-rc              | 350   | 33.32           |
| 4 | can_latency_pt | an_DIV2K_360_640_45.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_resnet     | pt_OFA-resnet50_       | 350   | 711.82          |
| 5 | 50_baseline_pt | imagenet_224_224_15.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 941.31          |
| 6 | pruned_0_45_pt | enet_224_224_0.45_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 971.98          |
| 7 | pruned_0_60_pt | enet_224_224_0.60_6.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 1514.09         |
| 8 | pruned_0_74_pt | enet_192_192_0.74_3.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_re         | pt_OFA-resnet50_imag   | 350   | 2264.41         |
| 9 | snet50_0_9B_pt | enet_160_160_0.88_1.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | ofa_yolo_pt    | pt_OFA-yo              | 350   | 220.24          |
| 0 |                | lo_coco_640_640_48.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | ofa_yolo_      | pt_OFA-yolo_c          | 350   | 276.05          |
| 1 | pruned_0_30_pt | oco_640_640_0.3_34.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | ofa_yolo_      | pt_OFA-yolo_c          | 350   | 318.73          |
| 2 | pruned_0_50_pt | oco_640_640_0.6_24.62G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | person-orienta | pt_person-or           | 350   | 5444.55         |
| 3 | tion_pruned_pt | ientation_224_112_558M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | perso          | pt_personreid-res50_m  | 350   | 2012.23         |
| 4 | nreid-res50_pt | arket1501_256_128_5.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pe             | pt_                    | 350   | 2480.24         |
| 5 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_4_pt | t1501_256_128_0.4_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pe             | pt_                    | 350   | 2610.40         |
| 6 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_5_pt | t1501_256_128_0.5_2.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pe             | pt_                    | 350   | 2734.31         |
| 7 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_6_pt | t1501_256_128_0.6_2.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pe             | pt_                    | 350   | 2934.00         |
| 8 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_7_pt | t1501_256_128_0.7_1.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | perso          | pt_personreid-res18_   | 350   | 4782.06         |
| 9 | nreid-res18_pt | market1501_176_80_1.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | pmg_pt         | pt_                    | 350   | 2000.55         |
| 0 |                | pmg_rp2k_224_224_2.28G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | pointpai       | pt_point               | 350   | 20.38           |
| 1 | nting_nuscenes | painting_nuscenes_126G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | poin           | pt_pointpillars        | 350   | 3.13            |
| 2 | tpillars_kitti | _kitti_12000_100_10.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | pointpi        | pt_pointpillars_       | 350   | 39.08           |
| 3 | llars_nuscenes | nuscenes_40000_64_108G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | rcan_pruned_tf | tf_rcan_DIV            | 350   | 35.28           |
| 4 |                | 2K_360_640_0.98_86.95G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | re             | tf_refin               | 350   | 200.90          |
| 5 | finedet_VOC_tf | edet_VOC_320_320_81.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | Refin          | tf_RefineDet-Medi      | 350   | 200.96          |
| 6 | eDet-Medical_E | cal_EDD_320_320_81.28G |       |                 |
|   | DD_baseline_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | RefineD        | tf_RefineDet-Medical_  | 350   | 332.21          |
| 7 | et-Medical_EDD | EDD_320_320_0.5_41.42G |       |                 |
|   | _pruned_0_5_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | RefineDe       | tf_RefineDet-Medical_E | 350   | 418.89          |
| 8 | t-Medical_EDD_ | DD_320_320_0.75_20.54G |       |                 |
|   | pruned_0_75_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | RefineDe       | tf_RefineDet-Medical_E | 350   | 536.56          |
| 9 | t-Medical_EDD_ | DD_320_320_0.85_12.32G |       |                 |
|   | pruned_0_85_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | RefineDet-     | tf_RefineDet-Medical_  | 350   | 555.17          |
| 0 | Medical_EDD_tf | EDD_320_320_0.88_9.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | r              | tf_resnetv1_50_        | 350   | 1817.21         |
| 1 | esnet_v1_50_tf | imagenet_224_224_6.97G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet_v1_50_  | tf_resnetv1_50_imag    | 350   | 1909.43         |
| 2 | pruned_0_38_tf | enet_224_224_0.38_4.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet_v1_50_  | tf_resnetv1_50_image   | 350   | 2310.18         |
| 3 | pruned_0_65_tf | net_224_224_0.65_2.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | re             | tf_resnetv1_101_       | 350   | 1146.44         |
| 4 | snet_v1_101_tf | imagenet_224_224_14.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | re             | tf_resnetv1_152_i      | 350   | 825.95          |
| 5 | snet_v1_152_tf | magenet_224_224_21.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | r              | tf_resnetv2_50_        | 350   | 427.18          |
| 6 | esnet_v2_50_tf | imagenet_299_299_13.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | re             | tf_resnetv2_101_i      | 350   | 245.96          |
| 7 | snet_v2_101_tf | magenet_299_299_26.78G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | re             | tf_resnetv2_152_i      | 350   | 172.53          |
| 8 | snet_v2_152_tf | magenet_299_299_40.47G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50_tf2   | tf2_resnet50_          | 350   | 1533.14         |
| 9 |                | imagenet_224_224_7.76G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | resnet50_pt    | pt_resnet50            | 350   | 1675.05         |
| 0 |                | _imagenet_224_224_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | resnet50       | pt_resnet50_ima        | 350   | 1698.25         |
| 1 | _pruned_0_3_pt | genet_224_224_0.3_5.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | resnet50       | pt_resnet50_ima        | 350   | 1765.32         |
| 2 | _pruned_0_4_pt | genet_224_224_0.4_4.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | resnet50       | pt_resnet50_ima        | 350   | 1837.51         |
| 3 | _pruned_0_5_pt | genet_224_224_0.5_4.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | resnet50       | pt_resnet50_ima        | 350   | 1955.57         |
| 4 | _pruned_0_6_pt | genet_224_224_0.6_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | resnet50       | pt_resnet50_ima        | 350   | 2068.41         |
| 5 | _pruned_0_7_pt | genet_224_224_0.7_2.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | S              | pt_sa-g                | 350   | 11.28           |
| 6 | A_gate_base_pt | ate_NYUv2_360_360_178G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | salsanext_pt   | p                      | 350   | 108.73          |
| 7 |                | t_salsanext_semantic-k |       |                 |
|   |                | itti_64_2048_0.6_20.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | s              | pt                     | 350   | 60.46           |
| 8 | alsanext_v2_pt | _salsanextv2_semantic- |       |                 |
|   |                | kitti_64_2048_0.75_32G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | semantic       | tf2_erfnet_c           | 350   | 63.70           |
| 9 | _seg_citys_tf2 | ityscapes_512_1024_54G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | SemanticFPN    | pt_SemanticFPN_        | 350   | 731.00          |
| 0 | _cityscapes_pt | cityscapes_256_512_10G |       |                 |
| 0 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | SemanticFPN_   | pt_Sema                | 350   | 207.71          |
| 0 | Mobilenetv2_pt | nticFPN-mobilenetv2_ci |       |                 |
| 1 |                | tyscapes_512_1024_5.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | SESR_S_pt      | pt_SESR                | 350   | 113.79          |
| 0 |                | -S_DIV2K_360_640_7.48G |       |                 |
| 2 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | solo_pt        | pt_                    | 350   | 24.77           |
| 0 |                | SOLO_coco_640_640_107G |       |                 |
| 3 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | SqueezeNet_pt  | pt_squeezenet_i        | 350   | 3375.85         |
| 0 |                | magenet_224_224_351.7M |       |                 |
| 4 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ssd_incept     | tf_ssdinceptio         | 350   | 105.80          |
| 0 | ion_v2_coco_tf | nv2_coco_300_300_9.62G |       |                 |
| 5 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ssd_mobile     | tf_ssdmobilene         | 350   | 2286.02         |
| 0 | net_v1_coco_tf | tv1_coco_300_300_2.47G |       |                 |
| 6 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ssd_mobile     | tf_ssdmobilene         | 350   | 1291.50         |
| 0 | net_v2_coco_tf | tv2_coco_300_300_3.75G |       |                 |
| 7 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ssd_resnet_    | tf_ssdresnet50v1_f     | 350   | 80.70           |
| 0 | 50_fpn_coco_tf | pn_coco_640_640_178.4G |       |                 |
| 8 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ssdlite_mobile | tf_ssdlite_mobilen     | 350   | 1875.26         |
| 0 | net_v2_coco_tf | etv2_coco_300_300_1.5G |       |                 |
| 9 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ssr_pt         | pt_                    | 350   | 70.78           |
| 1 |                | SSR_CVC_256_256_39.72G |       |                 |
| 0 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | t              | pt_textmountai         | 350   | 30.81           |
| 1 | extmountain_pt | n_ICDAR_960_960_575.2G |       |                 |
| 1 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | tsd_yolox_pt   | pt_yo                  | 350   | 192.58          |
| 1 |                | lox_TT100K_640_640_73G |       |                 |
| 2 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ultrafast_pt   | pt_ultrafa             | 350   | 592.00          |
| 1 |                | st_CULane_288_800_8.4G |       |                 |
| 3 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | un             | pt_unet_               | 350   | 128.55          |
| 1 | et_chaos-CT_pt | chaos-CT_512_512_23.3G |       |                 |
| 4 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | chen_col       | pt_vehi                | 350   | 2839.65         |
| 1 | or_resnet18_pt | cle-color-classificati |       |                 |
| 5 |                | on_color_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vehicle_ma     | pt_vehicl              | 350   | 2830.61         |
| 1 | ke_resnet18_pt | e-make-classification_ |       |                 |
| 6 |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vehicle_ty     | pt_vehicl              | 350   | 2840.23         |
| 1 | pe_resnet18_pt | e-type-classification_ |       |                 |
| 7 |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16_tf      | tf_vgg16_i             | 350   | 328.51          |
| 1 |                | magenet_224_224_30.96G |       |                 |
| 8 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16_        | tf_vgg16_imagen        | 350   | 596.46          |
| 1 | pruned_0_43_tf | et_224_224_0.43_17.67G |       |                 |
| 9 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16         | tf_vgg16_image         | 350   | 647.41          |
| 2 | _pruned_0_5_tf | net_224_224_0.5_15.64G |       |                 |
| 0 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_19_tf      | tf_vgg19_i             | 350   | 293.45          |
| 2 |                | magenet_224_224_39.28G |       |                 |
| 1 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | yolov3_voc_tf  | tf_yol                 | 350   | 279.33          |
| 2 |                | ov3_voc_416_416_65.63G |       |                 |
| 2 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | y              | tf2_yol                | 350   | 276.70          |
| 2 | olov3_coco_tf2 | ov3_coco_416_416_65.9G |       |                 |
| 3 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | yolov4_416_tf  | tf_yol                 | 350   | 240.10          |
| 2 |                | ov4_coco_416_416_60.3G |       |                 |
| 4 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | yolov4_512_tf  | tf_yol                 | 350   | 137.83          |
| 2 |                | ov4_coco_512_512_91.2G |       |                 |
| 5 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+

The following table lists the performance number including end-to-end
throughput for models on the ``Versal ACAP VCK5000`` board with
DPUCVDX8H-aieDWC running at 6PE@350 MHz in Gen3x16:

+---+----------------+------------------------+-------+-----------------+
| N | Model          | GPU Model Standard     | DPU   | E2E throughput  |
| o |                | Name                   | Freq  | (fps) Multi     |
| . |                |                        | uency | Thread          |
|   |                |                        | (MHz) |                 |
+===+================+========================+=======+=================+
| 1 | bcc_pt         | pt_BCC_shangh          | 350   | 79.10           |
|   |                | aitech_800_1000_268.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | c2d2_lite_pt   | pt_C2D2l               | 350   | 36.73           |
|   |                | ite_CC20_512_512_6.86G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | drunet_pt      | pt_DRUN                | 350   | 153.31          |
|   |                | et_Kvasir_528_608_0.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | efficientNe    | tf_e                   | 350   | 1377.58         |
|   | t-edgetpu-M_tf | fficientnet-edgetpu-M_ |       |                 |
|   |                | imagenet_240_240_7.34G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | efficientNe    | tf_e                   | 350   | 2488.34         |
|   | t-edgetpu-S_tf | fficientnet-edgetpu-S_ |       |                 |
|   |                | imagenet_224_224_4.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | ENet           | pt_ENet_ci             | 350   | 141.03          |
|   | _cityscapes_pt | tyscapes_512_1024_8.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | face_mas       | pt_face-mask-d         | 350   | 1335.01         |
|   | k_detection_pt | etection_512_512_0.59G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | f              | pt_fac                 | 350   | 30992.50        |
|   | ace-quality_pt | e-quality_80_60_61.68M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | facerec-res    | pt_facerec-resn        | 350   | 4411.13         |
|   | net20_mixed_pt | et20_mixed_112_96_3.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fac            | pt_fac                 | 350   | 20918.30        |
| 0 | ereid-large_pt | ereid-large_96_96_515M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fac            | pt_fa                  | 350   | 32349.60        |
| 1 | ereid-small_pt | cereid-small_80_80_90M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fadnet         | pt_fadnet_             | 350   | 9.23            |
| 2 |                | sceneflow_576_960_441G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fadnet_pruned  | pt_fadnet_scene        | 350   | 10.35           |
| 3 |                | flow_576_960_0.65_154G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FairMot_pt     | pt_FairMOT             | 350   | 426.19          |
| 4 |                | _mixed_640_480_0.5_36G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FPN-resnet18_  | pt_FPN-resnet18_cov    | 350   | 958.15          |
| 5 | covid19-seg_pt | id19-seg_352_352_22.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | Inceptio       | tf_inceptionresnetv2_i | 350   | 491.97          |
| 6 | n_resnet_v2_tf | magenet_299_299_26.35G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inception           | 350   | 3500.30         |
| 7 | nception_v1_tf | v1_imagenet_224_224_3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | i              | tf_inceptionv2_        | 350   | 330.52          |
| 8 | nception_v2_tf | imagenet_224_224_3.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inceptionv3_i       | 350   | 914.35          |
| 9 | nception_v3_tf | magenet_299_299_11.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | tf_inceptionv3_ima     | 350   | 1001.82         |
| 0 | _pruned_0_2_tf | genet_299_299_0.2_9.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | tf_inceptionv3_ima     | 350   | 1078.26         |
| 1 | _pruned_0_4_tf | genet_299_299_0.4_6.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | In             | tf2_inceptionv3_       | 350   | 963.51          |
| 2 | ception_v3_tf2 | imagenet_299_299_11.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | pt_inceptionv3_        | 350   | 909.00          |
| 3 | nception_v3_pt | imagenet_299_299_11.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_i       | 350   | 1062.79         |
| 4 | _pruned_0_3_pt | magenet_299_299_0.3_8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 350   | 1064.91         |
| 5 | _pruned_0_4_pt | genet_299_299_0.4_6.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 350   | 1253.56         |
| 6 | _pruned_0_5_pt | genet_299_299_0.5_5.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 350   | 1406.71         |
| 7 | _pruned_0_6_pt | genet_299_299_0.6_4.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | tf_inceptionv4_i       | 350   | 503.54          |
| 8 | nception_v4_tf | magenet_299_299_24.55G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | medica         | tf2_2d-une             | 350   | 1352.53         |
| 9 | l_seg_cell_tf2 | t_nuclei_128_128_5.31G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mlperf_s       | tf_mlperf_resnet       | 350   | 73.27           |
| 0 | sd_resnet34_tf | 34_coco_1200_1200_433G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mlpe           | tf_mlperf_resnet50_    | 350   | 3403.95         |
| 1 | rf_resnet50_tf | imagenet_224_224_8.19G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilene       | tf_mobilenetEdge0.75   | 350   | 5689.70         |
| 2 | t_edge_0_75_tf | _imagenet_224_224_624M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf_mobilenetEdge1.0    | 350   | 5176.39         |
| 3 | et_edge_1_0_tf | _imagenet_224_224_990M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf2_mobilenetv1_       | 350   | 7820.25         |
| 4 | et_1_0_224_tf2 | imagenet_224_224_1.15G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet_     | tf_mobilenetv1_0.2     | 350   | 20510.00        |
| 5 | v1_0_25_128_tf | 5_imagenet_128_128_27M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet      | tf_mobilenetv1_0.5     | 350   | 14825.30        |
| 6 | _v1_0_5_160_tf | _imagenet_160_160_150M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet      | tf_mobilenetv1_1.0_    | 350   | 8058.30         |
| 7 | _v1_1_0_224_tf | imagenet_224_224_1.14G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf                     | 350   | 8241.64         |
| 8 | et_v1_1_0_224_ | _mobilenetv1_1.0_image |       |                 |
|   | pruned_0_11_tf | net_224_224_0.11_1.02G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf_mobilenetv1_1.0_im  | 350   | 8330.08         |
| 9 | et_v1_1_0_224_ | agenet_224_224_0.12_1G |       |                 |
|   | pruned_0_12_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet      | tf_mobilenetv2_1.0     | 350   | 6504.78         |
| 0 | _v2_1_0_224_tf | _imagenet_224_224_602M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet      | tf_mobilenetv2_1.4_    | 350   | 4971.92         |
| 1 | _v2_1_4_224_tf | imagenet_224_224_1.16G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet_v2   | tf_mobilenetv2_citysc  | 350   | 18.42           |
| 2 | _cityscapes_tf | apes_1024_2048_132.74G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | movenet_ntd_pt | pt_mov                 | 350   | 2267.16         |
| 3 |                | enet_coco_192_192_0.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | MT-res         | pt_MT-resnet1          | 350   | 421.09          |
| 4 | net18_mixed_pt | 8_mixed_320_512_13.65G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mu             | pt_multitaskv          | 350   | 203.77          |
| 5 | lti_task_v3_pt | 3_mixed_320_512_25.44G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ocr_pt         | pt_OCR_                | 350   | 11.79           |
| 6 |                | ICDAR2015_960_960_875G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_dept       | p                      | 350   | 3237.38         |
| 7 | hwise_res50_pt | t_OFA-depthwise-res50_ |       |                 |
|   |                | imagenet_176_176_2.49G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_r          | pt_OFA-rc              | 350   | 50.14           |
| 8 | can_latency_pt | an_DIV2K_360_640_45.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_resnet     | pt_OFA-resnet50_       | 350   | 1206.24         |
| 9 | 50_baseline_pt | imagenet_224_224_15.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 1767.11         |
| 0 | pruned_0_45_pt | enet_224_224_0.45_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 1784.54         |
| 1 | pruned_0_60_pt | enet_224_224_0.60_6.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 2813.63         |
| 2 | pruned_0_74_pt | enet_192_192_0.74_3.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_re         | pt_OFA-resnet50_imag   | 350   | 4013.12         |
| 3 | snet50_0_9B_pt | enet_160_160_0.88_1.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_yolo_pt    | pt_OFA-yo              | 350   | 285.31          |
| 4 |                | lo_coco_640_640_48.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_yolo_      | pt_OFA-yolo_c          | 350   | 376.22          |
| 5 | pruned_0_30_pt | oco_640_640_0.3_34.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_yolo_      | pt_OFA-yolo_c          | 350   | 441.71          |
| 6 | pruned_0_50_pt | oco_640_640_0.6_24.62G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | person-orienta | pt_person-or           | 350   | 9448.00         |
| 7 | tion_pruned_pt | ientation_224_112_558M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | perso          | pt_personreid-res50_m  | 350   | 3750.97         |
| 8 | nreid-res50_pt | arket1501_256_128_5.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pe             | pt_                    | 350   | 4676.61         |
| 9 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_4_pt | t1501_256_128_0.4_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pe             | pt_                    | 350   | 5033.46         |
| 0 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_5_pt | t1501_256_128_0.5_2.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pe             | pt_                    | 350   | 5360.20         |
| 1 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_6_pt | t1501_256_128_0.6_2.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pe             | pt_                    | 350   | 6037.33         |
| 2 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_7_pt | t1501_256_128_0.7_1.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | perso          | pt_personreid-res18_   | 350   | 8047.31         |
| 3 | nreid-res18_pt | market1501_176_80_1.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pmg_pt         | pt_                    | 350   | 3423.98         |
| 4 |                | pmg_rp2k_224_224_2.28G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pointpai       | pt_pointpain           | 350   | 18.09           |
| 5 | nting_nuscenes | ting_nuscenes_126G_2.5 |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pointpi        | pt_pointpillars_       | 350   | 37.10           |
| 6 | llars_nuscenes | nuscenes_40000_64_108G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | rcan_pruned_tf | tf_rcan_DIV            | 350   | 53.41           |
| 7 |                | 2K_360_640_0.98_86.95G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | re             | tf_refin               | 350   | 307.64          |
| 8 | finedet_VOC_tf | edet_VOC_320_320_81.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | Refin          | tf_RefineDet-Medi      | 350   | 307.59          |
| 9 | eDet-Medical_E | cal_EDD_320_320_81.28G |       |                 |
|   | DD_baseline_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | RefineD        | tf_RefineDet-Medical_  | 350   | 528.82          |
| 0 | et-Medical_EDD | EDD_320_320_0.5_41.42G |       |                 |
|   | _pruned_0_5_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | RefineDe       | tf_RefineDet-Medical_E | 350   | 663.61          |
| 1 | t-Medical_EDD_ | DD_320_320_0.75_20.54G |       |                 |
|   | pruned_0_75_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | RefineDe       | tf_RefineDet-Medical_E | 350   | 957.42          |
| 2 | t-Medical_EDD_ | DD_320_320_0.85_12.32G |       |                 |
|   | pruned_0_85_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | RefineDet-     | tf_RefineDet-Medical_  | 350   | 998.94          |
| 3 | Medical_EDD_tf | EDD_320_320_0.88_9.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | r              | tf_resnetv1_50_        | 350   | 3736.48         |
| 4 | esnet_v1_50_tf | imagenet_224_224_6.97G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet_v1_50_  | tf_resnetv1_50_imag    | 350   | 4023.41         |
| 5 | pruned_0_38_tf | enet_224_224_0.38_4.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet_v1_50_  | tf_resnetv1_50_image   | 350   | 5432.02         |
| 6 | pruned_0_65_tf | net_224_224_0.65_2.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | re             | tf_resnetv1_101_       | 350   | 2244.46         |
| 7 | snet_v1_101_tf | imagenet_224_224_14.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | re             | tf_resnetv1_152_i      | 350   | 1598.02         |
| 8 | snet_v1_152_tf | magenet_224_224_21.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50_tf2   | tf2_resnet50_          | 350   | 3152.91         |
| 9 |                | imagenet_224_224_7.76G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50_pt    | pt_resnet50            | 350   | 3432.18         |
| 0 |                | _imagenet_224_224_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50       | pt_resnet50_ima        | 350   | 3504.63         |
| 1 | _pruned_0_3_pt | genet_224_224_0.3_5.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50       | pt_resnet50_ima        | 350   | 3718.85         |
| 2 | _pruned_0_4_pt | genet_224_224_0.4_4.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50       | pt_resnet50_ima        | 350   | 3982.00         |
| 3 | _pruned_0_5_pt | genet_224_224_0.5_4.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50       | pt_resnet50_ima        | 350   | 4381.54         |
| 4 | _pruned_0_6_pt | genet_224_224_0.6_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50       | pt_resnet50_ima        | 350   | 4804.25         |
| 5 | _pruned_0_7_pt | genet_224_224_0.7_2.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | salsanext_pt   | p                      | 350   | 158.09          |
| 6 |                | t_salsanext_semantic-k |       |                 |
|   |                | itti_64_2048_0.6_20.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | s              | pt                     | 350   | 91.22           |
| 7 | alsanext_v2_pt | _salsanextv2_semantic- |       |                 |
|   |                | kitti_64_2048_0.75_32G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | semantic       | tf2_erfnet_c           | 350   | 114.93          |
| 8 | _seg_citys_tf2 | ityscapes_512_1024_54G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | SemanticFPN    | pt_SemanticFPN_        | 350   | 1049.65         |
| 9 | _cityscapes_pt | cityscapes_256_512_10G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | SemanticFPN_   | pt_Sema                | 350   | 234.42          |
| 0 | Mobilenetv2_pt | nticFPN-mobilenetv2_ci |       |                 |
|   |                | tyscapes_512_1024_5.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | SESR_S_pt      | pt_SESR                | 350   | 185.87          |
| 1 |                | -S_DIV2K_360_640_7.48G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | SqueezeNet_pt  | pt_squeezenet_i        | 350   | 7062.16         |
| 2 |                | magenet_224_224_351.7M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssd_incept     | tf_ssdinceptio         | 350   | 166.21          |
| 3 | ion_v2_coco_tf | nv2_coco_300_300_9.62G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssd_mobile     | tf_ssdmobilene         | 350   | 2471.66         |
| 4 | net_v1_coco_tf | tv1_coco_300_300_2.47G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssd_mobile     | tf_ssdmobilene         | 350   | 1879.99         |
| 5 | net_v2_coco_tf | tv2_coco_300_300_3.75G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssd_resnet_    | tf_ssdresnet50v1_f     | 350   | 108.15          |
| 6 | 50_fpn_coco_tf | pn_coco_640_640_178.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssdlite_mobile | tf_ssdlite_mobilen     | 350   | 2552.13         |
| 7 | net_v2_coco_tf | etv2_coco_300_300_1.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssr_pt         | pt_                    | 350   | 90.06           |
| 8 |                | SSR_CVC_256_256_39.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | t              | pt_textmountai         | 350   | 41.83           |
| 9 | extmountain_pt | n_ICDAR_960_960_575.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | tsd_yolox_pt   | pt_yo                  | 350   | 261.90          |
| 0 |                | lox_TT100K_640_640_73G |       |                 |
| 0 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ultrafast_pt   | pt_ultrafa             | 350   | 1039.88         |
| 0 |                | st_CULane_288_800_8.4G |       |                 |
| 1 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | un             | pt_unet_               | 350   | 201.48          |
| 0 | et_chaos-CT_pt | chaos-CT_512_512_23.3G |       |                 |
| 2 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | chen_col       | pt_vehi                | 350   | 5301.04         |
| 0 | or_resnet18_pt | cle-color-classificati |       |                 |
| 3 |                | on_color_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vehicle_ma     | pt_vehicl              | 350   | 5284.14         |
| 0 | ke_resnet18_pt | e-make-classification_ |       |                 |
| 4 |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vehicle_ty     | pt_vehicl              | 350   | 5300.12         |
| 0 | pe_resnet18_pt | e-type-classification_ |       |                 |
| 5 |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16_tf      | tf_vgg16_i             | 350   | 500.50          |
| 0 |                | magenet_224_224_30.96G |       |                 |
| 6 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16_        | tf_vgg16_imagen        | 350   | 965.05          |
| 0 | pruned_0_43_tf | et_224_224_0.43_17.67G |       |                 |
| 7 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16         | tf_vgg16_image         | 350   | 1047.76         |
| 0 | _pruned_0_5_tf | net_224_224_0.5_15.64G |       |                 |
| 8 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_19_tf      | tf_vgg19_i             | 350   | 446.08          |
| 0 |                | magenet_224_224_39.28G |       |                 |
| 9 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | yolov3_voc_tf  | tf_yol                 | 350   | 389.71          |
| 1 |                | ov3_voc_416_416_65.63G |       |                 |
| 0 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | y              | tf2_yol                | 350   | 385.25          |
| 1 | olov3_coco_tf2 | ov3_coco_416_416_65.9G |       |                 |
| 1 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+

The following table lists the performance number including end-to-end
throughput for models on the ``Versal ACAP VCK5000`` board with
DPUCVDX8H-aieMISC running at 6PE@350 MHz in Gen3x16:

+---+----------------+------------------------+-------+-----------------+
| N | Model          | GPU Model Standard     | DPU   | E2E throughput  |
| o |                | Name                   | Freq  | (fps) Multi     |
| . |                |                        | uency | Thread          |
|   |                |                        | (MHz) |                 |
+===+================+========================+=======+=================+
| 1 | centerpoint    | pt_centerp             | 350   | 10.83           |
|   |                | oint_astyx_2560_40_54G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | CLOCs          | pt_CLOCs_kitti_2.0     | 350   | 17.10           |
+---+----------------+------------------------+-------+-----------------+
| 3 | drunet_pt      | pt_DRUN                | 350   | 141.07          |
|   |                | et_Kvasir_528_608_0.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ENet           | pt_ENet_ci             | 350   | 91.51           |
|   | _cityscapes_pt | tyscapes_512_1024_8.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | f              | pt_fac                 | 350   | 24426.30        |
|   | ace-quality_pt | e-quality_80_60_61.68M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | facerec-res    | pt_facerec-resn        | 350   | 4397.97         |
|   | net20_mixed_pt | et20_mixed_112_96_3.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | fac            | pt_fac                 | 350   | 17183.60        |
|   | ereid-large_pt | ereid-large_96_96_515M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | fac            | pt_fa                  | 350   | 25900.50        |
|   | ereid-small_pt | cereid-small_80_80_90M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | fadnet         | pt_fadnet_             | 350   | 8.91            |
|   |                | sceneflow_576_960_441G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FairMot_pt     | pt_FairMOT             | 350   | 381.48          |
| 0 |                | _mixed_640_480_0.5_36G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FPN-resnet18_  | pt_FPN-resnet18_cov    | 350   | 880.96          |
| 1 | covid19-seg_pt | id19-seg_352_352_22.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | Inceptio       | tf_inceptionresnetv2_i | 350   | 448.06          |
| 2 | n_resnet_v2_tf | magenet_299_299_26.35G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inception           | 350   | 2224.44         |
| 3 | nception_v1_tf | v1_imagenet_224_224_3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inceptionv3_i       | 350   | 711.53          |
| 4 | nception_v3_tf | magenet_299_299_11.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | tf_inceptionv3_ima     | 350   | 781.03          |
| 5 | _pruned_0_2_tf | genet_299_299_0.2_9.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | tf_inceptionv3_ima     | 350   | 842.44          |
| 6 | _pruned_0_4_tf | genet_299_299_0.4_6.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | In             | tf2_inceptionv3_       | 350   | 719.18          |
| 7 | ception_v3_tf2 | imagenet_299_299_11.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | pt_inceptionv3_        | 350   | 710.52          |
| 8 | nception_v3_pt | imagenet_299_299_11.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | pt_inceptionv3_i       | 350   | 817.96          |
| 9 | _pruned_0_3_pt | magenet_299_299_0.3_8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 350   | 833.27          |
| 0 | _pruned_0_4_pt | genet_299_299_0.4_6.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 350   | 962.62          |
| 1 | _pruned_0_5_pt | genet_299_299_0.5_5.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 350   | 1087.39         |
| 2 | _pruned_0_6_pt | genet_299_299_0.6_4.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | tf_inceptionv4_i       | 350   | 397.71          |
| 3 | nception_v4_tf | magenet_299_299_24.55G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | medica         | tf2_2d-une             | 350   | 1294.82         |
| 4 | l_seg_cell_tf2 | t_nuclei_128_128_5.31G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | mlperf_s       | tf_mlperf_resnet       | 350   | 67.65           |
| 5 | sd_resnet34_tf | 34_coco_1200_1200_433G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | mlpe           | tf_mlperf_resnet50_    | 350   | 2454.78         |
| 6 | rf_resnet50_tf | imagenet_224_224_8.19G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ocr_pt         | pt_OCR_                | 350   | 33.85           |
| 7 |                | ICDAR2015_960_960_875G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ofa_r          | pt_OFA-rc              | 350   | 49.92           |
| 8 | can_latency_pt | an_DIV2K_360_640_45.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ofa_resnet     | pt_OFA-resnet50_       | 350   | 1033.25         |
| 9 | 50_baseline_pt | imagenet_224_224_15.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 1376.67         |
| 0 | pruned_0_45_pt | enet_224_224_0.45_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 1402.76         |
| 1 | pruned_0_60_pt | enet_224_224_0.60_6.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 2216.75         |
| 2 | pruned_0_74_pt | enet_192_192_0.74_3.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_re         | pt_OFA-resnet50_imag   | 350   | 3305.77         |
| 3 | snet50_0_9B_pt | enet_160_160_0.88_1.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_yolo_pt    | pt_OFA-yo              | 350   | 279.77          |
| 4 |                | lo_coco_640_640_48.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_yolo_      | pt_OFA-yolo_c          | 350   | 366.23          |
| 5 | pruned_0_30_pt | oco_640_640_0.3_34.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_yolo_      | pt_OFA-yolo_c          | 350   | 424.64          |
| 6 | pruned_0_50_pt | oco_640_640_0.6_24.62G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | person-orienta | pt_person-or           | 350   | 8060.23         |
| 7 | tion_pruned_pt | ientation_224_112_558M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | perso          | pt_personreid-res50_m  | 350   | 3012.43         |
| 8 | nreid-res50_pt | arket1501_256_128_5.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | pe             | pt_                    | 350   | 3697.90         |
| 9 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_4_pt | t1501_256_128_0.4_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pe             | pt_                    | 350   | 3885.44         |
| 0 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_5_pt | t1501_256_128_0.5_2.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pe             | pt_                    | 350   | 4073.29         |
| 1 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_6_pt | t1501_256_128_0.6_2.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pe             | pt_                    | 350   | 4374.36         |
| 2 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_7_pt | t1501_256_128_0.7_1.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | perso          | pt_personreid-res18_   | 350   | 7050.56         |
| 3 | nreid-res18_pt | market1501_176_80_1.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pmg_pt         | pt_                    | 350   | 2887.85         |
| 4 |                | pmg_rp2k_224_224_2.28G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pointpai       | pt_pointpain           | 350   | 15.55           |
| 5 | nting_nuscenes | ting_nuscenes_126G_2.5 |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | poin           | pt_pointpillars        | 350   | 3.10            |
| 6 | tpillars_kitti | _kitti_12000_100_10.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pointpi        | pt_pointpillars_       | 350   | 34.46           |
| 7 | llars_nuscenes | nuscenes_40000_64_108G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | rcan_pruned_tf | tf_rcan_DIV            | 350   | 52.80           |
| 8 |                | 2K_360_640_0.98_86.95G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | re             | tf_refin               | 350   | 292.19          |
| 9 | finedet_VOC_tf | edet_VOC_320_320_81.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | Refin          | tf_RefineDet-Medi      | 350   | 292.72          |
| 0 | eDet-Medical_E | cal_EDD_320_320_81.28G |       |                 |
|   | DD_baseline_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | RefineD        | tf_RefineDet-Medical_  | 350   | 486.09          |
| 1 | et-Medical_EDD | EDD_320_320_0.5_41.42G |       |                 |
|   | _pruned_0_5_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | RefineDe       | tf_RefineDet-Medical_E | 350   | 597.70          |
| 2 | t-Medical_EDD_ | DD_320_320_0.75_20.54G |       |                 |
|   | pruned_0_75_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | RefineDe       | tf_RefineDet-Medical_E | 350   | 800.49          |
| 3 | t-Medical_EDD_ | DD_320_320_0.85_12.32G |       |                 |
|   | pruned_0_85_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | RefineDet-     | tf_RefineDet-Medical_  | 350   | 828.60          |
| 4 | Medical_EDD_tf | EDD_320_320_0.88_9.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | r              | tf_resnetv1_50_        | 350   | 2718.97         |
| 5 | esnet_v1_50_tf | imagenet_224_224_6.97G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | resnet_v1_50_  | tf_resnetv1_50_imag    | 350   | 2852.94         |
| 6 | pruned_0_38_tf | enet_224_224_0.38_4.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | resnet_v1_50_  | tf_resnetv1_50_image   | 350   | 3448.88         |
| 7 | pruned_0_65_tf | net_224_224_0.65_2.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | re             | tf_resnetv1_101_       | 350   | 1715.80         |
| 8 | snet_v1_101_tf | imagenet_224_224_14.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | re             | tf_resnetv1_152_i      | 350   | 1237.54         |
| 9 | snet_v1_152_tf | magenet_224_224_21.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50_tf2   | tf2_resnet50_          | 350   | 2306.45         |
| 0 |                | imagenet_224_224_7.76G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50_pt    | pt_resnet50            | 350   | 2507.43         |
| 1 |                | _imagenet_224_224_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50       | pt_resnet50_ima        | 350   | 2635.89         |
| 2 | _pruned_0_3_pt | genet_224_224_0.3_5.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50       | pt_resnet50_ima        | 350   | 2744.08         |
| 3 | _pruned_0_4_pt | genet_224_224_0.4_4.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50       | pt_resnet50_ima        | 350   | 2918.96         |
| 4 | _pruned_0_5_pt | genet_224_224_0.5_4.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50       | pt_resnet50_ima        | 350   | 3086.25         |
| 5 | _pruned_0_6_pt | genet_224_224_0.6_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50       | pt_resnet50_ima        | 350   | 2504.13         |
| 6 | _pruned_0_7_pt | genet_224_224_0.7_2.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | salsanext_pt   | p                      | 350   | 152.58          |
| 7 |                | t_salsanext_semantic-k |       |                 |
|   |                | itti_64_2048_0.6_20.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | s              | pt                     | 350   | 83.71           |
| 8 | alsanext_v2_pt | _salsanextv2_semantic- |       |                 |
|   |                | kitti_64_2048_0.75_32G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | semantic       | tf2_erfnet_c           | 350   | 80.99           |
| 9 | _seg_citys_tf2 | ityscapes_512_1024_54G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | SemanticFPN    | pt_SemanticFPN_        | 350   | 992.31          |
| 0 | _cityscapes_pt | cityscapes_256_512_10G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | SESR_S_pt      | pt_SESR                | 350   | 170.61          |
| 1 |                | -S_DIV2K_360_640_7.48G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | solo_pt        | pt_                    | 350   | 24.13           |
| 2 |                | SOLO_coco_640_640_107G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | SqueezeNet_pt  | pt_squeezenet_i        | 350   | 4617.91         |
| 3 |                | magenet_224_224_351.7M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | ssd_resnet_    | tf_ssdresnet50v1_f     | 350   | 105.05          |
| 4 | 50_fpn_coco_tf | pn_coco_640_640_178.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | t              | pt_textmountai         | 350   | 40.61           |
| 5 | extmountain_pt | n_ICDAR_960_960_575.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | tsd_yolox_pt   | pt_yo                  | 350   | 257.27          |
| 6 |                | lox_TT100K_640_640_73G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | ultrafast_pt   | pt_ultrafa             | 350   | 870.67          |
| 7 |                | st_CULane_288_800_8.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | un             | pt_unet_               | 350   | 185.08          |
| 8 | et_chaos-CT_pt | chaos-CT_512_512_23.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | chen_col       | pt_vehi                | 350   | 4166.42         |
| 9 | or_resnet18_pt | cle-color-classificati |       |                 |
|   |                | on_color_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | vehicle_ma     | pt_vehicl              | 350   | 4161.00         |
| 0 | ke_resnet18_pt | e-make-classification_ |       |                 |
|   |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | vehicle_ty     | pt_vehicl              | 350   | 4168.73         |
| 1 | pe_resnet18_pt | e-type-classification_ |       |                 |
|   |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | vgg_16_tf      | tf_vgg16_i             | 350   | 478.68          |
| 2 |                | magenet_224_224_30.96G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | vgg_16_        | tf_vgg16_imagen        | 350   | 887.03          |
| 3 | pruned_0_43_tf | et_224_224_0.43_17.67G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | vgg_16_        | tf_vgg16_image         | 350   | 958.77          |
| 4 | pruned_0_43_tf | net_224_224_0.5_15.64G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | vgg_19_tf      | tf_vgg19_i             | 350   | 428.68          |
| 5 |                | magenet_224_224_39.28G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | yolov3_voc_tf  | tf_yol                 | 350   | 386.94          |
| 6 |                | ov3_voc_416_416_65.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | y              | tf2_yol                | 350   | 382.30          |
| 7 | olov3_coco_tf2 | ov3_coco_416_416_65.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | yolov4_416_tf  | tf_yol                 | 350   | 315.31          |
| 8 |                | ov4_coco_416_416_60.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | yolov4_512_tf  | tf_yol                 | 350   | 171.52          |
| 9 |                | ov4_coco_512_512_91.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+

The following table lists the performance number including end-to-end
throughput for models on the ``Versal ACAP VCK5000`` board with
DPUCVDX8H running at 8PE@350 MHz in Gen3x16:

+---+----------------+------------------------+-------+-----------------+
| N | Model          | GPU Model Standard     | DPU   | E2E throughput  |
| o |                | Name                   | Freq  | (fps) Multi     |
| . |                |                        | uency | Thread          |
|   |                |                        | (MHz) |                 |
+===+================+========================+=======+=================+
| 1 | drunet_pt      | pt_DRUN                | 350   | 204.59          |
|   |                | et_Kvasir_528_608_0.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ENet           | pt_ENet_ci             | 350   | 148.32          |
|   | _cityscapes_pt | tyscapes_512_1024_8.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | f              | pt_fac                 | 350   | 31080.90        |
|   | ace-quality_pt | e-quality_80_60_61.68M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | fadnet         | pt_fadnet_             | 350   | 9.65            |
|   |                | sceneflow_576_960_441G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | FairMot_pt     | pt_FairMOT             | 350   | 506.42          |
|   |                | _mixed_640_480_0.5_36G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | FPN-resnet18_  | pt_FPN-resnet18_cov    | 350   | 1177.00         |
|   | covid19-seg_pt | id19-seg_352_352_22.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | I              | tf_inception           | 350   | 4224.61         |
|   | nception_v1_tf | v1_imagenet_224_224_3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | medica         | tf2_2d-une             | 350   | 1504.80         |
|   | l_seg_cell_tf2 | t_nuclei_128_128_5.31G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | mlperf_s       | tf_mlperf_resnet       | 350   | 76.78           |
|   | sd_resnet34_tf | 34_coco_1200_1200_433G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | mlpe           | tf_mlperf_resnet50_    | 350   | 4519.23         |
| 0 | rf_resnet50_tf | imagenet_224_224_8.19G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ocr_pt         | pt_OCR_                | 350   | 12.69           |
| 1 |                | ICDAR2015_960_960_875G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ofa_r          | pt_OFA-rc              | 350   | 66.81           |
| 2 | can_latency_pt | an_DIV2K_360_640_45.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ofa_resnet     | pt_OFA-resnet50_       | 350   | 1490.15         |
| 3 | 50_baseline_pt | imagenet_224_224_15.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 2288.00         |
| 4 | pruned_0_45_pt | enet_224_224_0.45_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 350   | 2166.96         |
| 5 | pruned_0_60_pt | enet_224_224_0.60_6.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | pe             | pt_                    | 350   | 6182.04         |
| 6 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_4_pt | t1501_256_128_0.4_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | pe             | pt_                    | 350   | 6635.75         |
| 7 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_5_pt | t1501_256_128_0.5_2.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | pe             | pt_                    | 350   | 7057.42         |
| 8 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_6_pt | t1501_256_128_0.6_2.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | pe             | pt_                    | 350   | 7852.00         |
| 9 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_7_pt | t1501_256_128_0.7_1.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | rcan_pruned_tf | tf_rcan_DIV            | 350   | 71.19           |
| 0 |                | 2K_360_640_0.98_86.95G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | re             | tf_refin               | 350   | 399.88          |
| 1 | finedet_VOC_tf | edet_VOC_320_320_81.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | Refin          | tf_RefineDet-Medi      | 350   | 399.94          |
| 2 | eDet-Medical_E | cal_EDD_320_320_81.28G |       |                 |
|   | DD_baseline_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | RefineD        | tf_RefineDet-Medical_  | 350   | 692.54          |
| 3 | et-Medical_EDD | EDD_320_320_0.5_41.42G |       |                 |
|   | _pruned_0_5_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | RefineDe       | tf_RefineDet-Medical_E | 350   | 738.23          |
| 4 | t-Medical_EDD_ | DD_320_320_0.75_20.54G |       |                 |
|   | pruned_0_75_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | RefineDe       | tf_RefineDet-Medical_E | 350   | 1220.63         |
| 5 | t-Medical_EDD_ | DD_320_320_0.85_12.32G |       |                 |
|   | pruned_0_85_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | RefineDet-     | tf_RefineDet-Medical_  | 350   | 1265.40         |
| 6 | Medical_EDD_tf | EDD_320_320_0.88_9.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | r              | tf_resnetv1_50_        | 350   | 4944.30         |
| 7 | esnet_v1_50_tf | imagenet_224_224_6.97G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | resnet_v1_50_  | tf_resnetv1_50_imag    | 350   | 5335.53         |
| 8 | pruned_0_38_tf | enet_224_224_0.38_4.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | resnet_v1_50_  | tf_resnetv1_50_image   | 350   | 7077.62         |
| 9 | pruned_0_65_tf | net_224_224_0.65_2.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | re             | tf_resnetv1_101_       | 350   | 2979.15         |
| 0 | snet_v1_101_tf | imagenet_224_224_14.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | re             | tf_resnetv1_152_i      | 350   | 2122.98         |
| 1 | snet_v1_152_tf | magenet_224_224_21.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | resnet50_tf2   | tf2_resnet50_          | 350   | 4167.80         |
| 2 |                | imagenet_224_224_7.76G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | resnet50_pt    | pt_resnet50            | 350   | 4544.06         |
| 3 |                | _imagenet_224_224_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | resnet50       | pt_resnet50_ima        | 350   | 4651.83         |
| 4 | _pruned_0_3_pt | genet_224_224_0.3_5.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | resnet50       | pt_resnet50_ima        | 350   | 4942.05         |
| 5 | _pruned_0_4_pt | genet_224_224_0.4_4.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | resnet50       | pt_resnet50_ima        | 350   | 5280.03         |
| 6 | _pruned_0_5_pt | genet_224_224_0.5_4.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | resnet50       | pt_resnet50_ima        | 350   | 5813.44         |
| 7 | _pruned_0_6_pt | genet_224_224_0.6_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | resnet50       | pt_resnet50_ima        | 350   | 6370.42         |
| 8 | _pruned_0_7_pt | genet_224_224_0.7_2.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | salsanext_pt   | p                      | 350   | 159.51          |
| 9 |                | t_salsanext_semantic-k |       |                 |
|   |                | itti_64_2048_0.6_20.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | s              | pt                     | 350   | 89.41           |
| 0 | alsanext_v2_pt | _salsanextv2_semantic- |       |                 |
|   |                | kitti_64_2048_0.75_32G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | semantic       | tf2_erfnet_c           | 350   | 117.98          |
| 1 | _seg_citys_tf2 | ityscapes_512_1024_54G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | SemanticFPN    | pt_SemanticFPN_        | 350   | 1084.85         |
| 2 | _cityscapes_pt | cityscapes_256_512_10G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | SESR_S_pt      | pt_SESR                | 350   | 233.24          |
| 3 |                | -S_DIV2K_360_640_7.48G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | SqueezeNet_pt  | pt_squeezenet_i        | 350   | 7536.39         |
| 4 |                | magenet_224_224_351.7M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ssd_resnet_    | tf_ssdresnet50v1_f     | 350   | 114.53          |
| 5 | 50_fpn_coco_tf | pn_coco_640_640_178.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | t              | pt_textmountai         | 350   | 49.48           |
| 6 | extmountain_pt | n_ICDAR_960_960_575.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ultrafast_pt   | pt_ultrafa             | 350   | 1359.92         |
| 7 |                | st_CULane_288_800_8.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | un             | pt_unet_               | 350   | 247.93          |
| 8 | et_chaos-CT_pt | chaos-CT_512_512_23.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | chen_col       | pt_vehi                | 350   | 6526.42         |
| 9 | or_resnet18_pt | cle-color-classificati |       |                 |
|   |                | on_color_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | vehicle_ma     | pt_vehicl              | 350   | 6910.37         |
| 0 | ke_resnet18_pt | e-make-classification_ |       |                 |
|   |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | vehicle_ty     | pt_vehicl              | 350   | 6945.23         |
| 1 | pe_resnet18_pt | e-type-classification_ |       |                 |
|   |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | yolov3_voc_tf  | tf_yol                 | 350   | 476.97          |
| 2 |                | ov3_voc_416_416_65.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | y              | tf2_yol                | 350   | 469.79          |
| 3 | olov3_coco_tf2 | ov3_coco_416_416_65.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+

.. raw:: html

   </details>

Performance on U50lv
~~~~~~~~~~~~~~~~~~~~

Measured with Vitis AI 2.5 and Vitis AI Library 2.5

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput for each model on the ``Alveo U50lv`` board with 10 DPUCAHX8H
kernels running at 275Mhz in Gen3x4:

+---+----------------+------------------------+-------+-----------------+
| N | Model          | GPU Model Standard     | DPU   | E2E throughput  |
| o |                | Name                   | Freq  | (fps) Multi     |
| . |                |                        | uency | Thread          |
|   |                |                        | (MHz) |                 |
+===+================+========================+=======+=================+
| 1 | drunet_pt      | pt_DRUN                | 165   | 336.95          |
|   |                | et_Kvasir_528_608_0.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ENet           | pt_ENet_ci             | 165   | 87.66           |
|   | _cityscapes_pt | tyscapes_512_1024_8.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | f              | pt_fac                 | 165   | 21381.20        |
|   | ace-quality_pt | e-quality_80_60_61.68M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | facerec-res    | pt_facerec-resn        | 165   | 1251.82         |
|   | net20_mixed_pt | et20_mixed_112_96_3.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | fac            | pt_fac                 | 165   | 7323.87         |
|   | ereid-large_pt | ereid-large_96_96_515M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | fac            | pt_fa                  | 165   | 21136.40        |
|   | ereid-small_pt | cereid-small_80_80_90M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | fadnet_pt      | pt_fadnet_             | 165   | 1.16            |
|   |                | sceneflow_576_960_441G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | FairMot_pt     | pt_FairMOT             | 165   | 138.17          |
|   |                | _mixed_640_480_0.5_36G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | FPN-resnet18_  | pt_FPN-resnet18_cov    | 165   | 212.52          |
|   | covid19-seg_pt | id19-seg_352_352_22.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | Inceptio       | tf_inceptionresnetv2_i | 165   | 158.87          |
| 0 | n_resnet_v2_tf | magenet_299_299_26.35G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inception           | 165   | 1172.51         |
| 1 | nception_v1_tf | v1_imagenet_224_224_3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inceptionv3_i       | 165   | 376.13          |
| 2 | nception_v3_tf | magenet_299_299_11.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | tf_inceptionv3_ima     | 165   | 408.94          |
| 3 | _pruned_0_2_tf | genet_299_299_0.2_9.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | tf_inceptionv3_ima     | 165   | 496.03          |
| 4 | _pruned_0_4_tf | genet_299_299_0.4_6.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | In             | tf2_inceptionv3_       | 165   | 381.25          |
| 5 | ception_v3_tf2 | imagenet_299_299_11.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | pt_inceptionv3_        | 165   | 436.47          |
| 6 | nception_v3_pt | imagenet_299_299_11.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | pt_inceptionv3_i       | 165   | 449.85          |
| 7 | _pruned_0_3_pt | magenet_299_299_0.3_8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | pt_inceptionv3_ima     | 165   | 493.43          |
| 8 | _pruned_0_4_pt | genet_299_299_0.4_6.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | pt_inceptionv3_ima     | 165   | 550.77          |
| 9 | _pruned_0_5_pt | genet_299_299_0.5_5.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 165   | 632.56          |
| 0 | _pruned_0_6_pt | genet_299_299_0.6_4.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | tf_inceptionv4_i       | 165   | 170.86          |
| 1 | nception_v4_tf | magenet_299_299_24.55G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | medica         | tf2_2d-une             | 165   | 1068.04         |
| 2 | l_seg_cell_tf2 | t_nuclei_128_128_5.31G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | mlperf_s       | tf_mlperf_resnet       | 165   | 511.80          |
| 3 | sd_resnet34_tf | 34_coco_1200_1200_433G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | mlpe           | tf_mlperf_resnet50_    | 165   | 14.11           |
| 4 | rf_resnet50_tf | imagenet_224_224_8.19G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ocr_pt         | pt_OCR_                | 165   | 4.85            |
| 5 |                | ICDAR2015_960_960_875G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ofa_r          | pt_OFA-rc              | 165   | 45.04           |
| 6 | can_latency_pt | an_DIV2K_360_640_45.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ofa_resnet     | pt_OFA-resnet50_       | 165   | 306.88          |
| 7 | 50_baseline_pt | imagenet_224_224_15.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 165   | 476.08          |
| 8 | pruned_0_45_pt | enet_224_224_0.45_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 165   | 598.32          |
| 9 | pruned_0_60_pt | enet_224_224_0.60_6.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 165   | 909.32          |
| 0 | pruned_0_74_pt | enet_192_192_0.74_3.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_re         | pt_OFA-resnet50_imag   | 165   | 1548.17         |
| 1 | snet50_0_9B_pt | enet_160_160_0.88_1.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_yolo_pt    | pt_OFA-yo              | 165   | 98.38           |
| 2 |                | lo_coco_640_640_48.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_yolo_      | pt_OFA-yolo_c          | 165   | 124.95          |
| 3 | pruned_0_30_pt | oco_640_640_0.3_34.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | ofa_yolo_      | pt_OFA-yolo_c          | 165   | 159.48          |
| 4 | pruned_0_50_pt | oco_640_640_0.6_24.62G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | person-orienta | pt_person-or           | 165   | 5858.88         |
| 5 | tion_pruned_pt | ientation_224_112_558M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | perso          | pt_personreid-res50_m  | 165   | 825.99          |
| 6 | nreid-res50_pt | arket1501_256_128_5.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | pe             | pt_                    | 165   | 1078.24         |
| 7 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_4_pt | t1501_256_128_0.4_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | pe             | pt_                    | 165   | 1181.30         |
| 8 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_5_pt | t1501_256_128_0.5_2.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | pe             | pt_                    | 165   | 1338.94         |
| 9 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_6_pt | t1501_256_128_0.6_2.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pe             | pt_                    | 165   | 1500.32         |
| 0 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_7_pt | t1501_256_128_0.7_1.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | perso          | pt_personreid-res18_   | 165   | 3502.12         |
| 1 | nreid-res18_pt | market1501_176_80_1.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pmg_pt         | pt_                    | 165   | 1005.87         |
| 2 |                | pmg_rp2k_224_224_2.28G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pointpai       | pt_pointpain           | 165   | 11.36           |
| 3 | nting_nuscenes | ting_nuscenes_126G_2.5 |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | pointpi        | pt_pointpillars_       | 165   | 21.16           |
| 4 | llars_nuscenes | nuscenes_40000_64_108G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | rcan_pruned_tf | tf_rcan_DIV            | 165   | 46.73           |
| 5 |                | 2K_360_640_0.98_86.95G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | re             | tf_refin               | 165   | 72.55           |
| 6 | finedet_VOC_tf | edet_VOC_320_320_81.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | Refin          | tf_RefineDet-Medi      | 165   | 73.04           |
| 7 | eDet-Medical_E | cal_EDD_320_320_81.28G |       |                 |
|   | DD_baseline_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | RefineD        | tf_RefineDet-Medical_  | 165   | 135.76          |
| 8 | et-Medical_EDD | EDD_320_320_0.5_41.42G |       |                 |
|   | _pruned_0_5_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | RefineDe       | tf_RefineDet-Medical_E | 165   | 246.23          |
| 9 | t-Medical_EDD_ | DD_320_320_0.75_20.54G |       |                 |
|   | pruned_0_75_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | RefineDe       | tf_RefineDet-Medical_E | 165   | 377.29          |
| 0 | t-Medical_EDD_ | DD_320_320_0.85_12.32G |       |                 |
|   | pruned_0_85_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | RefineDet-     | tf_RefineDet-Medical_  | 165   | 430.81          |
| 1 | Medical_EDD_tf | EDD_320_320_0.88_9.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | r              | tf_resnetv1_50_        | 165   | 593.21          |
| 2 | esnet_v1_50_tf | imagenet_224_224_6.97G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | resnet_v1_50_  | tf_resnetv1_50_imag    | 165   | 770.88          |
| 3 | pruned_0_38_tf | enet_224_224_0.38_4.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | resnet_v1_50_  | tf_resnetv1_50_image   | 165   | 1033.57         |
| 4 | pruned_0_65_tf | net_224_224_0.65_2.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | re             | tf_resnetv1_101_       | 165   | 307.93          |
| 5 | snet_v1_101_tf | imagenet_224_224_14.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | re             | tf_resnetv1_152_i      | 165   | 205.48          |
| 6 | snet_v1_152_tf | magenet_224_224_21.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | resnet50_tf2   | tf2_resnet50_          | 165   | 593.40          |
| 7 |                | imagenet_224_224_7.76G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | resnet50_pt    | pt_resnet50            | 165   | 511.61          |
| 8 |                | _imagenet_224_224_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | resnet50       | pt_resnet50_ima        | 165   | 616.01          |
| 9 | _pruned_0_3_pt | genet_224_224_0.3_5.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50       | pt_resnet50_ima        | 165   | 669.76          |
| 0 | _pruned_0_4_pt | genet_224_224_0.4_4.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50       | pt_resnet50_ima        | 165   | 740.15          |
| 1 | _pruned_0_5_pt | genet_224_224_0.5_4.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50       | pt_resnet50_ima        | 165   | 841.86          |
| 2 | _pruned_0_6_pt | genet_224_224_0.6_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet50       | pt_resnet50_ima        | 165   | 939.27          |
| 3 | _pruned_0_7_pt | genet_224_224_0.7_2.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | salsanext_pt   | p                      | 165   | 148.27          |
| 4 |                | t_salsanext_semantic-k |       |                 |
|   |                | itti_64_2048_0.6_20.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | s              | pt                     | 165   | 32.70           |
| 5 | alsanext_v2_pt | _salsanextv2_semantic- |       |                 |
|   |                | kitti_64_2048_0.75_32G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | semantic       | tf2_erfnet_c           | 165   | 56.95           |
| 6 | _seg_citys_tf2 | ityscapes_512_1024_54G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | SemanticFPN    | pt_SemanticFPN_        | 165   | 432.47          |
| 7 | _cityscapes_pt | cityscapes_256_512_10G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | SESR_S_pt      | pt_SESR                | 165   | 188.92          |
| 8 |                | -S_DIV2K_360_640_7.48G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | SqueezeNet_pt  | pt_squeezenet_i        | 165   | 4127.72         |
| 9 |                | magenet_224_224_351.7M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | ssd_resnet_    | tf_ssdresnet50v1_f     | 165   | 32.34           |
| 0 | 50_fpn_coco_tf | pn_coco_640_640_178.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | t              | pt_textmountai         | 165   | 4.34            |
| 1 | extmountain_pt | n_ICDAR_960_960_575.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | tsd_yolox_pt   | pt_yo                  | 165   | 71.97           |
| 2 |                | lox_TT100K_640_640_73G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | ultrafast_pt   | pt_ultrafa             | 165   | 247.12          |
| 3 |                | st_CULane_288_800_8.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | un             | pt_unet_               | 165   | 86.41           |
| 4 | et_chaos-CT_pt | chaos-CT_512_512_23.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | chen_col       | pt_vehi                | 165   | 1316.57         |
| 5 | or_resnet18_pt | cle-color-classificati |       |                 |
|   |                | on_color_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | vehicle_ma     | pt_vehicl              | 165   | 1314.66         |
| 6 | ke_resnet18_pt | e-make-classification_ |       |                 |
|   |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | vehicle_ty     | pt_vehicl              | 165   | 1099.59         |
| 7 | pe_resnet18_pt | e-type-classification_ |       |                 |
|   |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | vgg_16_tf      | tf_vgg16_i             | 165   | 151.61          |
| 8 |                | magenet_224_224_30.96G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | vgg_16_        | tf_vgg16_imagen        | 165   | 283.28          |
| 9 | pruned_0_43_tf | et_224_224_0.43_17.67G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | vgg_16         | tf_vgg16_image         | 165   | 306.23          |
| 0 | _pruned_0_5_tf | net_224_224_0.5_15.64G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | vgg_19_tf      | tf_vgg19_i             | 165   | 125.71          |
| 1 |                | magenet_224_224_39.28G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | yolov3_voc_tf  | tf_yol                 | 165   | 78.62           |
| 2 |                | ov3_voc_416_416_65.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | y              | tf2_yol                | 165   | 77.83           |
| 3 | olov3_coco_tf2 | ov3_coco_416_416_65.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+

The following table lists the performance number including end-to-end
throughput for each model on the ``Alveo U50lv`` board with 8
DPUCAHX8H-DWC kernels running at 275Mhz in Gen3x4:

+---+----------------+------------------------+-------+-----------------+
| N | Model          | GPU Model Standard     | DPU   | E2E throughput  |
| o |                | Name                   | Freq  | (fps) Multi     |
| . |                |                        | uency | Thread          |
|   |                |                        | (MHz) |                 |
+===+================+========================+=======+=================+
| 1 | bcc_pt         | pt_BCC_shangh          | 165   | 16.67           |
|   |                | aitech_800_1000_268.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | c2d2_lite_pt   | pt_C2D2l               | 165   | 17.55           |
|   |                | ite_CC20_512_512_6.86G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | drunet_pt      | pt_DRUN                | 165   | 268.16          |
|   |                | et_Kvasir_528_608_0.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | efficientNe    | tf_e                   | 165   | 543.66          |
|   | t-edgetpu-M_tf | fficientnet-edgetpu-M_ |       |                 |
|   |                | imagenet_240_240_7.34G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | efficientNe    | tf_e                   | 165   | 883.96          |
|   | t-edgetpu-S_tf | fficientnet-edgetpu-S_ |       |                 |
|   |                | imagenet_224_224_4.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | ENet           | pt_ENet_ci             | 165   | 69.98           |
|   | _cityscapes_pt | tyscapes_512_1024_8.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | face_mas       | pt_face-mask-d         | 165   | 737.28          |
|   | k_detection_pt | etection_512_512_0.59G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | f              | pt_fac                 | 165   | 20109.10        |
|   | ace-quality_pt | e-quality_80_60_61.68M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | facerec-res    | pt_facerec-resn        | 165   | 1001.98         |
|   | net20_mixed_pt | et20_mixed_112_96_3.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fac            | pt_fac                 | 165   | 5908.28         |
| 0 | ereid-large_pt | ereid-large_96_96_515M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fac            | pt_fa                  | 165   | 17368.60        |
| 1 | ereid-small_pt | cereid-small_80_80_90M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fadnet_pt      | pt_fadnet_             | 165   | 3.86            |
| 2 |                | sceneflow_576_960_441G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FairMot_pt     | pt_FairMOT             | 165   | 110.50          |
| 3 |                | _mixed_640_480_0.5_36G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FPN-resnet18_  | pt_FPN-resnet18_cov    | 165   | 170.04          |
| 4 | covid19-seg_pt | id19-seg_352_352_22.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | Inceptio       | tf_inceptionresnetv2_i | 165   | 127.05          |
| 5 | n_resnet_v2_tf | magenet_299_299_26.35G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inception           | 165   | 938.83          |
| 6 | nception_v1_tf | v1_imagenet_224_224_3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | i              | tf_inceptionv2_        | 165   | 322.89          |
| 7 | nception_v2_tf | imagenet_224_224_3.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inceptionv3_i       | 165   | 300.88          |
| 8 | nception_v3_tf | magenet_299_299_11.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | tf_inceptionv3_ima     | 165   | 327.95          |
| 9 | _pruned_0_2_tf | genet_299_299_0.2_9.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | tf_inceptionv3_ima     | 165   | 396.64          |
| 0 | _pruned_0_4_tf | genet_299_299_0.4_6.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | In             | tf2_inceptionv3_       | 165   | 305.50          |
| 1 | ception_v3_tf2 | imagenet_299_299_11.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | pt_inceptionv3_        | 165   | 300.75          |
| 2 | nception_v3_pt | imagenet_299_299_11.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_i       | 165   | 360.21          |
| 3 | _pruned_0_3_pt | magenet_299_299_0.3_8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 165   | 394.91          |
| 4 | _pruned_0_4_pt | genet_299_299_0.4_6.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 165   | 440.54          |
| 5 | _pruned_0_5_pt | genet_299_299_0.5_5.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 165   | 507.14          |
| 6 | _pruned_0_6_pt | genet_299_299_0.6_4.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | tf_inceptionv4_i       | 165   | 136.73          |
| 7 | nception_v4_tf | magenet_299_299_24.55G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | medica         | tf2_2d-une             | 165   | 855.48          |
| 8 | l_seg_cell_tf2 | t_nuclei_128_128_5.31G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | mlperf_s       | tf_mlperf_resnet       | 165   | 11.29           |
| 9 | sd_resnet34_tf | 34_coco_1200_1200_433G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mlpe           | tf_mlperf_resnet50_    | 165   | 409.83          |
| 0 | rf_resnet50_tf | imagenet_224_224_8.19G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilene       | tf_mobilenetEdge0.75   | 165   | 1954.19         |
| 1 | t_edge_0_75_tf | _imagenet_224_224_624M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf_mobilenetEdge1.0    | 165   | 1583.69         |
| 2 | et_edge_1_0_tf | _imagenet_224_224_990M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf2_mobilenetv1_       | 165   | 2415.83         |
| 3 | et_1_0_224_tf2 | imagenet_224_224_1.15G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet_     | tf_mobilenetv1_0.2     | 165   | 14834.80        |
| 4 | v1_0_25_128_tf | 5_imagenet_128_128_27M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet      | tf_mobilenetv1_0.5     | 165   | 8837.20         |
| 5 | _v1_0_5_160_tf | _imagenet_160_160_150M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet      | tf_mobilenetv1_1.0_    | 165   | 2415.28         |
| 6 | _v1_1_0_224_tf | imagenet_224_224_1.14G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf                     | 165   | 2487.63         |
| 7 | et_v1_1_0_224_ | _mobilenetv1_1.0_image |       |                 |
|   | pruned_0_11_tf | net_224_224_0.11_1.02G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf_mobilenetv1_1.0_im  | 165   | 2507.15         |
| 8 | et_v1_1_0_224_ | agenet_224_224_0.12_1G |       |                 |
|   | pruned_0_12_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet      | tf_mobilenetv2_1.0     | 165   | 2226.09         |
| 9 | _v2_1_0_224_tf | _imagenet_224_224_602M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mobilenet      | tf_mobilenetv2_1.4_    | 165   | 1498.50         |
| 0 | _v2_1_4_224_tf | imagenet_224_224_1.16G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | movenet_ntd_pt | pt_mov                 | 165   | 1747.97         |
| 1 |                | enet_coco_192_192_0.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | MT-res         | pt_MT-resnet1          | 165   | 217.50          |
| 2 | net18_mixed_pt | 8_mixed_320_512_13.65G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | mu             | pt_multitaskv          | 165   | 118.96          |
| 3 | lti_task_v3_pt | 3_mixed_320_512_25.44G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ocr_pt         | pt_OCR_                | 165   | 4.72            |
| 4 |                | ICDAR2015_960_960_875G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_dept       | p                      | 165   | 2269.73         |
| 5 | hwise_res50_pt | t_OFA-depthwise-res50_ |       |                 |
|   |                | imagenet_176_176_2.49G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_r          | pt_OFA-rc              | 165   | 36.06           |
| 6 | can_latency_pt | an_DIV2K_360_640_45.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_resnet     | pt_OFA-resnet50_       | 165   | 245.50          |
| 7 | 50_baseline_pt | imagenet_224_224_15.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 165   | 379.84          |
| 8 | pruned_0_45_pt | enet_224_224_0.45_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 165   | 479.30          |
| 9 | pruned_0_60_pt | enet_224_224_0.60_6.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 165   | 729.18          |
| 0 | pruned_0_74_pt | enet_192_192_0.74_3.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_re         | pt_OFA-resnet50_imag   | 165   | 1242.57         |
| 1 | snet50_0_9B_pt | enet_160_160_0.88_1.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_yolo_pt    | pt_OFA-yo              | 165   | 78.87           |
| 2 |                | lo_coco_640_640_48.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_yolo_      | pt_OFA-yolo_c          | 165   | 99.68           |
| 3 | pruned_0_30_pt | oco_640_640_0.3_34.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | ofa_yolo_      | pt_OFA-yolo_c          | 165   | 127.52          |
| 4 | pruned_0_50_pt | oco_640_640_0.6_24.62G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | person-orienta | pt_person-or           | 165   | 4727.32         |
| 5 | tion_pruned_pt | ientation_224_112_558M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | perso          | pt_personreid-res50_m  | 165   | 661.48          |
| 6 | nreid-res50_pt | arket1501_256_128_5.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pe             | pt_                    | 165   | 865.41          |
| 7 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_4_pt | t1501_256_128_0.4_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pe             | pt_                    | 165   | 946.77          |
| 8 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_5_pt | t1501_256_128_0.5_2.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pe             | pt_                    | 165   | 1074.36         |
| 9 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_6_pt | t1501_256_128_0.6_2.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pe             | pt_                    | 165   | 1204.93         |
| 0 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_7_pt | t1501_256_128_0.7_1.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | perso          | pt_personreid-res18_   | 165   | 2813.31         |
| 1 | nreid-res18_pt | market1501_176_80_1.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pmg_pt         | pt_                    | 165   | 806.20          |
| 2 |                | pmg_rp2k_224_224_2.28G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pointpai       | pt_pointpain           | 165   | 11.33           |
| 3 | nting_nuscenes | ting_nuscenes_126G_2.5 |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | pointpi        | pt_pointpillars_       | 165   | 20.55           |
| 4 | llars_nuscenes | nuscenes_40000_64_108G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | rcan_pruned_tf | tf_rcan_DIV            | 165   | 37.41           |
| 5 |                | 2K_360_640_0.98_86.95G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | re             | tf_refin               | 165   | 58.01           |
| 6 | finedet_VOC_tf | edet_VOC_320_320_81.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | Refin          | tf_RefineDet-Medi      | 165   | 58.51           |
| 7 | eDet-Medical_E | cal_EDD_320_320_81.28G |       |                 |
|   | DD_baseline_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | RefineD        | tf_RefineDet-Medical_  | 165   | 108.58          |
| 8 | et-Medical_EDD | EDD_320_320_0.5_41.42G |       |                 |
|   | _pruned_0_5_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | RefineDe       | tf_RefineDet-Medical_E | 165   | 196.84          |
| 9 | t-Medical_EDD_ | DD_320_320_0.75_20.54G |       |                 |
|   | pruned_0_75_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | RefineDe       | tf_RefineDet-Medical_E | 165   | 302.12          |
| 0 | t-Medical_EDD_ | DD_320_320_0.85_12.32G |       |                 |
|   | pruned_0_85_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | RefineDet-     | tf_RefineDet-Medical_  | 165   | 345.01          |
| 1 | Medical_EDD_tf | EDD_320_320_0.88_9.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | r              | tf_resnetv1_50_        | 165   | 475.39          |
| 2 | esnet_v1_50_tf | imagenet_224_224_6.97G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet_v1_50_  | tf_resnetv1_50_imag    | 165   | 618.43          |
| 3 | pruned_0_38_tf | enet_224_224_0.38_4.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet_v1_50_  | tf_resnetv1_50_image   | 165   | 828.47          |
| 4 | pruned_0_65_tf | net_224_224_0.65_2.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | re             | tf_resnetv1_101_       | 165   | 246.71          |
| 5 | snet_v1_101_tf | imagenet_224_224_14.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | re             | tf_resnetv1_152_i      | 165   | 164.32          |
| 6 | snet_v1_152_tf | magenet_224_224_21.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50_tf2   | tf2_resnet50_          | 165   | 475.35          |
| 7 |                | imagenet_224_224_7.76G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50_pt    | pt_resnet50            | 165   | 409.64          |
| 8 |                | _imagenet_224_224_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50       | pt_resnet50_ima        | 165   | 493.81          |
| 9 | _pruned_0_3_pt | genet_224_224_0.3_5.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50       | pt_resnet50_ima        | 165   | 536.85          |
| 0 | _pruned_0_4_pt | genet_224_224_0.4_4.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50       | pt_resnet50_ima        | 165   | 592.43          |
| 1 | _pruned_0_5_pt | genet_224_224_0.5_4.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50       | pt_resnet50_ima        | 165   | 674.89          |
| 2 | _pruned_0_6_pt | genet_224_224_0.6_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | resnet50       | pt_resnet50_ima        | 165   | 753.52          |
| 3 | _pruned_0_7_pt | genet_224_224_0.7_2.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | salsanext_pt   | p                      | 165   | 139.21          |
| 4 |                | t_salsanext_semantic-k |       |                 |
|   |                | itti_64_2048_0.6_20.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | s              | pt                     | 165   | 28.22           |
| 5 | alsanext_v2_pt | _salsanextv2_semantic- |       |                 |
|   |                | kitti_64_2048_0.75_32G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | semantic       | tf2_erfnet_c           | 165   | 47.44           |
| 6 | _seg_citys_tf2 | ityscapes_512_1024_54G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | SemanticFPN    | pt_SemanticFPN_        | 165   | 346.75          |
| 7 | _cityscapes_pt | cityscapes_256_512_10G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | SemanticFPN_   | pt_Sema                | 165   | 135.54          |
| 8 | Mobilenetv2_pt | nticFPN-mobilenetv2_ci |       |                 |
|   |                | tyscapes_512_1024_5.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | SESR_S_pt      | pt_SESR                | 165   | 151.05          |
| 9 |                | -S_DIV2K_360_640_7.48G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | SqueezeNet_pt  | pt_squeezenet_i        | 165   | 2884.24         |
| 0 |                | magenet_224_224_351.7M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssd_incept     | tf_ssdinceptio         | 165   | 156.87          |
| 1 | ion_v2_coco_tf | nv2_coco_300_300_9.62G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssd_mobile     | tf_ssdmobilene         | 165   | 1127.75         |
| 2 | net_v1_coco_tf | tv1_coco_300_300_2.47G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssd_mobile     | tf_ssdmobilene         | 165   | 635.81          |
| 3 | net_v2_coco_tf | tv2_coco_300_300_3.75G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssd_resnet_    | tf_ssdresnet50v1_f     | 165   | 25.93           |
| 4 | 50_fpn_coco_tf | pn_coco_640_640_178.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssdlite_mobile | tf_ssdlite_mobilen     | 165   | 1045.29         |
| 5 | net_v2_coco_tf | etv2_coco_300_300_1.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssr_pt         | pt_                    | 165   | 26.84           |
| 6 |                | SSR_CVC_256_256_39.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | t              | pt_textmountai         | 165   | 7.83            |
| 7 | extmountain_pt | n_ICDAR_960_960_575.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | tsd_yolox_pt   | pt_yo                  | 165   | 57.60           |
| 8 |                | lox_TT100K_640_640_73G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ultrafast_pt   | pt_ultrafa             | 165   | 197.57          |
| 9 |                | st_CULane_288_800_8.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | un             | pt_unet_               | 165   | 69.14           |
| 0 | et_chaos-CT_pt | chaos-CT_512_512_23.3G |       |                 |
| 0 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | chen_col       | pt_vehi                | 165   | 1056.01         |
| 0 | or_resnet18_pt | cle-color-classificati |       |                 |
| 1 |                | on_color_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vehicle_ma     | pt_vehicl              | 165   | 1054.04         |
| 0 | ke_resnet18_pt | e-make-classification_ |       |                 |
| 2 |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vehicle_ty     | pt_vehicl              | 165   | 1055.42         |
| 0 | pe_resnet18_pt | e-type-classification_ |       |                 |
| 3 |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16_tf      | tf_vgg16_i             | 165   | 121.12          |
| 0 |                | magenet_224_224_30.96G |       |                 |
| 4 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16_        | tf_vgg16_imagen        | 165   | 226.44          |
| 0 | pruned_0_43_tf | et_224_224_0.43_17.67G |       |                 |
| 5 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16_        | tf_vgg16_image         | 165   | 244.66          |
| 0 | pruned_0_43_tf | net_224_224_0.5_15.64G |       |                 |
| 6 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_19_tf      | tf_vgg19_i             | 165   | 100.65          |
| 0 |                | magenet_224_224_39.28G |       |                 |
| 7 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | yolov3_voc_tf  | tf_yol                 | 165   | 62.93           |
| 0 |                | ov3_voc_416_416_65.63G |       |                 |
| 8 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | y              | tf2_yol                | 165   | 62.35           |
| 0 | olov3_coco_tf2 | ov3_coco_416_416_65.9G |       |                 |
| 9 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+

.. raw:: html

   </details>

Performance on U55C DWC
~~~~~~~~~~~~~~~~~~~~~~~

Measured with Vitis AI 2.5 and Vitis AI Library 2.5

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput for each model on the ``Alveo U55C`` board with 11
DPUCAHX8H-DWC kernels running at 300Mhz in Gen3x4:

+---+----------------+------------------------+-------+-----------------+
| N | Model          | GPU Model Standard     | DPU   | E2E throughput  |
| o |                | Name                   | Freq  | (fps) Multi     |
| . |                |                        | uency | Thread          |
|   |                |                        | (MHz) |                 |
+===+================+========================+=======+=================+
| 1 | bcc_pt         | pt_BCC_shangh          | 300   | 37.43           |
|   |                | aitech_800_1000_268.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | c2d2_lite_pt   | pt_C2D2l               | 300   | 28.25           |
|   |                | ite_CC20_512_512_6.86G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | drunet_pt      | pt_DRUN                | 300   | 469.21          |
|   |                | et_Kvasir_528_608_0.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | efficientNe    | tf_e                   | 300   | 1204.68         |
|   | t-edgetpu-M_tf | fficientnet-edgetpu-M_ |       |                 |
|   |                | imagenet_240_240_7.34G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | efficientNe    | tf_e                   | 300   | 2000.42         |
|   | t-edgetpu-S_tf | fficientnet-edgetpu-S_ |       |                 |
|   |                | imagenet_224_224_4.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | ENet           | pt_ENet_ci             | 300   | 147.29          |
|   | _cityscapes_pt | tyscapes_512_1024_8.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | face_mas       | pt_face-mask-d         | 300   | 1581.35         |
|   | k_detection_pt | etection_512_512_0.59G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | f              | pt_fac                 | 300   | 30918.60        |
|   | ace-quality_pt | e-quality_80_60_61.68M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | facerec-res    | pt_facerec-resn        | 300   | 2423.09         |
|   | net20_mixed_pt | et20_mixed_112_96_3.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fac            | pt_fac                 | 300   | 15138.20        |
| 0 | ereid-large_pt | ereid-large_96_96_515M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | fac            | pt_fa                  | 300   | 32819.60        |
| 1 | ereid-small_pt | cereid-small_80_80_90M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FairMot_pt     | pt_FairMOT             | 300   | 239.29          |
| 2 |                | _mixed_640_480_0.5_36G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | FPN-resnet18_  | pt_FPN-resnet18_cov    | 300   | 401.98          |
| 3 | covid19-seg_pt | id19-seg_352_352_22.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | Inceptio       | tf_inceptionresnetv2_i | 300   | 298.63          |
| 4 | n_resnet_v2_tf | magenet_299_299_26.35G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inception           | 300   | 2185.25         |
| 5 | nception_v1_tf | v1_imagenet_224_224_3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | i              | tf_inceptionv2_        | 300   | 672.69          |
| 6 | nception_v2_tf | imagenet_224_224_3.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | I              | tf_inceptionv3_i       | 300   | 691.41          |
| 7 | nception_v3_tf | magenet_299_299_11.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | tf_inceptionv3_ima     | 300   | 692.45          |
| 8 | _pruned_0_2_tf | genet_299_299_0.2_9.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | inception_v3   | tf_inceptionv3_ima     | 300   | 809.48          |
| 9 | _pruned_0_4_tf | genet_299_299_0.4_6.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | In             | tf2_inceptionv3_       | 300   | 709.43          |
| 0 | ception_v3_tf2 | imagenet_299_299_11.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | pt_inceptionv3_        | 300   | 690.40          |
| 1 | nception_v3_pt | imagenet_299_299_11.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_i       | 300   | 754.35          |
| 2 | _pruned_0_3_pt | magenet_299_299_0.3_8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 300   | 811.57          |
| 3 | _pruned_0_4_pt | genet_299_299_0.4_6.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 300   | 911.55          |
| 4 | _pruned_0_5_pt | genet_299_299_0.5_5.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | inception_v3   | pt_inceptionv3_ima     | 300   | 1045.46         |
| 5 | _pruned_0_6_pt | genet_299_299_0.6_4.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | I              | tf_inceptionv4_i       | 300   | 324.68          |
| 6 | nception_v4_tf | magenet_299_299_24.55G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | medica         | tf2_2d-une             | 300   | 1972.68         |
| 7 | l_seg_cell_tf2 | t_nuclei_128_128_5.31G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | mlperf_s       | tf_mlperf_resnet       | 300   | 25.84           |
| 8 | sd_resnet34_tf | 34_coco_1200_1200_433G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 2 | mlpe           | tf_mlperf_resnet50_    | 300   | 1012.03         |
| 9 | rf_resnet50_tf | imagenet_224_224_8.19G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilene       | tf_mobilenetEdge0.75   | 300   | 4716.44         |
| 0 | t_edge_0_75_tf | _imagenet_224_224_624M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf_mobilenetEdge1.0    | 300   | 3834.40         |
| 1 | et_edge_1_0_tf | _imagenet_224_224_990M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf2_mobilenetv1_       | 300   | 5305.78         |
| 2 | et_1_0_224_tf2 | imagenet_224_224_1.15G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet_     | tf_mobilenetv1_0.2     | 300   | 18873.80        |
| 3 | v1_0_25_128_tf | 5_imagenet_128_128_27M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet      | tf_mobilenetv1_0.5     | 300   | 12862.50        |
| 4 | _v1_0_5_160_tf | _imagenet_160_160_150M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilenet      | tf_mobilenetv1_1.0_    | 300   | 5305.64         |
| 5 | _v1_1_0_224_tf | imagenet_224_224_1.14G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf                     | 300   | 5432.80         |
| 6 | et_v1_1_0_224_ | _mobilenetv1_1.0_image |       |                 |
|   | pruned_0_11_tf | net_224_224_0.11_1.02G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mobilen        | tf_mobilenetv1_1.0_im  | 300   | 5460.44         |
| 7 | et_v1_1_0_224_ | agenet_224_224_0.12_1G |       |                 |
|   | pruned_0_12_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | MT-res         | pt_MT-resnet1          | 300   | 511.94          |
| 8 | net18_mixed_pt | 8_mixed_320_512_13.65G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 3 | mu             | pt_multitaskv          | 300   | 291.27          |
| 9 | lti_task_v3_pt | 3_mixed_320_512_25.44G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_dept       | p                      | 300   | 2887.10         |
| 0 | hwise_res50_pt | t_OFA-depthwise-res50_ |       |                 |
|   |                | imagenet_176_176_2.49G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_r          | pt_OFA-rc              | 300   | 65.53           |
| 1 | can_latency_pt | an_DIV2K_360_640_45.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_resnet     | pt_OFA-resnet50_       | 300   | 597.57          |
| 2 | 50_baseline_pt | imagenet_224_224_15.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 300   | 913.97          |
| 3 | pruned_0_45_pt | enet_224_224_0.45_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 300   | 1138.00         |
| 4 | pruned_0_60_pt | enet_224_224_0.60_6.0G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_resnet50_  | pt_OFA-resnet50_imag   | 300   | 1724.06         |
| 5 | pruned_0_74_pt | enet_192_192_0.74_3.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_re         | pt_OFA-resnet50_imag   | 300   | 2922.00         |
| 6 | snet50_0_9B_pt | enet_160_160_0.88_1.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_yolo_pt    | pt_OFA-yo              | 300   | 178.70          |
| 7 |                | lo_coco_640_640_48.88G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_yolo_      | pt_OFA-yolo_c          | 300   | 223.19          |
| 8 | pruned_0_30_pt | oco_640_640_0.3_34.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 4 | ofa_yolo_      | pt_OFA-yolo_c          | 300   | 283.17          |
| 9 | pruned_0_50_pt | oco_640_640_0.6_24.62G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | person-orienta | pt_person-or           | 300   | 11379.50        |
| 0 | tion_pruned_pt | ientation_224_112_558M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | perso          | pt_personreid-res50_m  | 300   | 1611.62         |
| 1 | nreid-res50_pt | arket1501_256_128_5.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pe             | pt_                    | 300   | 2123.04         |
| 2 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_4_pt | t1501_256_128_0.4_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pe             | pt_                    | 300   | 2323.59         |
| 3 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_5_pt | t1501_256_128_0.5_2.7G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pe             | pt_                    | 300   | 2629.79         |
| 4 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_6_pt | t1501_256_128_0.6_2.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pe             | pt_                    | 300   | 2954.94         |
| 5 | rsonreid_res50 | personreid-res50_marke |       |                 |
|   | _pruned_0_7_pt | t1501_256_128_0.7_1.6G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | perso          | pt_personreid-res18_   | 300   | 6648.41         |
| 6 | nreid-res18_pt | market1501_176_80_1.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pmg_pt         | pt_                    | 300   | 1989.61         |
| 7 |                | pmg_rp2k_224_224_2.28G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pointpai       | pt_pointpain           | 300   | 20.11           |
| 8 | nting_nuscenes | ting_nuscenes_126G_2.5 |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 5 | pointpi        | pt_pointpillars_       | 300   | 39.93           |
| 9 | llars_nuscenes | nuscenes_40000_64_108G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | rcan_pruned_tf | tf_rcan_DIV            | 300   | 80.06           |
| 0 |                | 2K_360_640_0.98_86.95G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | re             | tf_refin               | 300   | 138.34          |
| 1 | finedet_VOC_tf | edet_VOC_320_320_81.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | Refin          | tf_RefineDet-Medi      | 300   | 139.50          |
| 2 | eDet-Medical_E | cal_EDD_320_320_81.28G |       |                 |
|   | DD_baseline_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | RefineD        | tf_RefineDet-Medical_  | 1300  | 248.60          |
| 3 | et-Medical_EDD | EDD_320_320_0.5_41.42G |       |                 |
|   | _pruned_0_5_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | RefineDe       | tf_RefineDet-Medical_E | 300   | 433.45          |
| 4 | t-Medical_EDD_ | DD_320_320_0.75_20.54G |       |                 |
|   | pruned_0_75_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | RefineDe       | tf_RefineDet-Medical_E | 300   | 698.40          |
| 5 | t-Medical_EDD_ | DD_320_320_0.85_12.32G |       |                 |
|   | pruned_0_85_tf |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | RefineDet-     | tf_RefineDet-Medical_  | 300   | 789.95          |
| 6 | Medical_EDD_tf | EDD_320_320_0.88_9.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | r              | tf_resnetv1_50_        | 300   | 1175.61         |
| 7 | esnet_v1_50_tf | imagenet_224_224_6.97G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet_v1_50_  | tf_resnetv1_50_imag    | 300   | 1521.60         |
| 8 | pruned_0_38_tf | enet_224_224_0.38_4.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 6 | resnet_v1_50_  | tf_resnetv1_50_image   | 300   | 2031.64         |
| 9 | pruned_0_65_tf | net_224_224_0.65_2.45G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | re             | tf_resnetv1_101_       | 300   | 610.35          |
| 0 | snet_v1_101_tf | imagenet_224_224_14.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | re             | tf_resnetv1_152_i      | 300   | 407.06          |
| 1 | snet_v1_152_tf | magenet_224_224_21.83G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50_tf2   | tf2_resnet50_          | 300   | 1174.21         |
| 2 |                | imagenet_224_224_7.76G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50_pt    | pt_resnet50            | 300   | 1012.40         |
| 3 |                | _imagenet_224_224_8.2G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50       | pt_resnet50_ima        | 300   | 1214.86         |
| 4 | _pruned_0_3_pt | genet_224_224_0.3_5.8G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50       | pt_resnet50_ima        | 300   | 1321.38         |
| 5 | _pruned_0_4_pt | genet_224_224_0.4_4.9G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50       | pt_resnet50_ima        | 300   | 1458.08         |
| 6 | _pruned_0_5_pt | genet_224_224_0.5_4.1G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50       | pt_resnet50_ima        | 300   | 1654.50         |
| 7 | _pruned_0_6_pt | genet_224_224_0.6_3.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | resnet50       | pt_resnet50_ima        | 300   | 1845.19         |
| 8 | _pruned_0_7_pt | genet_224_224_0.7_2.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 7 | salsanext_pt   | p                      | 300   | 153.16          |
| 9 |                | t_salsanext_semantic-k |       |                 |
|   |                | itti_64_2048_0.6_20.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | s              | pt                     | 300   | 58.41           |
| 0 | alsanext_v2_pt | _salsanextv2_semantic- |       |                 |
|   |                | kitti_64_2048_0.75_32G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | semantic       | tf2_erfnet_c           | 300   | 90.03           |
| 1 | _seg_citys_tf2 | ityscapes_512_1024_54G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | SemanticFPN    | pt_SemanticFPN_        | 300   | 803.50          |
| 2 | _cityscapes_pt | cityscapes_256_512_10G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | SemanticFPN_   | pt_Sema                | 300   | 228.29          |
| 3 | Mobilenetv2_pt | nticFPN-mobilenetv2_ci |       |                 |
|   |                | tyscapes_512_1024_5.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | SESR_S_pt      | pt_SESR                | 300   | 282.56          |
| 4 |                | -S_DIV2K_360_640_7.48G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | SqueezeNet_pt  | pt_squeezenet_i        | 300   | 6398.41         |
| 5 |                | magenet_224_224_351.7M |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | ssd_incept     | tf_ssdinceptio         | 300   | 329.92          |
| 6 | ion_v2_coco_tf | nv2_coco_300_300_9.62G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | ssd_mobile     | tf_ssdmobilene         | 300   | 2115.80         |
| 7 | net_v1_coco_tf | tv1_coco_300_300_2.47G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | ssd_mobile     | tf_ssdmobilene         | 300   | 1458.46         |
| 8 | net_v2_coco_tf | tv2_coco_300_300_3.75G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 8 | ssd_resnet_    | tf_ssdresnet50v1_f     | 300   | 58.99           |
| 9 | 50_fpn_coco_tf | pn_coco_640_640_178.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssdlite_mobile | tf_ssdlite_mobilen     | 300   | 2061.95         |
| 0 | net_v2_coco_tf | etv2_coco_300_300_1.5G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ssr_pt         | pt_                    | 300   | 62.02           |
| 1 |                | SSR_CVC_256_256_39.72G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | tsd_yolox_pt   | pt_yo                  | 300   | 131.89          |
| 2 |                | lox_TT100K_640_640_73G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | ultrafast_pt   | pt_ultrafa             | 300   | 474.24          |
| 3 |                | st_CULane_288_800_8.4G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | un             | pt_unet_               | 300   | 136.80          |
| 4 | et_chaos-CT_pt | chaos-CT_512_512_23.3G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | chen_col       | pt_vehi                | 300   | 2645.11         |
| 5 | or_resnet18_pt | cle-color-classificati |       |                 |
|   |                | on_color_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | vehicle_ma     | pt_vehicl              | 300   | 2634.76         |
| 6 | ke_resnet18_pt | e-make-classification_ |       |                 |
|   |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | vehicle_ty     | pt_vehicl              | 300   | 2642.15         |
| 7 | pe_resnet18_pt | e-type-classification_ |       |                 |
|   |                | CompCars_224_224_3.63G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | vgg_16_tf      | tf_vgg16_i             | 300   | 283.80          |
| 8 |                | magenet_224_224_30.96G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 9 | vgg_16_        | tf_vgg16_imagen        | 300   | 540.89          |
| 9 | pruned_0_43_tf | et_224_224_0.43_17.67G |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_16         | tf_vgg16_image         | 300   | 584.30          |
| 0 | _pruned_0_5_tf | net_224_224_0.5_15.64G |       |                 |
| 0 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | vgg_19_tf      | tf_vgg19_i             | 300   | 238.41          |
| 0 |                | magenet_224_224_39.28G |       |                 |
| 1 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | yolov3_voc_tf  | tf_yol                 | 300   | 152.75          |
| 0 |                | ov3_voc_416_416_65.63G |       |                 |
| 2 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+
| 1 | y              | tf2_yol                | 300   | 150.59          |
| 0 | olov3_coco_tf2 | ov3_coco_416_416_65.9G |       |                 |
| 3 |                |                        |       |                 |
+---+----------------+------------------------+-------+-----------------+

.. raw:: html

   </details>

Performance on U200
~~~~~~~~~~~~~~~~~~~

Measured with Vitis AI 2.5 and Vitis AI Library 2.5

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``Alveo U200`` board with 2
DPUCADF8H kernels running at 300Mhz:

+---+------+----------------------+-----------------+-------------------+
| N | M    | Name                 | E2E latency     | E2E throughput    |
| o | odel |                      | (ms) Thread num | -fps(Multi        |
| . |      |                      | =20             | Thread)           |
+===+======+======================+=================+===================+
| 1 | resn | cf_resnet50_i        | 3.8             | 1054              |
|   | et50 | magenet_224_224_7.7G |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 2 | Ince | tf_inceptionv1       | 2.2             | 1834              |
|   | ptio | _imagenet_224_224_3G |                 |                   |
|   | n_v1 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 3 | Ince | tf_inceptionv3_ima   | 18.4            | 218               |
|   | ptio | genet_299_299_11.45G |                 |                   |
|   | n_v3 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 4 | res  | tf_resnetv1_50_im    | 4.2             | 947               |
|   | netv | agenet_224_224_6.97G |                 |                   |
|   | 1_50 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 5 | resn | tf_resnetv1_101_im   | 8.5             | 472               |
|   | etv1 | agenet_224_224_14.4G |                 |                   |
|   | _101 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 6 | resn | tf_resnetv1_152_ima  | 12.7            | 316               |
|   | etv1 | genet_224_224_21.83G |                 |                   |
|   | _152 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``Alveo U200`` board with 2
DPUCADX8G kernels running at 350Mhz with xilinx_u200_xdma_201830_2
shell:

+---+---------+----------------+-----------+-------------+-------------+
| N | Model   | Name           | E2E       | E2E         | E2E         |
| o |         |                | latency   | throughput  | throughput  |
| . |         |                | (ms)      | -fps(Single | -fps(Multi  |
|   |         |                | Thread    | Thread)     | Thread)     |
|   |         |                | num =1    |             |             |
+===+=========+================+===========+=============+=============+
| 1 | r       | cf_re          | 2.13      | 470.6       | 561.3       |
|   | esnet50 | snet50_imagene |           |             |             |
|   |         | t_224_224_7.7G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 2 | r       | cf_res         | 2.08      | 481         | 1157.8      |
|   | esnet18 | net18_imagenet |           |             |             |
|   |         | _224_224_3.65G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 3 | Incep   | cf_incept      | 2.39      | 418.5       | 1449.4      |
|   | tion_v1 | ionv1_imagenet |           |             |             |
|   |         | _224_224_3.16G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 4 | Incep   | cf_inc         | 2.11      | 475.1       | 1129.2      |
|   | tion_v2 | eptionv2_image |           |             |             |
|   |         | net_224_224_4G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 5 | Incep   | cf_incept      | 15.67     | 63.8        | 371.6       |
|   | tion_v3 | ionv3_imagenet |           |             |             |
|   |         | _299_299_11.4G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 6 | Incep   | cf_incept      | 10.77     | 92.8        | 221.2       |
|   | tion_v4 | ionv4_imagenet |           |             |             |
|   |         | _299_299_24.5G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 7 | Squ     | cf_sq          | 10.99     | 91          | 1157.1      |
|   | eezeNet | ueeze_imagenet |           |             |             |
|   |         | _227_227_0.76G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 8 | de      | cf_            | 8.69      | 115.1       | 667.9       |
|   | nsebox_ | densebox_wider |           |             |             |
|   | 320_320 | _320_320_0.49G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 9 | yol     | dk_yolov3_bdd  | 14.53     | 68.8        | 75.9        |
|   | ov3_bdd | _288_512_53.7G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 1 | yol     | dk_yolov3_voc_ | 19.90     | 50.3        | 82.1        |
| 0 | ov3_voc | 416_416_65.42G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+

.. raw:: html

   </details>

Performance on U250
~~~~~~~~~~~~~~~~~~~

Measured with Vitis AI 2.5 and Vitis AI Library 2.5

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``Alveo U250`` board with 4
DPUCADF8H kernels running at 300Mhz:

+---+------+----------------------+-----------------+-------------------+
| N | M    | Name                 | E2E latency     | E2E throughput    |
| o | odel |                      | (ms) Thread num | -fps(Multi        |
| . |      |                      | =20             | Thread)           |
+===+======+======================+=================+===================+
| 1 | resn | cf_resnet50_i        | 1.94            | 2134.8            |
|   | et50 | magenet_224_224_7.7G |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 2 | Ince | tf_inceptionv1       | 1.10            | 3631.7            |
|   | ptio | _imagenet_224_224_3G |                 |                   |
|   | n_v1 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 3 | Ince | tf_inceptionv3_ima   | 9.20            | 434.9             |
|   | ptio | genet_299_299_11.45G |                 |                   |
|   | n_v3 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 4 | res  | tf_resnetv1_50_im    | 2.13            | 1881.6            |
|   | netv | agenet_224_224_6.97G |                 |                   |
|   | 1_50 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 5 | resn | tf_resnetv1_101_im   | 4.24            | 941.9             |
|   | etv1 | agenet_224_224_14.4G |                 |                   |
|   | _101 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+
| 6 | resn | tf_resnetv1_152_ima  | 6.35            | 630.3             |
|   | etv1 | genet_224_224_21.83G |                 |                   |
|   | _152 |                      |                 |                   |
+---+------+----------------------+-----------------+-------------------+

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``Alveo U250`` board with 4
DPUCADX8G kernels running at 350Mhz with xilinx_u250_xdma_201830_1
shell:

+---+---------+----------------+-----------+-------------+-------------+
| N | Model   | Name           | E2E       | E2E         | E2E         |
| o |         |                | latency   | throughput  | throughput  |
| . |         |                | (ms)      | -fps(Single | -fps(Multi  |
|   |         |                | Thread    | Thread)     | Thread)     |
|   |         |                | num =1    |             |             |
+===+=========+================+===========+=============+=============+
| 1 | r       | cf_re          | 1.68      | 595.5       | 1223.95     |
|   | esnet50 | snet50_imagene |           |             |             |
|   |         | t_224_224_7.7G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 2 | r       | cf_res         | 1.67      | 600.5       | 2422.5      |
|   | esnet18 | net18_imagenet |           |             |             |
|   |         | _224_224_3.65G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 3 | Incep   | cf_incept      | 1.93      | 517.1       | 4059.8      |
|   | tion_v1 | ionv1_imagenet |           |             |             |
|   |         | _224_224_3.16G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 4 | Incep   | cf_inc         | 1.65      | 607.8       | 23221       |
|   | tion_v2 | eptionv2_image |           |             |             |
|   |         | net_224_224_4G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 5 | Incep   | cf_incept      | 6.18      | 161.8       | 743.8       |
|   | tion_v3 | ionv3_imagenet |           |             |             |
|   |         | _299_299_11.4G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 6 | Incep   | cf_incept      | 5.77      | 173.4       | 452.4       |
|   | tion_v4 | ionv4_imagenet |           |             |             |
|   |         | _299_299_24.5G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 7 | Squ     | cf_sq          | 5.44      | 183.7       | 2349.7      |
|   | eezeNet | ueeze_imagenet |           |             |             |
|   |         | _227_227_0.76G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 8 | de      | cf_            | 7.43      | 167.2       | 898.5       |
|   | nsebox_ | densebox_wider |           |             |             |
|   | 320_320 | _320_320_0.49G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 9 | yol     | dk_yolov3_bdd  | 14.27     | 70.1        | 146.7       |
|   | ov3_bdd | _288_512_53.7G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+
| 1 | yol     | dk_yolov3_voc_ | 9.46      | 105.7       | 139.4       |
| 0 | ov3_voc | 416_416_65.42G |           |             |             |
+---+---------+----------------+-----------+-------------+-------------+

Note: For xilinx_u250_gen3x16_xdma_shell_3_1_202020_1 latest shell U250
xclbins, Alveo U250 board would be having only 3 DPUCADF8H kernels
instead of 4, thereby the performance numbers for 1 Alveo U250 board
with xilinx_u250_gen3x16_xdma_shell_3_1 shell xclbins would be 75% of
the above reported performance numbers which is for 4 DPUCADF8H kernels.

.. raw:: html

   </details>

Performance on Ultra96
~~~~~~~~~~~~~~~~~~~~~~

The performance number shown below was measured with the previous AI SDK
v2.0.4 on Ultra96 v1. The Vitis platform of Ultra96 v2 has not been
released yet. So the performance numbers are therefore not reported for
this Model Zoo release.

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``Ultra96`` board with a
``1 * B1600  @ 287MHz   V1.4.0`` DPU configuration:

**Note:** The original power supply of Ultra96 is not designed for high
performance AI workload. The board may occasionally hang to run few
models, When multi-thread is used. For such situations, ``NA`` is
specified in the following table.

+---+----------+-----------------+----------+-------------+-------------+
| N | Model    | Name            | E2E      | E2E         | E2E         |
| o |          |                 | latency  | throughput  | throughput  |
| . |          |                 | (ms)     | -fps(Single | -fps(Multi  |
|   |          |                 | Thread   | Thread)     | Thread)     |
|   |          |                 | num =1   |             |             |
+===+==========+=================+==========+=============+=============+
| 1 | resnet50 | cf_             | 30.8     | 32.4667     | 33.4667     |
|   |          | resnet50_imagen |          |             |             |
|   |          | et_224_224_7.7G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | Ince     | cf_ince         | 13.98    | 71.55       | 75.0667     |
|   | ption_v1 | ptionv1_imagene |          |             |             |
|   |          | t_224_224_3.16G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | Ince     | cf_i            | 17.16    | 58.2667     | 61.2833     |
|   | ption_v2 | nceptionv2_imag |          |             |             |
|   |          | enet_224_224_4G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 4 | Ince     | cf_ince         | 44.05    | 22.7        | 23.4333     |
|   | ption_v3 | ptionv3_imagene |          |             |             |
|   |          | t_299_299_11.4G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 5 | mobi     | cf_mobi         | 7.34     | 136.183     | NA          |
|   | leNet_v2 | lenetv2_imagene |          |             |             |
|   |          | t_224_224_0.59G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 6 | tf_      | tf_r            | 28.02    | 35.6833     | 36.6        |
|   | resnet50 | esnet50_imagene |          |             |             |
|   |          | t_224_224_6.97G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 7 | tf_ince  | tf_i            | 16.96    | 58.9667     | 61.2833     |
|   | ption_v1 | nceptionv1_imag |          |             |             |
|   |          | enet_224_224_3G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 8 | tf_mobi  | tf_mobi         | 10.17    | 98.3        | 104.25      |
|   | lenet_v2 | lenetv2_imagene |          |             |             |
|   |          | t_224_224_1.17G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 9 | ssd_     | cf              | 24.3     | 41.15       | 46.2        |
|   | adas_pru | _ssdadas_bdd_36 |          |             |             |
|   | ned_0.95 | 0_480_0.95_6.3G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | ss       | cf_ssdped       | 23.29    | 42.9333     | 50.8        |
| 0 | d_pedest | estrian_coco_36 |          |             |             |
|   | rian_pru | 0_640_0.97_5.9G |          |             |             |
|   | ned_0.97 |                 |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | ssd_tr   | c               | 35.5     | 28.1667     | 31.8        |
| 1 | affic_pr | f_ssdtraffic_36 |          |             |             |
|   | uned_0.9 | 0_480_0.9_11.6G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | ssd_mob  | cf_ss           | 60.79    | 16.45       | 27.8167     |
| 2 | ilnet_v2 | dmobilenetv2_bd |          |             |             |
|   |          | d_360_480_6.57G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | tf       | tf_ssd_voc      | 186.92   | 5.35        | 5.81667     |
| 3 | _ssd_voc | _300_300_64.81G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | densebox | c               | 4.17     | 239.883     | 334.167     |
| 4 | _320_320 | f_densebox_wide |          |             |             |
|   |          | r_320_320_0.49G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | densebox | c               | 8.55     | 117         | 167.2       |
| 5 | _360_640 | f_densebox_wide |          |             |             |
|   |          | r_360_640_1.11G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | yolov    | dk_yolov        | 22.79    | 43.8833     | 49.6833     |
| 6 | 3_adas_p | 3_cityscapes_25 |          |             |             |
|   | rune_0.9 | 6_512_0.9_5.46G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | yo       | dk_yolov3_voc   | 185.19   | 5.4         | 5.53        |
| 7 | lov3_voc | _416_416_65.42G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | tf_yo    | tf_yolov3_voc   | 199.34   | 5.01667     | 5.1         |
| 8 | lov3_voc | _416_416_65.63G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | refi     | cf_             | 66.37    | 15.0667     | NA          |
| 9 | nedet_pr | refinedet_coco_ |          |             |             |
|   | uned_0.8 | 360_480_0.8_25G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | refin    | cf_refi         | 32.17    | 31.0883     | 33.6667     |
| 0 | edet_pru | nedet_coco_360_ |          |             |             |
|   | ned_0.92 | 480_0.92_10.10G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | refin    | cf_ref          | 20.29    | 49.2833     | 55.25       |
| 1 | edet_pru | inedet_coco_360 |          |             |             |
|   | ned_0.96 | _480_0.96_5.08G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | FPN      | cf_fpn_cityscap | 36.34    | 27.5167     | NA          |
| 2 |          | es_256_512_8.9G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | VP       | cf_VPGnet       | 13.9     | 71.9333     | NA          |
| 3 | Gnet_pru | _caltechlane_48 |          |             |             |
|   | ned_0.99 | 0_640_0.99_2.5G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | SP-net   | cf_SP           | 3.82     | 261.55      | 277.4       |
| 4 |          | net_aichallenge |          |             |             |
|   |          | r_224_128_0.54G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | Ope      | cf_openpose_a   | 560.75   | 1.78333     | NA          |
| 5 | npose_pr | ichallenger_368 |          |             |             |
|   | uned_0.3 | _368_0.3_189.7G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | yo       | dk_yolov2_      | 118.11   | 8.46667     | 8.9         |
| 6 | lov2_voc | voc_448_448_34G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | yolov2   | dk_             | 37.5     | 26.6667     | 30.65       |
| 7 | _voc_pru | yolov2_voc_448_ |          |             |             |
|   | ned_0.66 | 448_0.66_11.56G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | yolov2   | dk              | 30.99    | 32.2667     | 38.35       |
| 8 | _voc_pru | _yolov2_voc_448 |          |             |             |
|   | ned_0.71 | _448_0.71_9.86G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | yolov2   | dk              | 26.29    | 38.03333    | 46.8333     |
| 9 | _voc_pru | _yolov2_voc_448 |          |             |             |
|   | ned_0.77 | _448_0.77_7.82G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | Ince     | cf_ince         | 88.76    | 11.2667     | 11.5333     |
| 0 | ption-v4 | ptionv4_imagene |          |             |             |
|   |          | t_299_299_24.5G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | Sq       | cf_             | 5.96     | 167.867     | 283.583     |
| 1 | ueezeNet | squeeze_imagene |          |             |             |
|   |          | t_227_227_0.76G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | face_    | cf_landmark_cel | 2.95     | 339.183     | 347.633     |
| 2 | landmark | eba_96_72_0.14G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | reid     | c               | 6.28     | 159.15      | 166.633     |
| 3 |          | f_reid_market15 |          |             |             |
|   |          | 01_160_80_0.95G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | yo       | dk_yolov3_bd    | 193.55   | 5.16667     | 5.31667     |
| 4 | lov3_bdd | d_288_512_53.7G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | tf_mobi  | tf_mobi         | 5.97     | 167.567     | 186.55      |
| 5 | lenet_v1 | lenetv1_imagene |          |             |             |
|   |          | t_224_224_1.14G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | resnet18 | cf_r            | 13.47    | 74.2167     | 77.8167     |
| 6 |          | esnet18_imagene |          |             |             |
|   |          | t_224_224_3.65G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | resne    | tf              | 97.72    | 10.2333     | 10.3833     |
| 7 | t18_wide | _resnet18_image |          |             |             |
|   |          | net_224_224_28G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+

.. raw:: html

   </details>

Performance on ZCU102 (0432055-04)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This version of ZCU102 is out of stock. The performance number shown
below was measured with the previous AI SDK v2.0.4. Now this form has
stopped updating. So the performance numbers are therefore not reported
for this Model Zoo release.

.. raw:: html

   <details>

Click here to view details

The following table lists the performance number including end-to-end
throughput and latency for each model on the ``ZCU102 (0432055-04)``
board with a ``3 * B4096  @ 287MHz   V1.4.0`` DPU configuration:

+---+----------+-----------------+----------+-------------+-------------+
| N | Model    | Name            | E2E      | E2E         | E2E         |
| o |          |                 | latency  | throughput  | throughput  |
| . |          |                 | (ms)     | -fps(Single | -fps(Multi  |
|   |          |                 | Thread   | Thread)     | Thread)     |
|   |          |                 | num =1   |             |             |
+===+==========+=================+==========+=============+=============+
| 1 | resnet50 | cf_             | 12.85    | 77.8        | 179.3       |
|   |          | resnet50_imagen |          |             |             |
|   |          | et_224_224_7.7G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | Ince     | cf_ince         | 5.47     | 182.683     | 485.533     |
|   | ption_v1 | ptionv1_imagene |          |             |             |
|   |          | t_224_224_3.16G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | Ince     | cf_i            | 6.76     | 147.933     | 373.267     |
|   | ption_v2 | nceptionv2_imag |          |             |             |
|   |          | enet_224_224_4G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 4 | Ince     | cf_ince         | 17       | 58.8333     | 155.4       |
|   | ption_v3 | ptionv3_imagene |          |             |             |
|   |          | t_299_299_11.4G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 5 | mobi     | cf_mobi         | 4.09     | 244.617     | 638.067     |
|   | leNet_v2 | lenetv2_imagene |          |             |             |
|   |          | t_224_224_0.59G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 6 | tf_      | tf_r            | 11.94    | 83.7833     | 191.417     |
|   | resnet50 | esnet50_imagene |          |             |             |
|   |          | t_224_224_6.97G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 7 | tf_ince  | tf_i            | 6.72     | 148.867     | 358.283     |
|   | ption_v1 | nceptionv1_imag |          |             |             |
|   |          | enet_224_224_3G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 8 | tf_mobi  | tf_mobi         | 5.46     | 183.117     | 458.65      |
|   | lenet_v2 | lenetv2_imagene |          |             |             |
|   |          | t_224_224_1.17G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 9 | ssd_     | cf              | 11.33    | 88.2667     | 320.5       |
|   | adas_pru | _ssdadas_bdd_36 |          |             |             |
|   | ned_0.95 | 0_480_0.95_6.3G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | ss       | cf_ssdped       | 12.96    | 77.1833     | 314.717     |
| 0 | d_pedest | estrian_coco_36 |          |             |             |
|   | rian_pru | 0_640_0.97_5.9G |          |             |             |
|   | ned_0.97 |                 |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | ssd_tr   | c               | 17.49    | 57.1833     | 218.183     |
| 1 | affic_pr | f_ssdtraffic_36 |          |             |             |
|   | uned_0.9 | 0_480_0.9_11.6G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | ssd_mob  | cf_ss           | 24.21    | 41.3        | 141.233     |
| 2 | ilnet_v2 | dmobilenetv2_bd |          |             |             |
|   |          | d_360_480_6.57G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | tf       | tf_ssd_voc      | 69.28    | 14.4333     | 46.7833     |
| 3 | _ssd_voc | _300_300_64.81G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | densebox | c               | 2.43     | 412.183     | 1416.63     |
| 4 | _320_320 | f_densebox_wide |          |             |             |
|   |          | r_320_320_0.49G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | densebox | c               | 5.01     | 199.717     | 719.75      |
| 5 | _360_640 | f_densebox_wide |          |             |             |
|   |          | r_360_640_1.11G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | yolov    | dk_yolov        | 11.09    | 90.1667     | 259.65      |
| 6 | 3_adas_p | 3_cityscapes_25 |          |             |             |
|   | rune_0.9 | 6_512_0.9_5.46G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | yo       | dk_yolov3_voc   | 70.51    | 14.1833     | 44.4        |
| 7 | lov3_voc | _416_416_65.42G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | tf_yo    | tf_yolov3_voc   | 70.75    | 14.1333     | 44.0167     |
| 8 | lov3_voc | _416_416_65.63G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 1 | refi     | cf_             | 29.91    | 33.4333     | 109.067     |
| 9 | nedet_pr | refinedet_coco_ |          |             |             |
|   | uned_0.8 | 360_480_0.8_25G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | refin    | cf_refi         | 15.39    | 64.9667     | 216.317     |
| 0 | edet_pru | nedet_coco_360_ |          |             |             |
|   | ned_0.92 | 480_0.92_10.10G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | refin    | cf_ref          | 11.04    | 90.5833     | 312         |
| 1 | edet_pru | inedet_coco_360 |          |             |             |
|   | ned_0.96 | _480_0.96_5.08G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | FPN      | cf_fpn_cityscap | 16.58    | 60.3        | 203.867     |
| 2 |          | es_256_512_8.9G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | VP       | cf_VPGnet       | 9.44     | 105.9       | 424.667     |
| 3 | Gnet_pru | _caltechlane_48 |          |             |             |
|   | ned_0.99 | 0_640_0.99_2.5G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | SP-net   | cf_SP           | 1.73     | 579.067     | 1620.67     |
| 4 |          | net_aichallenge |          |             |             |
|   |          | r_224_128_0.54G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | Ope      | cf_openpose_a   | 279.07   | 3.58333     | 38.5        |
| 5 | npose_pr | ichallenger_368 |          |             |             |
|   | uned_0.3 | _368_0.3_189.7G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | yo       | dk_yolov2_      | 39.76    | 25.15       | 86.35       |
| 6 | lov2_voc | voc_448_448_34G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | yolov2   | dk_             | 18.42    | 54.2833     | 211.217     |
| 7 | _voc_pru | yolov2_voc_448_ |          |             |             |
|   | ned_0.66 | 448_0.66_11.56G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | yolov2   | dk              | 16.42    | 60.9167     | 242.433     |
| 8 | _voc_pru | _yolov2_voc_448 |          |             |             |
|   | ned_0.71 | _448_0.71_9.86G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 2 | yolov2   | dk              | 14.46    | 69.1667     | 286.733     |
| 9 | _voc_pru | _yolov2_voc_448 |          |             |             |
|   | ned_0.77 | _448_0.77_7.82G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | Ince     | cf_ince         | 34.25    | 29.2        | 84.25       |
| 0 | ption-v4 | ptionv4_imagene |          |             |             |
|   |          | t_299_299_24.5G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | Sq       | cf_             | 3.6      | 277.65      | 1080.77     |
| 1 | ueezeNet | squeeze_imagene |          |             |             |
|   |          | t_227_227_0.76G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | face_    | cf_landmark_cel | 1.13     | 885.033     | 1623.3      |
| 2 | landmark | eba_96_72_0.14G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | reid     | c               | 2.67     | 375         | 773.533     |
| 3 |          | f_reid_marketcu |          |             |             |
|   |          | hk_160_80_0.95G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | yo       | dk_yolov3_bd    | 73.89    | 13.5333     | 42.8833     |
| 4 | lov3_bdd | d_288_512_53.7G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | tf_mobi  | tf_mobi         | 3.2      | 312.067     | 875.967     |
| 5 | lenet_v1 | lenetv1_imagene |          |             |             |
|   |          | t_224_224_1.14G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | resnet18 | cf_r            | 5.1      | 195.95      | 524.433     |
| 6 |          | esnet18_imagene |          |             |             |
|   |          | t_224_224_3.65G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+
| 3 | resne    | tf              | 33.28    | 30.05       | 83.4167     |
| 7 | t18_wide | _resnet18_image |          |             |             |
|   |          | net_224_224_28G |          |             |             |
+---+----------+-----------------+----------+-------------+-------------+

.. raw:: html

   </details>

Contributing
------------

We welcome community contributions. When contributing to this
repository, first discuss the change you wish to make via:

-  `GitHub Issues <https://github.com/Xilinx/Vitis-AI/issues>`__
-  `Forum <https://forums.xilinx.com/t5/AI-and-Vitis-AI/bd-p/AI>`__
-  Email

You can also submit a pull request with details on how to improve the
product. Prior to submitting your pull request, ensure that you can
build the product and run all the demos with your patch. In case of a
larger feature, provide a relevant demo.

.. raw:: html

   <hr/>
