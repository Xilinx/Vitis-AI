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

.. _workflow-model-zoo:

Introduction
============

The Vitis |trade| AI Model Zoo which is incorporated into the Vitis AI repository includes optimized deep learning models to speed up the deployment of deep learning inference on Xilinx™ platforms. These models cover different applications, including but not limited to ADAS/AD, medical, video surveillance, robotics, data center, etc. You can get started with these free pre-trained models to enjoy the benefits of deep learning acceleration. 

Model Zoo Details & Performance
-------------------------------

All the models in the Model Zoo have been deployed on Xilinx hardware with `Vitis AI <https://github.com/Xilinx/Vitis-AI>`__ and `Vitis AI Library <https://github.com/Xilinx/Vitis-AI/tree/master/examples/Vitis-AI-Library>`__. The performance number including end-to-end throughput and latency for
each model on various boards with different DPU configurations.

To make the job of using the Model Zoo a little easier, we have provided a downloadable spreadsheet that incorporates key data about the Model Zoo models. The spreadsheet and tables include comprehensive information about all models, including application, framework, input size, computation Flops as well as float and quantized accuracy.  You can download that spreadsheet :download:`here <reference/ModelZoo_VAI3.0_Github.xlsx>`.

.. raw:: html 

    <a href="reference/ModelZoo_VAI3.0_Github_web.mht">Also, you can view the Model Zoo Details & Performance table online.</a><br><br>

.. note:: The model performance number listed in these tables was verified using Vitis AI v3.0 and the Vitis AI Lirary v3.0. For each platform, specific DPU configurations are used and are highlighted in the header of the table. Vitis AI and Vitis AI Library can be downloaded for free from `Vitis AI Github <https://github.com/Xilinx/Vitis-AI>`__ and `Vitis AI Library Github <https://github.com/Xilinx/Vitis-AI/tree/master/examples/Vitis-AI-Library>`__. We will continue to improve the performance with Vitis AI. The performance number reported here is subject to change in the near future. 


Model Filename Rules
--------------------

When downloading the models from the Model Zoo, it may be important to you to understand the nomenclature used for each file.  Here is an explaination:

Model name: ``F_M_(D)_H_W_(P)_C_V`` \* ``F`` specifies training framework: ``tf`` is Tensorflow 1.x, ``tf2`` is Tensorflow 2.x, ``pt`` is PyTorch \* ``M`` specifies the model \* ``D`` specifies the dataset. It is optional depending on whether the dataset is public or private \* ``H`` specifies the height of input data \* ``W`` specifies the width of input data \* ``P`` specifies the pruning ratio, it means how much computation is reduced. It is optional depending on whether the model is pruned or not \* ``C`` specifies the computation of the model: how many Gops per image \* ``V`` specifies the version of Vitis-AI 

For example, ``pt_fadnet_sceneflow_576_960_0.65_154G_2.5`` is ``FADNet`` model trained with ``Pytorch`` using ``SceneFlow`` dataset, input size is ``576*960``, ``65%`` pruned, the computation per image is ``154 Gops`` and Vitis-AI version is ``2.5``.

Model Index
-----------

-  Computation OPS in the table are counted as FLOPs
-  Float & Quantized Accuracy unless otherwise specified, the default
   refers to top1 or top1/top5
-  The models found in the Vitis AI repository are for Xilinx SoC and Alveo targets.  For details and to download AMD Epyc CPU models, please refer to
   `UIF <https://github.com/amd/UIF>`__


.. raw:: html

   </details>

Automated Download Script
~~~~~~~~~~~~~~~~~~~~~~~~~

With downloader.py, you could quickly find the model you are interested in and specify a version to download it immediately. Please make sure that downloader.py and ‘/model_zoo/model-list’ folder are at the same level directory.

::

   python3  downloader.py  

Step1: You need input framework and model name keyword. Use space divide. If input ``all`` you will get list of all models.

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

Then you could choose it and input the number, the specified version of model will be automatically downloaded to the current directory. 

In addition, if you need download all models on all platforms at once, you just need enter number in the order indicated by the tips of Step 1/2/3 (select: all -> 0 -> 0).

Model Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

Download and extract the model archive to your working area on the local hard disk. For details on the various models, download link and MD5 checksum for the zip file of each model, refer to the subdirectory ‘/model_zoo/model-list’ within the Github repository.

Tensorflow Model Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a Tensorflow model, you should see the following directory structure:

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
       

.. note:: For more information on Vitis-AI Quantizer such as ``vai_q_tensorflow`` and ``vai_q_pytorch``, please see the `Vitis AI User Guide <https://docs.xilinx.com/r/en-US/ug1414-vitis-ai>`__.



For more information about DPU, see `DPU IP Product
Guide <https://www.xilinx.com/cgi-bin/docs/ipdoc?c=dpu;v=latest;d=pg338-dpu.pdf>`__.

For RNN models such as NLP, please refer to
`DPU-for-RNN <https://github.com/Xilinx/Vitis-AI/blob/master/demo/DPU-for-RNN>`__
for dpu specification information.

Besides, for Transformer demos such as ViT, Bert-base you could refer to
`Transformer <https://github.com/Xilinx/Vitis-AI/tree/master/examples/Transformer>`__.



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


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim: