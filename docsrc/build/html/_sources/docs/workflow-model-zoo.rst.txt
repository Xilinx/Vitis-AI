.. _workflow-model-zoo:

Vitis AI Model Zoo
==================

The Vitis |trade| AI Model Zoo, incorporated into the Vitis AI repository, includes optimized deep learning models to speed up the deployment of deep learning inference on Xilinx |reg| platforms. These models cover different applications, including but not limited to ADAS/AD, medical, video surveillance, robotics, data center, and so on. You can get started with these free pre-trained models to enjoy the benefits of deep learning acceleration.

Model Zoo Details and Performance
---------------------------------

All the models in the Model Zoo are deployed on Xilinx hardware with `Vitis AI <https://github.com/Xilinx/Vitis-AI>`__ and the `Vitis AI Library <https://github.com/Xilinx/Vitis-AI/tree/v3.0/examples/vai_library>`__. The performance benchmark data includes end-to-end throughput and latency for each model, targeting various boards with varied DPU configurations.

To make the job of using the Model Zoo a little easier, we have provided a downloadable spreadsheet and an online table that incorporates key data about the Model Zoo models. The spreadsheet and tables include comprehensive information about all models, including links to the original papers and datasets, source framework, input size, computational cost (GOPs), and float and quantized accuracy. **You can download the spreadsheet** :download:`here <reference/ModelZoo_VAI3.0_Github.xlsx>`.

.. raw:: html

    <a href="reference/ModelZoo_VAI3.0_Github_web.htm"><h4>Click here to view the Model Zoo Details & Performance table online.</h4></a><br><br>

.. note:: Please note that if the models are marked as "Non-Commercial Use Only", users must comply with this `AMD license agreement <https://github.com/Xilinx/Vitis-AI/blob/master/model_zoo/Xilinx-license-agreement-for-non-commercial-models.md>`__ 

.. note:: The model performance benchmarks listed in these tables are verified using Vitis AI v3.0 and Vitis AI Library v3.0. For each platform, specific DPU configurations are used and highlighted in the table's header. Free download of Vitis AI and Vitis AI Library from `Vitis AI Github <https://github.com/Xilinx/Vitis-AI>`__ and `Vitis AI Library Github <https://github.com/Xilinx/Vitis-AI/tree/v3.0/examples/vai_library>`__.

.. note:: Unless otherwise specified, the benchmarks for all models can be assumed to employ the maximum number of channels (i.e., for benchmarking, the images used for test have three color channels if the specified input dimensions are 299*299*3 (HWC)).



Model File Nomenclature
-----------------------

When downloading and using models from the Model Zoo, it will be important to you to understand the nomenclature used for each file.

Model File Nomenclature Decoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xilinx Model Zoo file names assume the format: `F_M_(D)_H_W_(P)_C_V`, where:

- `F` specifies the training framework: `tf` is TensorFlow 1.x, `tf2` is TensorFlow 2.x, `pt` is PyTorch

- `M` specifies the industry/base name of the model

- `D` specifies the public dataset used to train the model.  This field is not present if the model was trained using private datasets

- `H` specifies the height of the input tensor to the first input layer

- `W` specifies the width of the input tensor to the first input layer

- `P` specifies the pruning ratio (percentage computational complexity reduction from the base model). This field is present only if the model has been pruned

- `C` specifies the computational cost of the model for deployment in GOPs (billion quantized operations) per image

- `V` specifies the version of Vitis-AI in which the model was deployed

For example, `pt_inceptionv3_imagenet_299_299_0.6_4.5G_3.0` is the `inception v3` model trained with `PyTorch` using the `ImageNet` dataset, the input size for the network is `299*299`, `60%` pruned, the computational cost per image is `4.5 G FLOPs` and the Vitis AI version for that model is `3.0`.

Model Download
--------------

.. note:: Each model is associated with a .yaml file encapsulating the download link and MD5 checksum for a tar.gz file. These YAML files are in the Vitis AI repository ``/model_zoo/model-list``. There is a separate tar.gz file for each specific target platform. A simple way to download an individual model is to use the URLs provided in the .yaml file. This can be useful if you want to download and inspect the model outside a Python environment.

The download package includes the pre-compiled, pre-trained model, which you can leverage as a base reference (layer types, activation types, layer ordering) for your implementation or directly deploy that model on a Xilinx target.


Automated Download Script
~~~~~~~~~~~~~~~~~~~~~~~~~

The Vitis AI Model Zoo repository provides a Python ``/model_zoo/downloader.py`` that quickly downloads specific models.

.. note:: Ensure that the ``downloader.py`` script and the ``/model_zoo/model-list`` folder are at the same level in the directory hierarchy when executing this script.

1. Execute the script:

   .. code-block::

      python3  downloader.py

2. Input the framework keyword followed by a short-form version of the model name (if known) (example: resnet). Use a space as a separator (example: tf2 vgg16). If you input all, you will get a list of all models.

   The available framework keywords are listed here:

   **tf**: tensorflow1.x,  **tf2**: tensorflow2.x,  **pt**: pytorch,  **cf**: caffe,  **dk**: darknet, **all**: list all models

3. Select the desired target hardware platform for the version of the model you need.

   For example, after running downloader.py, input ``tf resnet`` and you will see a list of models that include the text `resnet`:

   .. code-block ::

      0:  all
      1:  tf_resnetv1_50_imagenet_224_224_6.97G_3.0
      2:  tf_resnetv1_101_imagenet_224_224_14.4G_3.0
      3:  tf_resnetv1_152_imagenet_224_224_21.83G_3.0
      ......


   Proceed by entering one of the numbers from the list.  As an example, if you input '1' the script will list all options that match your selection:

   .. code-block::

      0:  all
      1:  tf_resnetv1_50_imagenet_224_224_6.97G_3.0    GPU
      2:  resnet_v1_50_tf    ZCU102 & ZCU104 & KV260
      3:  resnet_v1_50_tf    VCK190
      4:  resnet_v1_50_tf    vck50006pe-DPUCVDX8H
      5:  resnet_v1_50_tf    vck50008pe-DPUCVDX8H-DWC
      6:  resnet_v1_50_tf    u50lv-DPUCAHX8H
      ......

   Proceed by entering one of the numbers from the list.  The specified version of the model will be downloaded automatically to the current directory. Entering '0' will download all models matching your search criteria.


Model Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have downloaded one or more models, you can extract the model archive into your selected workspace.

Tensorflow Model Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow models have the following directory structure:

::

    ├── code                            # Contains test code that can execute the model on the target and showcase model performance.
    │
    │
    ├── readme.md                       # Documents the environment requirements, data pre-processing requirements, and model information.
    │                                     Developers should refer to this to understand how to test the model with scripts.
    │
    ├── data                            # The dataset target directory that can be used for model verification and training.
    │                                     When test or training scripts run successfully, the dataset will be placed in this directory.
    │
    ├── quantized
    │   └── quantize_eval_model.pb      # Quantized model for evaluation.
    │
    └── float
        └── frozen.pb                   # The floating-point frozen model is used as the input to the quantizer.
                                          The naming of the protobuf file may differ from the model naming used in the model list.

Pytorch Model Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch models have the following directory structure:

::

    ├── code                            # Contains test and training code.
    │
    │
    ├── readme.md                       # Contains the environment requirements, data pre-processing requirements and model information.
    │                                     Developers should refer to this to understand how to test and train the model with scripts.
    │
    ├── data                            # The dataset target directory that is used for model verification and training.
    │                                     When test or training scripts run successfully, the dataset will be placed in this directory.
    │
    ├── qat                             # Contains the QAT (Quantization Aware Training) results.
    │                                     For some models, the accuracy of QAT is higher than with Post Training Quantization (PTQ) methods.
    │                                     Some models, but not all, provide QAT reference results, and only these models have a QAT folder.
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

.. note:: 

   - For more information on Vitis-AI Quantizer executables ``vai_q_tensorflow`` and ``vai_q_pytorch``, please see the `Vitis AI User Guide <https://docs.xilinx.com/r/en-US/ug1414-vitis-ai>`__.
   - Due to licensing restrictions, some model archives include instructions as to how the user can leverage that model architecture with Vitis AI, but do not include the pretrained model.  In these cases, the user must leverage the documentation provided to build and train their own version of the model.  
   - For more information about the various Xilinx DPUs, see the :doc:`DPU IP Product Guides <reference/release_documentation>`




.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:

