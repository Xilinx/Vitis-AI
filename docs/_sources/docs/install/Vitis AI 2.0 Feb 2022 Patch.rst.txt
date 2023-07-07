February 2022 Patch
-------------------

New Features/Highlights
~~~~~~~~~~~~~~~~~~~~~~~

-  Fixed a compiler bug for pt_yolox_TT100K_640_640_73G_2.0 model
-  Fixed a quantizer bug in QAT in tensorflow 1.15 models
-  Fixed a bug which gives low accuracy for TVM compiled inception_v3 model with DPUCADF8H(dpuv3int8)
-  Fixed a bug in AI optimizer to get the best accuracy in run_evolutionary_search function

New Packages
~~~~~~~~~~~~

-  `unilog-2.0.1-hea4fdf2_32.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=unilog-2.0.1-hea4fdf2_32.tar.bz2>`__
-  `target_factory-2.0.1-h680af44_32.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=target_factory-2.0.1-h680af44_32.tar.bz2>`__
-  `xir-2.0.1-py36h893bffd_32.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xir-2.0.1-py36h893bffd_32.tar.bz2>`__
-  `xir-2.0.1-py37h893bffd_32.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xir-2.0.1-py37h893bffd_32.tar.bz2>`__
-  `dpuv3-pycompiler-2.0.1-py36h3f44cd5_8.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=dpuv3-pycompiler-2.0.1-py36h3f44cd5_8.tar.bz2>`__
-  `dpuv3-pycompiler-2.0.1-py37h3f44cd5_8.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=dpuv3-pycompiler-2.0.1-py37h3f44cd5_8.tar.bz2>`__
-  `vai_optimizer_pytorch_gpu-2.0.1-py36h164a702_28.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=vai_optimizer_pytorch_gpu-2.0.1-py36h164a702_28.tar.bz2>`__
-  `vai_q_tensorflow-1.15-py36hc48d084_2.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=vai_q_tensorflow-1.15-py36hc48d084_2.tar.bz2>`__
-  `vai_q_tensorflow_gpu-1.15-py36h91ed69b_2.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=vai_q_tensorflow_gpu-1.15-py36h91ed69b_2.tar.bz2>`__
-  `vaic-2.0.1-py36hd51751d_11.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=vaic-2.0.1-py36hd51751d_11.tar.bz2>`__
-  `vaic-2.0.1-py37hd51751d_11.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=vaic-2.0.1-py37hd51751d_11.tar.bz2>`__
-  `xcompiler-2.0.1-py36hea4fdf2_32.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xcompiler-2.0.1-py36hea4fdf2_32.tar.bz2>`__
-  `xcompiler-2.0.1-py37hea4fdf2_32.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xcompiler-2.0.1-py37hea4fdf2_32.tar.bz2>`__
-  `xnnc-2.0.1-py36hd51751d_27.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xnnc-2.0.1-py36hd51751d_27.tar.bz2>`__
-  `xnnc-2.0.1-py37hd51751d_27.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xnnc-2.0.1-py37hd51751d_27.tar.bz2>`__

Installation
------------

Download the packages from the link above. Apply the conda patch to the conda environment (Machine Learning
framework) that you wish to update in this format

.. code-block::

   sudo conda install -n <CONDA_ENVIRONMENT> <URL or PATH to conda package>

For example, to update the ``vitis-ai-pytorch`` conda environment:

::

   sudo conda install -n vitis-ai-pytorch https://www.xilinx.com/bin/public/openDownload?filename=unilog-2.0.1-hea4fdf2_32.tar.bz2

Make sure you install them in the order of unilog, target_factory, xir,
xcompiler, xnnc, and use the proper version according to the python
version installed in your current conda env.
