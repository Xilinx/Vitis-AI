August 2022 Patch
-----------------

New Features/Highlights
~~~~~~~~~~~~~~~~~~~~~~~

-  Supported correlation 1d and correlation 2d operators with DPUCZDX8G and DPUCVDX8G
-  Supported concatenate operator with multiple identical input tensors
-  Fixed a compiler bug to support concatenate operator cascaded with multiple reshape operators

New Packages
~~~~~~~~~~~~

-  `target_factory-2.5.0-py36h680af44_202.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=target_factory-2.5.0-py36h680af44_202.tar.bz2>`__
-  `target_factory-2.5.0-py37h680af44_202.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=target_factory-2.5.0-py37h680af44_202.tar.bz2>`__
-  `xir-2.5.0-py36h893bffd_202.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xir-2.5.0-py36h893bffd_202.tar.bz2>`__
-  `xir-2.5.0-py37h893bffd_202.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xir-2.5.0-py37h893bffd_202.tar.bz2>`__
-  `xcompiler-2.5.0-py36hea4fdf2_202.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xcompiler-2.5.0-py36hea4fdf2_202.tar.bz2>`__
-  `xcompiler-2.5.0-py37hea4fdf2_202.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xcompiler-2.5.0-py37hea4fdf2_202.tar.bz2>`__
-  `vart-2.5.0-py36h07a2524_202.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=vart-2.5.0-py36h07a2524_202.tar.bz2>`__
-  `vart-2.5.0-py37h07a2524_202.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=vart-2.5.0-py37h07a2524_202.tar.bz2>`__

Installation
------------

Download the packages from the link above. Apply the conda patch to the conda environment (Machine Learning
framework) that you wish to update in this format.

.. code-block::

   sudo conda install -n <CONDA_ENVIRONMENT> <URL or PATH to conda package>

For example, to update the ``vitis-ai-pytorch`` conda environment:

.. code-block::

   sudo conda install -n vitis-ai-pytorch https://www.xilinx.com/bin/public/openDownload?filename=target_factory-2.5.0-py36h680af44_202.tar.bz2

Make sure you install them in the order of unilog, target_factory, xir, xcompiler, xnnc, and use the proper version according to the Python version installed in your current conda env.
