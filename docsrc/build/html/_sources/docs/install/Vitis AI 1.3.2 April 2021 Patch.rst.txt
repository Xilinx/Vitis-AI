=================
April 2021 Patch
=================

New Features/Highlights
~~~~~~~~~~~~~~~~~~~~~~~

-  Fixed a compiler bug about “XIR_REMOVE_OP_FAIL”
-  Updated target description to support pool kernel=1 in DPUCZDX8G
-  Updated parser to support integrated activation in tf2 dense
-  Updated parser to fix a bug when using fine-tuned tf2 model
-  Removed some unnecessary assertions in parser to support unquantized leakyrelu
-  Updated parser to support customized tf2 UpsampleLike operator
-  Updated compiler to support depth-wise engine in DPUCVDX8G
-  Updated XIR to raise compilation warnings for using legacy APIs other than runtime warnings

New Packages
~~~~~~~~~~~~

-  `unilog-1.3.2-h7b12538_35.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=unilog-1.3.2-h7b12538_35.tar.bz2>`__
-  `target_factory-1.3.2-hf484d3e_35.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=target_factory-1.3.2-hf484d3e_35.tar.bz2>`__
-  `xir-1.3.2-py36h7b12538_47.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xir-1.3.2-py36h7b12538_47.tar.bz2>`__
-  `xir-1.3.2-py37h7b12538_47.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xir-1.3.2-py37h7b12538_47.tar.bz2>`__
-  `xcompiler-1.3.2-py36h7b12538_53.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xcompiler-1.3.2-py36h7b12538_53.tar.bz2>`__
-  `xcompiler-1.3.2-py37h7b12538_53.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xcompiler-1.3.2-py37h7b12538_53.tar.bz2>`__
-  `xnnc-1.3.2-py36_48.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xnnc-1.3.2-py36_48.tar.bz2>`__
-  `xnnc-1.3.2-py37_48.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=xnnc-1.3.2-py37_48.tar.bz2>`__

Installation
------------

Download the packages from the link above.

.. code-block::

   sudo env PATH=/opt/vitis_ai/conda/bin:$PATH CONDA_PREFIX=/opt/vitis_ai/conda/envs/YOUR_ENV_NAME conda install PATCH_PACKAGE.tar.bz2

You can install the packages with the command above. Make sure you install them in the order of unilog, target_factory, xir, xcompiler, xnnc, and use the proper version according to the python version installed in your current conda env.
