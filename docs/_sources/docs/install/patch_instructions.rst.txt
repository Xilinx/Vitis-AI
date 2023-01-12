===========================
Installing a Vitis AI Patch
===========================

Most Vitis AI components consist of Anaconda packages. These packages are distributed as tarballs, for example
`unilog-1.3.2-h7b12538_35.tar.bz2 <https://www.xilinx.com/bin/public/openDownload?filename=unilog-1.3.2-h7b12538_35.tar.bz2>`__.

You can install the patches by starting the Vitis AI Docker container, and installing the package to a specific conda environment. For example patching the ``unilog`` package in the ``vitis-ai-caffe`` conda environment:

.. code-block::

   Vitis-AI /workspace > cd /tmp
   Vitis-AI /tmp > wget https://www.xilinx.com/bin/public/openDownload?filename=unilog-1.3.2-h7b12538_35.tar.bz2 -O unilog-1.3.2-h7b12538_35.tar.bz2
   Vitis-AI /tmp > sudo conda install -n vitis-ai-caffe ./unilog-1.3.2-h7b12538_35.tar.bz2
