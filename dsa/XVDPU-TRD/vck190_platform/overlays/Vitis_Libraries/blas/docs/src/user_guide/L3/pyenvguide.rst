.. meta::
   :keywords: BLAS, Library, Vitis BLAS Library, python, setup
   :description: Python environment setup guide.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



Python Environment Setup Guide
===============================

**1. Installing Anaconda3**

1) Download Anaconda3

.. code-block:: bash

  $ wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh

2) Run the installer (Installer requires bzip, please install it if you don't have it)

.. code-block:: bash

  $ bash ./Anaconda3-2019.03-Linux-x86_64.sh

Choose "Yes" for question "Do you wish the installer to initialize Anaconda3 by running conda init?". More information about Anaconda can be found from `Anaconda Documentation`_.

.. _Anaconda Documentation: https://docs.anaconda.com/anaconda/

3) Add Anaconda3 to PATH, for example:

.. code-block:: bash

  $ export PATH=/home/<user>/anaconda3/bin:$PATH
  $ . /home/<user>/anaconda3/etc/profile.d/conda.sh


**2. Setting up xf_blas environment to include all conda packages used by xf_blas L1 primitive testing infrastructure.**

Please run following command under directory xf_blas/. 

.. code-block:: bash

  $ conda config --add channels anaconda
  $ conda env create -f environment.yml
  $ conda activate xf_blas
  $ conda install --file requirements.txt


**3. Deactivate xf_blas environment after testing**

Note: Please don't take this step if you intend to run L1 primitives' testing process. 
You only take it after you've finished all testing.

.. code-block:: bash

  $ conda deactivate
