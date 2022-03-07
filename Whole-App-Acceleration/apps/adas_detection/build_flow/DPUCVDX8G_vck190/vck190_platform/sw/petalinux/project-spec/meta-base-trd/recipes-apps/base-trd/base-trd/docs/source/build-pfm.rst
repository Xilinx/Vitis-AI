Create the Vitis Platform
=========================

Prerequisites
-------------

* Reference Design zip file

* Vitis Unified Software Platform 2020.2

Build Flow Tutorial
-------------------

.. tip::

   You can skip this tutorial and move straight to the next tutorial if desired.
   A pre-built Vitis platform file is provided at
   *$working_dir/platform/vck190_base_trd_platform1*

.. note::

   The below steps use platform 1 as an example. The same steps can be used for
   other platforms as well, only the file/directory names with platform1 will be
   replaced with the targeted platform.

**Download Reference Design Files:**

Skip the following steps if the design zip file has already been downloaded and
extracted to a working directory

#. Download the VCK190 Base Targeted Reference Design ZIP file

#. Unzip Contents

The directory structure is described in the Introduction Section

**Create a Vitis Extensible Platform:**

#. Use XSCT to generate the Vitis platform

   To create the Vitis platform, run the following xsct tcl script:

   .. code-block:: bash

      cd $working_dir/platform
      xsct pfm.tcl -xsa $working_dir/vivado/project/vck190_base_trd_platform1.sdk/vck190_base_trd_platform1.xsa

   The generated platform will be located at:

   *$working_dir/platform/ws/vck190_base_trd_platform1/export/vck190_base_trd_platform1*

   It will be used as input when building the Vitis accelerator projects.
