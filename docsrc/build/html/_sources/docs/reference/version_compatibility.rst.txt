.. _version-compatibility:

IP and Tool Version Compatibility
=================================

This page provides information on the compatibility between tools, IP, and Vitis |trade| AI release versions. Ensure that you are using aligned versions of all components.

Current Release
---------------

Vitis AI v3.5 and the DPU IP released with the v3.5 branch of this repository are verified as compatible with Vitis, Vivado |trade|, and PetaLinux version 2023.1. If you are using a previous release of Vitis AI, please refer to the table below release.


All Releases
------------

Versal |trade| AI Edge with AIE-ML

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Vitis AI Release Version
     - DPUCV2DX8G IP Version
     - Software Tools Version
     - Linux Kernel Version Tested
	 
   * - v3.5
     - 1.0
     - Vivado / Vitis / PetaLinux 2023.1
     - 6.1	 
	 
   * - v3.0
     - Early Access
     - Vivado / Vitis / PetaLinux 2022.2
     - 5.15

	 
Alveo |trade| V70

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Vitis AI Release Version
     - DPUCV2DX8G IP Version
     - Software Tools Version
     - Linux Kernel Version Tested
	 
   * - v3.5
     - 1.0
     - Vivado / Vitis 2022.2
     - 5.15	 
	 
   * - v3.0
     - Early Access
     - Vivado / Vitis 2022.2
     - 5.15

Zynq |trade| Ultrascale+ |trade|

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Vitis AI Release Version
     - DPUCZDX8G IP Version
     - Software Tools Version
     - Linux Kernel Version Tested

   * - v3.5
     - 4.1 (not updated*)
     - Vivado / Vitis / PetaLinux 2023.1
     - 6.1

   * - v3.0
     - 4.1
     - Vivado / Vitis / PetaLinux 2022.2
     - 5.15

   * - v2.5
     - 4.0
     - Vivado / Vitis / PetaLinux 2022.1
     - 5.15

   * - v2.0
     - 3.4
     - Vivado / Vitis / PetaLinux 2021.2
     - 5.10

   * - v1.4
     - 3.3
     - Vivado / Vitis / PetaLinux 2021.1
     - 5.10

   * - v1.3
     - 3.3
     - Vivado / Vitis / PetaLinux 2020.2
     - 5.4	 

   * - v1.2
     - 3.2
     - Vivado / Vitis / PetaLinux 2020.1
     - 5.4
	 
   * - v1.1
     - 3.2
     - Vivado / Vitis / PetaLinux 2019.2 
     - 4.19

   * - v1.0
     - 3.1
     - Vivado / Vitis / PetaLinux 2019.1
     - 4.19

   * - N/A (DNNDK)
     - 3.0
     - Vivado / Vitis / PetaLinux 2019.1
     - 4.19

   * - N/A (DNNDK)
     - 2.0
     - Vivado / Vitis / PetaLinux 2018.2
     - 4.14

   * - First Release (DNNDK)
     - 1.0
     - Vivado / Vitis / PetaLinux 2018.1
     - 4.14

Versal |trade| AI Core with AIE-ML

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Vitis AI Release Version
     - DPUCV2DX8G IP Version
     - Software Tools Version
     - Linux Kernel Version Tested

   * - v3.5
     - 1.0
     - Vivado / Vitis / PetaLinux 2023.1
     - 6.1

   * - v3.0
     - Early Access
     - Vivado / Vitis / PetaLinux 2022.2
     - 5.15


Versal |trade| AI Core with AIE

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Vitis AI Release Version
     - DPUCVDX8G IP Version
     - Software Tools Version
     - Linux Kernel Version Tested

   * - v3.5
     - 1.3 (not updated*)
     - Vitis 2022.2
     - 5.15

   * - v3.0
     - 1.3
     - Vitis 2022.2
     - 5.15

   * - v2.5
     - 1.2
     - Vitis 2022.1
     - 5.15

   * - v2.0
     - 1.1
     - Vitis 2021.2
     - 5.10

   * - v1.4
     - 1.0
     - Vitis 2021.1
     - 5.10

   * - v1.3
     - Early Access
     - Vitis 2020.2
     - 5.4
 

Entries marked `not updated` are considered mature IP and do not have a corresponding pre-built board image and reference design.  Extensive compatibility testing will not be done for mature IP for minor (x.5) releases, but will be refreshed with each major (x.0) release.


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
