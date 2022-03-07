Build the PetaLinux Image
=========================

Prerequisites
-------------

* Reference Design zip file

* PetaLinux Tools 2020.2

* Linux host machine

Build Flow Tutorial
-------------------

.. note::

   The below steps use platform 1 as an example. The same steps can be used for
   other platforms as well, only the file/directory names with platform1 will be
   replaced with the targeted platform.

**Download Reference Design Files**

Skip the following steps if the design zip file has already been downloaded and
extracted to a working directory

#. Download the VCK190 Base Targeted Reference Design ZIP file

#. Unzip Contents

The directory structure is described in the Introduction Section.

**Generate PetaLinux Image**

#. Enable Versal device support

   To enable Versal device support in PetaLinux, follow the instructions
   in this README file: https://www.xilinx.com/member/forms/download/xef.html?filename=petalinux-README_2020.2.tar.gz

#. Create a new PetaLinux project from the provided BSP:

   .. code-block:: bash

      cd $working_dir/petalinux
      petalinux-create -t project -s xilinx-vck190-es1-base-trd-platform1-2020.2.bsp
      cd xilinx-vck190-es1-base-trd-platform1-2020.2


#. Configure and build the PetaLinux project.

   .. note::

      The XSA file used for project configuration is included with this BSP.
      The build step performs the configuration step implicitly. If you want
      to configure your BSP with the XSA file generated in the previous
      tutorial, run the following command **prior** to the *petalinux-build*
      command, otherwise the config step can be skipped:

      .. code-block:: bash

         petalinux-config --get-hw-description=../../accelerators/examples/filter2d_combined/ --silentconfig

   .. code-block:: bash

      petalinux-build

#. Create a boot image:

   .. code-block:: bash

      petalinux-package --boot --u-boot --qemu-rootfs no --force

#. Create a bootable SD card image:

   .. code-block:: bash

      cp pre-built/binary_container_1.xclbin images/linux
      petalinux-package --wic -w project-spec/configs/sdimage.wks --extra-bootfiles "binary_container_1.xclbin"

The resulting build artifacts will be available in the *images/linux/* folder.
The following is a list of important output files:

* **binary_container_1.xclbin**: This is the kernel meta data file used by XRT

* **BOOT.BIN**: This is the boot image which includes:

  * Platform Loader and Manager (PLM)

  * PS Management (PSM) firmware

  * Platform Device Image (PDI)

  * ARM trusted firmware

  * u-boot

  * Device tree blob

  * Merged AI Engine application and configuration data object (CDO) file

* **boot.scr**: A u-boot boot script

* **Image**: Linux kernel image

* **rootfs.tar.gz**: Compressed root file system tar ball

* **petalinux-sdimage.wic**: SD card image file in wic format

The SD card image is now ready to be used to boot the device into Linux, see
Section *Run the Prebuilt Image* for details.

