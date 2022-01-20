## Level 3: Pipeline Applications

This directory contains whole applications formed by stitching a pipeline of Vitis Vision functions. The host code shows how to call this multiple functions in OpenCL.

'examples' folder contains the OpenCL host code file and a C++ accel file that demonstrate the call of Vitis Vision functions to build for Vitis.

'examples/build' folder has the configuration file that would help modify the default configuration of the function.

'tests' folder has sub-folders named according to the function and the configuration it would run. Each individual folder has Makefiles and config files that would perform software emulation, hardware emulation and hardware build of the corresponding function in examples folder, based on the 'Board' the user selects.

'benchmarks' directory has applications ready to build that give out their performance comparison against other architectures.

### Commands to run:

    source < path-to-Vitis-installation-directory >/settings64.sh

    export DEVICE=< path-to-platform-directory >/< platform >.xpfm

**For PCIe devices:**

    source < part-to-XRT-installation-directory >/setup.sh

	export OPENCV_INCLUDE=< path-to-opencv-include-folder >

	export OPENCV_LIB=< path-to-opencv-lib-folder >

	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder >

    make host xclbin TARGET=< sw_emu|hw_emu|hw >

    make run TARGET=< sw_emu|hw_emu|hw >

**For embedded devices:**

	Download the platform, and common-image from Xilinx Download Center. Run the sdk.sh script from the common-image directory to install sysroot using the command : "./sdk.sh -y -d ./ -p"

	Unzip the rootfs file : "gunzip ./rootfs.ext4.gz"

    export SYSROOT=< path-to-platform-sysroot >

	export EDGE_COMMON_SW=< path-to-rootfs-and-Image-files >

	export PERL=<path-to-perl-installation-location> #For example, "export PERL=/usr/bin/perl". Please make sure that Expect.pm package is available in your Perl installation.

    make host xclbin TARGET=< sw_emu|hw_emu|hw > HOST_ARCH=< aarch32 | aarch64 >

    make run TARGET=< sw_emu|hw_emu|hw > HOST_ARCH=< aarch32 | aarch64 > #This command will generate only the sd_card folder in case of hardware build.

**Note**. For hw run on embedded devices, copy the generated sd_card folder content under package_hw to an SD Card. More information on preparing the SD Card is available [here](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18842385/How+to+format+SD+card+for+SD+boot#HowtoformatSDcardforSDboot-CopingtheImagestotheNewPartitions). After successful booting of the board, run the following commands:

    cd /mnt

    export XCL_BINDIR=< xclbin-folder-present-in-the-sd_card > #For example, "export XCL_BINDIR=xclbin_zcu102_base_hw"

    ./run_script.sh
