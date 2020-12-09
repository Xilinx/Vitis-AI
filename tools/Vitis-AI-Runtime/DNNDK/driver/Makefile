## Copyright (C) 2019 Xilinx, Inc.
##
## This software is licensed under the terms of the GNU General Public
## License version 2, as published by the Free Software Foundation, and
## may be copied, distributed, and modified under those terms.

#---------------------------------------------------------------------------------------------
# makefile for DPU driver outside the linux kernel tree, and generate dpu.ko file
#
# a typical command for build the driver for Zynq 7000:
#		make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- KERNELDIR=/path/to/your/kernel
#
# a typical command for build the driver for UltraScale+:
#		make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- KERNELDIR=/path/to/your/kernel
#---------------------------------------------------------------------------------------------

modname:=dpu
obj-m:=$(modname).o
dpu-objs:=dpucore.o

PWD :=$(shell pwd)
MAKE :=make


KCFLAGS=KCFLAGS="
ifeq ($(DPU_TARGET),1.1)
	KCFLAGS +=-DCONFIG_DPU_v1_1_X
else
	KCFLAGS +=-DCONFIG_DPU_v1_3_0
endif

ifeq ($(ARCH),arm)
	KCFLAGS +=-DSIG_BASE_ADDR=0X4FF00000 -DCACHE_OFF
endif
ifeq ($(ARCH),arm64)
	KCFLAGS +=-DSIG_BASE_ADDR=0X8FF00000
endif

# check the compiler version
GCCV1 := $(shell $(CROSS_COMPILE)gcc -dumpversion | cut -f1 -d. )
GCCV2 := $(shell $(CROSS_COMPILE)gcc -dumpversion | cut -f2 -d. )
GCCV3 := $(shell expr `echo $(GCCV1)"*100+"$(GCCV2) | bc`  )

DT_FLAG := $(shell expr `echo $(GCCV3)` \>= 409)
ifeq ($(DT_FLAG),1)
    KCFLAGS += -Wno-error=date-time -Wno-date-time
endif
KCFLAGS+="

all:
	$(KCFLAGS) $(MAKE) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) -C $(KERNELDIR) M=$(PWD) modules

clean:
	rm -rf $(modname).ko *.o *mod* \.*cmd *odule* .tmp_versions

.PHONY: all clean

