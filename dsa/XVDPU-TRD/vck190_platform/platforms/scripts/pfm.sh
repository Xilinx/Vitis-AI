# (C) Copyright 2020 - 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

PLATFORM=$1
PFM_TOP=$2

cat >> $PFM_TOP/sw/$PLATFORM/boot/linux.bif << EOF
  /* linux */
  the_ROM_image:
  {
    { load=0x1000, file=<dtb,${PLATFORM}/boot/system.dtb> }
    { core=a72-0, exception_level=el-3, trustzone, file=<atf,${PLATFORM}/boot/bl31.elf> }
    { core=a72-0, exception_level=el-2, load=0x8000000, file=<uboot,${PLATFORM}/boot/u-boot.elf> }
  }
EOF
