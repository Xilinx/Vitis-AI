# (C) Copyright 2020 - 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0

CP = cp -f

PWD = $(shell readlink -f .)

# the platform directory has to be an absolute path when passed to v++
PFM_DIR = $(PWD)/platforms
PFM_VER = 202110_1

# valid platforms / overlays
PFM = vck190_mipiRxSingle_hdmiTx

PFM_XPFM = $(PFM_DIR)/xilinx_$(PFM)_$(PFM_VER)/$(PFM).xpfm
PFM_TOP = $(PFM_DIR)/xilinx_$(PFM)_$(PFM_VER)

PLNX_DIR = petalinux/xilinx-vck190-base-trd
PLNX_WIC = $(PLNX_DIR)/images/linux/petalinux-sdimage.wic.xz

VIV_DIR = $(PFM_DIR)/vivado/$(PFM)
VIV_XSA = $(VIV_DIR)/project/$(PFM).xsa

.PHONY: help
help:
	@echo 'Usage:'
	@echo ''
	@echo '  make all'
	@echo '    Build the Vitis platform.'
	@echo ''
	@echo '  make clean'
	@echo '    Clean sw only'
	@echo ''
	@echo '  make distclean'
	@echo '    Clean hw and sw'
	@echo ''

.PHONY: all
all: platform

$(PLNX_DIR):
	$(MAKE) -C petalinux project

.PHONY: linux
linux: $(PFM_XPFM)
	$(MAKE) -C $(PLNX_DIR) -f Makefile pfm PFM=$(PFM) PFM_TOP=$(PFM_TOP)

.PHONY: platform
platform: $(PFM_XPFM) linux $(PLNX_DIR)
$(PFM_XPFM):
	$(MAKE) -C $(PFM_DIR) platform PLATFORM=$(PFM) VERSION=$(PFM_VER)

.PHONY: clean
clean:
	$(MAKE) -C $(PLNX_DIR) clean

.PHONY: distclean
distclean:
	$(MAKE) -C $(PLNX_DIR) clean
	$(MAKE) -C $(PFM_DIR) clean PLATFORM=$(PFM) VERSION=$(PFM_VER);
