# (C) Copyright 2020 - 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0

CP = cp -rf
MV = mv
MKDIR = mkdir -p
RM = rm -rf
XSCT = $(XILINX_VITIS)/bin/xsct
JOBS ?= 8
REV ?= es1

PLATFORM ?= vck190_mipiRxSingle_hdmiTx
PFM ?= vck190_mipiRxSingle_hdmiTx

VERSION ?= 202120_1

PFM_DIR = xilinx_$(PLATFORM)_$(VERSION)
PFM_PRJ_DIR = xsct/$(PLATFORM)/$(PLATFORM)/export/$(PLATFORM)
PFM_SCRIPTS_DIR = scripts

PFM_TCL = $(PFM_SCRIPTS_DIR)/pfm.tcl
PFM_XPFM = $(PFM_DIR)/$(PLATFORM).xpfm

VIV_DIR = vivado/$(PLATFORM)
VIV_XSA = $(VIV_DIR)/project/$(PLATFORM).xsa

.PHONY: help
help:
	@echo 'Usage:'
	@echo ''
	@echo '  make platform PLATFORM=vck190_mipiRxSingle_hdmiTx JOBS=8'
	@echo '    Generate Vitis platform'
	@echo ''

.PHONY: all
all: platform

.PHONY: platform
platform: $(PFM_XPFM)
$(PFM_XPFM): $(VIV_XSA)
	$(XSCT) $(PFM_TCL) -xsa $(VIV_XSA)
	@$(CP) $(PFM_PRJ_DIR) $(PFM_DIR)
	@$(PFM_SCRIPTS_DIR)/pfm.sh $(PFM) $(PFM_DIR)
	@echo 'Vitis platform available at $(PFM_DIR)'

$(VIV_XSA):
	make -C $(VIV_DIR) xsa JOBS=$(JOBS)

.PHONY: clean
clean:
	-@$(RM) .Xil boot image linux.bif xsct ws $(PFM_DIR)
	make -C $(VIV_DIR) clean

