# (C) Copyright 2020 - 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0

RM = rm -rf
VIVADO = $(XILINX_VIVADO)/bin/vivado
JOBS ?= 8

VIV_DESIGN = vck190_mipiRxSingle_hdmiTx

VIV_PRJ_DIR = project
VIV_SCRIPTS_DIR = scripts

VIV_XSA = $(VIV_PRJ_DIR)/$(VIV_DESIGN).xsa
VIV_SRC = $(VIV_SCRIPTS_DIR)/main.tcl
ISP_DIR= ../ip/isp_single
ISP_IP = isppipeline.prj 

.PHONY: help
help:
	@echo 'Usage:'
	@echo ''
	@echo '  make xsa'
	@echo '    Generate extensible xsa for platform generation'
	@echo ''

.PHONY: all
all: xsa

$(ISP_IP):
	make -C $(ISP_DIR) ip XPART=xcvc1902-vsva2197-1LP-e-S

xsa: $(VIV_XSA)
$(VIV_XSA): $(VIV_SRC) $(ISP_IP)
	$(VIVADO) -mode batch -notrace -source $(VIV_SRC) -tclargs -jobs $(JOBS) 

.PHONY: clean
clean:
	$(RM) $(VIV_PRJ_DIR) vivado* .Xil *dynamic* *.log *.xpe
	make -C $(ISP_DIR) clean

