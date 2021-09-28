# (C) Copyright 2020 - 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0

RM = rm -rf
HLS = vitis_hls
CLKP ?= 6.6
XPART ?= xcvc1902-vsva2197-1LP-e-S-es1


.PHONY: help
help:
	@echo 'Usage:'
	@echo ''
	@echo '  make ip'
	@echo '    Generate IP for Single Sensor ISP'
	@echo ''

.PHONY: all
all: ip

ip:  
	@echo 'set CLKP $(CLKP)' >> ./settings.tcl
	@echo 'set XPART $(XPART)' >> ./settings.tcl
	$(HLS) -f run_hls.tcl

.PHONY: clean
clean:
	$(RM) settings.tcl *.prj *.log

