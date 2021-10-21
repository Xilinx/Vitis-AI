PLNX_VER = 2021.1
PLNX_PRJ = xilinx-vck190-base-trd
PLNX_BSP = $(PLNX_PRJ)-$(PLNX_VER).bsp

.PHONY: help
help:
	@echo 'Usage:'
	@echo ''
	@echo '  make project'
	@echo '    This rule creates a petalinux project from a bsp file.'
	@echo ''
	@echo '  make bsp'
	@echo '    This rule packages the petalinux project as a bsp file.'
	@echo ''

.PHONY: all
all: project

.PHONY: project
project: $(PLNX_BSP)
	@echo 'Create PetaLinux project from bsp file'
	petalinux-create -t project -s $(PLNX_BSP)

.PHONY: bsp
bsp: $(PLNX_PRJ)
	@echo 'Create PetaLinux BSP'
	petalinux-package --bsp -p $(PLNX_PRJ) -o $(PLNX_BSP)
