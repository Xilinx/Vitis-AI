export PLATFORM = xilinx_vck190_es1_trd
export CPU_ARCH = a72

export CWD              = $(shell readlink -f .)
export VERSION          = 202020_1
export VER              = 202020.1
export XSA_DIR          = $(CWD)/hw
export SYSROOT          = $(CWD)/sw/build/sysroot
export OUTPUT_PATH      = $(CWD)/platform_repo
export SW_COMP_DIR      = $(OUTPUT_PATH)/tmp
export XSCT             = $(XILINX_VITIS)/bin/xsct
export PLATFORM_NAME    = $(PLATFORM)_$(VERSION)
export XSA              = $(XSA_DIR)/vck190_es1_base_trd_platform1.xsa
export PLATFORM_SW_SRC  = $(CWD)/platform
export BOOT_FILES       = $(BOOT_FILES_PATH)

############################## Help Section ##############################
.PHONY: help

help::
	@echo 'Makefile Usage:'
	@echo ''
	@echo '  All the make commands install platform to "platform_repo/$(PLATFORM_NAME)/export/$(PLATFORM_NAME)"'
	@echo '  make all'
	@echo '      Command used to generate platform with petalinux. Source petalinux before running this command.'
	@echo '      This command builds all components.'
	@echo ''
	@echo '  make xsa'
	@echo '      Command to generate xsa.'
	@echo ''
	@echo '  make linux XSA_DIR=/home/user/$(PLATFORM_NAME)/vivado/build'
	@echo '  make platform'
	@echo '      Run these commands when XSA is already built in some other folder and no modifications are needed to it.'
	@echo '      XSA_DIR should point to directory containing $(PLATFORM_NAME).xsa file.'
	@echo ''
	@echo '  make clean '
	@echo '      Command to remove all the generated files.'
	@echo ''

xsa:
	$(MAKE) -C hw all

linux:
	$(MAKE) -C sw all

petalinux_sysroot:
	$(MAKE) -C sw/build sysroot
	@if [ -d $(SYSROOT) ]; then cp -rf $(SYSROOT) $(OUTPUT_PATH)/; fi

.PHONY: platform
platform:
	$(XSCT) -nodisp -sdx $(PLATFORM_SW_SRC)/${PLATFORM}_pfm.tcl ${PLATFORM_NAME} ${XSA_DIR} $(SW_COMP_DIR) ${OUTPUT_PATH}
	@if [ -d $(SW_COMP_DIR)/platform/filesystem ]; then cp -rf ${SW_COMP_DIR}/platform/filesystem $(OUTPUT_PATH)/${PLATFORM_NAME}/export/${PLATFORM_NAME}/sw/${PLATFORM_NAME}/xrt/; fi

pre-built:
	@if [ -f $(SW_COMP_DIR)/platform/BOOT.BIN ]; then mkdir -p $(OUTPUT_PATH)/${PLATFORM_NAME}/export/${PLATFORM_NAME}/pre-built/ && cp -rf $(SW_COMP_DIR)/platform/BOOT.BIN $(OUTPUT_PATH)/${PLATFORM_NAME}/export/${PLATFORM_NAME}/pre-built/; fi

clean:
	${RM} -r .Xil  
	${RM} -r $(OUTPUT_PATH)
	$(MAKE) -C hw clean
	$(MAKE) -C sw clean

all: xsa linux platform pre-built
	@echo 'Platform build complete'
