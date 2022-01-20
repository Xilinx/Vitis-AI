RM = rm -rf

# to be passed from top-level Makefile
PFM ?= vck190_mipiRxSingle_hdmiTx 
YES ?= 1

PLNX_IMG_DIR = images/linux
PLNX_SDK = $(PLNX_IMG_DIR)/sdk.sh

PLNX_BOOT = $(PLNX_IMG_DIR)/BOOT.BIN
PLNX_BOOT_OBJS += $(PLNX_IMG_DIR)/bl31.elf
PLNX_BOOT_OBJS += $(PLNX_IMG_DIR)/plm.elf
PLNX_BOOT_OBJS += $(PLNX_IMG_DIR)/psmfw.elf
PLNX_BOOT_OBJS += $(PLNX_IMG_DIR)/u-boot.elf
PLNX_BOOT_ARGS = --u-boot --qemu-rootfs no

PLNX_WIC = $(PLNX_IMG_DIR)/petalinux-sdimage.wic.xz
PLNX_WIC_OBJS += $(PLNX_IMG_DIR)/boot.scr
PLNX_WIC_OBJS += $(PLNX_IMG_DIR)/Image
PLNX_WIC_OBJS += $(PLNX_IMG_DIR)/rootfs.tar.gz
PLNX_WIC_OBJS += $(PLNX_IMG_DIR)/ramdisk.cpio.gz.u-boot
PLNX_WIC_OBJS += $(PLNX_IMG_DIR)/system.dtb
PLNX_WIC_ARGS += --bootfiles "ramdisk.cpio.gz.u-boot boot.scr Image system.dtb BOOT.BIN"
PLNX_WIC_ARGS += --wic-extra-args "-c xz"

PLNX_IMG_OBJS += $(PLNX_BOOT_OBJS)
PLNX_IMG_OBJS += $(PLNX_WIC_OBJS)

XSA_DIR = $(PFM_TOP)/hw

.PHONY: help
help:
	@echo 'Usage:'
	@echo ''
	@echo '  make wic'
	@echo '    This rule builds an SD card wic image.'
	@echo ''
	@echo '  make boot'
	@echo '    This rule builds a boot image (BOOT.BIN).'
	@echo ''
	@echo '  make image'
	@echo '    This rule builds a full Linux image. It is always executed'
	@echo '    unconditionally. The yocto build system tracks changes and'
	@echo '    performs an incremental build on subsequent runs.'
	@echo ''
	@echo '  make sdk'
	@echo '    This rule builds an SDK.'
	@echo ''

.PHONY: all
all: wic

.PHONY: wic
wic: $(PLNX_WIC)
$(PLNX_WIC): $(PLNX_WIC_OBJS) $(PLNX_BOOT)
	@echo 'Build PetaLinux wic image'
	petalinux-package --wic $(PLNX_WIC_ARGS)

.PHONY: pfm
pfm: $(PLNX_WIC_OBJS) $(PLNX_BOOT)
	@mkdir -p $(PFM_TOP)/sw/$(PFM)/xrt
	@mkdir -p $(PFM_TOP)/sw/$(PFM)/xrt/image
	@mkdir -p $(PFM_TOP)/sw/$(PFM)/xrt/filesystem
	@cp -a images/linux/bl31.elf $(PFM_TOP)/sw/$(PFM)/boot/bl31.elf
	@cp -a images/linux/system.dtb $(PFM_TOP)/sw/$(PFM)/boot/system.dtb
	@cp -a images/linux/u-boot.elf $(PFM_TOP)/sw/$(PFM)/boot/u-boot.elf
	@cp -a images/linux/boot.scr $(PFM_TOP)/sw/$(PFM)/xrt/image/boot.scr
	@cp -a images/linux/Image $(PFM_TOP)/sw/$(PFM)/xrt/image/Image
	@cp -a images/linux/ramdisk.cpio.gz.u-boot $(PFM_TOP)/sw/$(PFM)/xrt/image/ramdisk.cpio.gz.u-boot
	@cp -a images/linux/rootfs.ext4 $(PFM_TOP)/sw/$(PFM)/xrt/filesystem/rootfs.ext4

.PHONY: boot
boot: $(PLNX_BOOT)
$(PLNX_BOOT): $(PLNX_BOOT_OBJS)
	@echo 'Build PetaLinux boot image'
	petalinux-package --boot $(PLNX_BOOT_ARGS) --force

.PHONY: image
image: $(PLNX_IMG_OBJS)
$(PLNX_IMG_OBJS):
	@echo 'Build PetaLinux image'
	@./trd-pl-cfg -p $(PFM)
	@if [ $(YES) -eq 1 ]; then \
	  yes | petalinux-config --get-hw-description=$(XSA_DIR) --silentconfig; \
	else \
	  petalinux-config --get-hw-description=$(XSA_DIR) --silentconfig; \
	fi
	petalinux-build

.PHONY: sdk
sdk: $(PLNX_SDK)
$(PLNX_SDK):
	@echo 'Build PetaLinux SDK'
	@./trd-pl-cfg -p $(PFM)
	@if [ $(YES) -eq 1 ]; then \
	  yes | petalinux-config --silentconfig; \
	else \
	  petalinux-config --silentconfig; \
	fi
	petalinux-build --sdk

.PHONY: clean
clean:
	-@$(RM) build components/yocto images project-spec/configs/*.old
	-@$(RM) project-spec/hw-description/{drivers,system.xsa,vck190_mipiRxSingle_hdmiTx.pdi}
	-@$(RM) $(shell grep TMP_DIR project-spec/configs/config | sed -e 's/.*"\(.*\)\"/\1/')

