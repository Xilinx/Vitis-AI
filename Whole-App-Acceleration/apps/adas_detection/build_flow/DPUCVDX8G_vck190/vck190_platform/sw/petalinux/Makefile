check-petalinux:
ifeq ($(PETALINUX),)
    $(error ERROR: 'PETALINUX' variable not set, please set correctly and rerun)
endif

refresh_hw:
	petalinux-config --get-hw-description=${XSA_DIR} --silentconfig

linux:
	petalinux-build

sysroot:
	mkdir -p ${SYSROOT}
	petalinux-build --sdk
	petalinux-package --sysroot -d ${SYSROOT}

sw_comp:
	mkdir -p $(SW_COMP_DIR)
	cp -rf $(PLATFORM_SW_SRC) ${SW_COMP_DIR}
	cp -f images/linux/boot.scr ${SW_COMP_DIR}/platform/image/boot.scr
	cp -f images/linux/bl31.elf ${SW_COMP_DIR}/platform/boot/bl31.elf
	cp -f images/linux/u-boot.elf ${SW_COMP_DIR}/platform/boot/u-boot.elf
	cp -f images/linux/system.dtb ${SW_COMP_DIR}/platform/boot/system.dtb

image_rootfs:
	cp -f images/linux/Image ${SW_COMP_DIR}/platform/image/Image
	mkdir -p ${SW_COMP_DIR}/platform/filesystem
	cp -f images/linux/rootfs.tar.gz ${SW_COMP_DIR}/platform/filesystem/rootfs.tar.gz
	cp -f images/linux/rootfs.ext4 ${SW_COMP_DIR}/platform/filesystem/rootfs.ext4

all: refresh_hw linux sw_comp image_rootfs

bootimage:
	$(info "BOOT image for platforms")
	@if [ -f project-spec/hw-description/$(PLATFORM_NAME).pdi ]; then \
		echo "INFO: BOOT image generation started...";  \
		petalinux-package --boot --u-boot --plm no --psmfw no --qemu-rootfs no --force;  \
		cp -rf images/linux/BOOT.BIN ${SW_COMP_DIR}/platform/;  \
	else  \
		echo "WARNING: "$(PLATFORM_NAME).pdi" file not found. Skipping BOOT image generation.";  \
	fi

clean: check-petalinux
	petalinux-build -x cleanall
	${RM} -r images sysroot
	${RM} -r build ip_cache
	${RM} -r components/yocto/
