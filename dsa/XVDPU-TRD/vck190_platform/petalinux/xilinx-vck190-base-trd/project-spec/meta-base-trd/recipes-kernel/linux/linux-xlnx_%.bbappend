FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}:"
KERNEL_FEATURES_append = " bsp.cfg"

SRC_URI += " \
	file://0001-i2c-free_clk.patch \
	file://0003-drm-xlnx_mixer-Dont-enable-primary-plane-by-default.patch \
	file://bsp.cfg \
	file://0001-media-xilinx-Add-isppipeline-driver.patch \
	file://0001-arm-zynq-Add-MAX20087-driver.patch \
	file://0002-max20087-Remove-unused-members.patch \
	file://0003-media-i2c-max9286-fix-access-to-unallocated-memory.patch \
	file://0004-media-i2c-max9286-Break-out-reverse-channel-setup.patch \
	file://0005-media-i2c-max9286-Make-channel-amplitude-programmabl.patch \
	file://0006-media-i2c-max9286-Configure-reverse-channel-amplitud.patch \
	file://0007-media-i2c-Add-MARS-driver.patch \
	file://0008-media-i2c-max9286-add-support-for-port-regulators.patch \
	file://0009-max9286-WIP-modifications-for-mars-module.patch \
	file://0010-max9286-WIP-fix-endpoint-fwnode.patch \
	file://0011-max9286-WIP-modify-resolution.patch \
	file://0012-xilinx-vipp-WIP-blacklist-mars-module.patch \
	file://0001-media-i2c-max96705-fix-color-issue-due-to-bad-frame-.patch \
	file://0001-enable-primary-layer-default.patch \
"
