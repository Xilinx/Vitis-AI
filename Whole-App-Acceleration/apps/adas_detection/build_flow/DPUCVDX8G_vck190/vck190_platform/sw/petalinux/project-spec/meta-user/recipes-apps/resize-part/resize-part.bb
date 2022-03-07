#
# This recipe is used to resize SD card partition.
#

SUMMARY = "Resize sd card partition application"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
	file://resize-part \
"

S = "${WORKDIR}"

INSANE_SKIP_${PN} += "installed-vs-shipped"

FILESEXTRAPATHS_prepend := "${THISDIR}/files:"

RDEPENDS_${PN} = "parted e2fsprogs-resize2fs"

COMPATIBLE_MACHINE = "^$"
COMPATIBLE_MACHINE_zynq = "zynq"
COMPATIBLE_MACHINE_zynqmp = "zynqmp"
COMPATIBLE_MACHINE_versal = "versal"


do_install() {
	install -d ${D}/${bindir}
	install -d ${D}/${datadir}/resize_fs
	install -m 0755 ${S}/resize-part ${D}/${datadir}/resize_fs
	ln -sf -r ${D}${datadir}/resize_fs/resize-part ${D}/${bindir}/resize-part
}

FILES_${PN} += "${datadir}/*"
