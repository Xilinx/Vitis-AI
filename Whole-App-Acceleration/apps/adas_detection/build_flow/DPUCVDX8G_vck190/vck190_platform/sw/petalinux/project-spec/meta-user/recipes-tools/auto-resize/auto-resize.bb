#
# This file is used to resize SD card capability
#
SUMMARY = "startup script"
SECTION = "PETALINUX/setting "
#
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://COPYRIGHT;md5=fa7ffd0ad615cced5e214402246bb709"

SRC_URI = " \
	file://auto-resize \
	file://COPYRIGHT \
"

S = "${WORKDIR}"

FILESEXTRAPATHS_prepend := "${THISDIR}/files:"

inherit update-rc.d

RDEPENDS_${PN} = "parted e2fsprogs-resize2fs"

INITSCRIPT_NAME = "auto-resize"
INITSCRIPT_PARAMS = "start 99 5 ."

do_install() {

	install -d ${D}${sysconfdir}/init.d
	install -m 0755 ${S}/auto-resize ${D}${sysconfdir}/init.d/auto-resize

}

FILES_${PN} += "${sysconfdir}/*"
