FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

SRC_URI:append = " file://vaitracing.cfg"
KERNEL_FEATURES:append = " bsp.cfg"
