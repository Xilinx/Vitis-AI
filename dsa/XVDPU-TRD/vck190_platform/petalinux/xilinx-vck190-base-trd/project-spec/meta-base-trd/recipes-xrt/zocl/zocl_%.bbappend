# Use latest version

FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}:"
SRC_URI += " \
            file://0001-patch-to-enable-64-bit-addressing-on-zocl-dma.patch \
           "
