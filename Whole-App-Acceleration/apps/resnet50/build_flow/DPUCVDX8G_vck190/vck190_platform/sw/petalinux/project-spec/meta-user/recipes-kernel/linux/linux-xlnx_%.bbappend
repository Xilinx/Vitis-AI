SRC_URI += "file://bsp.cfg \
            file://0001-net-ethernet-cadence-macb_main-Fix-64-bit-DMA-featur.patch \
            file://0001-64bit-support-vipp.patch \
            file://0001-mixer-patch.patch \
            file://0001-linux-kernel-sound-audio-formatter.patch \
           "
KERNEL_FEATURES_append = " bsp.cfg"
FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}:"
