DESCRIPTION = "HDMI Rx Custom EDID Binary"
LICENSE = "GPLv2"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/GPL-2.0;md5=801f80980d171dd6425610833a22dbe6"

SRC_URI = " file://xilinx-hdmi-rx-edid.bin"

S = "${WORKDIR}"

do_install () {
    install -d ${D}/lib/firmware/xilinx/
    install -m 0644 xilinx-hdmi-rx-edid.bin ${D}/lib/firmware/xilinx/
}

FILES_${PN} = "lib/firmware/xilinx/*"
