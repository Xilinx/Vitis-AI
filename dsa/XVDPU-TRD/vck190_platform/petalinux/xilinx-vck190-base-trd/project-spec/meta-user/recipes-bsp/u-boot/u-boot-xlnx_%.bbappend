FILESEXTRAPATHS_prepend := "${THISDIR}/files:"

SRC_URI_append = " file://platform-top.h file://bsp.cfg"

do_configure_append () {
	install ${WORKDIR}/platform-top.h ${S}/include/configs/
}

do_configure_append_microblaze () {
	if [ "${U_BOOT_AUTO_CONFIG}" = "1" ]; then
		install ${WORKDIR}/platform-auto.h ${S}/include/configs/
		install -d ${B}/source/board/xilinx/microblaze-generic/
		install ${WORKDIR}/config.mk ${B}/source/board/xilinx/microblaze-generic/
	fi
}
