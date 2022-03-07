#
# This is the libxapm recipe
#

SUMMARY = "AXI performance monitor library with python bindings"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=b93c79fbbdb1cf9bfef6f39d93a5aaa0"

DEPENDS += " \
	boost \
	python3 \
	"

SRC_URI = " \
	file://inc \
	file://LICENSE \
	file://Makefile \
	file://src \
	"

S = "${WORKDIR}"

inherit python3native

TARGET_CC_ARCH += "${LDFLAGS}"
TARGET_CFLAGS += "-I${PYTHON_INCLUDE_DIR} -I${s}/inc"

do_install() {
	install -d ${D}${includedir}
	install -d -m 0655 ${D}${includedir}/libxperfmon
	install -m 0644 ${S}/inc/*.hpp ${D}${includedir}/libxperfmon/

	install -d ${D}${libdir}
	oe_libinstall -so libxperfmon ${D}${libdir}

	install -d ${D}${PYTHON_SITEPACKAGES_DIR}
	install -m 755 libxperfmon.so.0.1.0 ${D}${PYTHON_SITEPACKAGES_DIR}/libxperfmon.so
}

PACKAGES =+ "${PN}-python"
FILES_${PN}-python += "${PYTHON_SITEPACKAGES_DIR}"
