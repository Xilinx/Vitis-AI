#
# This is the base-trd recipe
#
#

SUMMARY = "Base TRD Jupyter notebooks"
SECTION = "PETALINUX/apps"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=ba05a6f05e084b7a9f5a28a629077646"

S = "${WORKDIR}"

SRC_URI = " \
	file://cmake \
	file://ivas-accel-sw-libs \
	file://notebooks \
	file://CMakeLists.txt \
	file://LICENSE \
	"

inherit cmake

DEPENDS += " \
	ivas-utils \
	jansson \
	opencv \
	"

RDEPENDS_${PN} += " \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-plugins-good \
	gstreamer1.0-python \
	ivas-accel-libs \
	ivas-gst \
	python3-notebook \
	python3-opencv \
	python3-pydot \
	"

FILES_${PN} += " \
	${libdir} \
	${datadir} \
	"
