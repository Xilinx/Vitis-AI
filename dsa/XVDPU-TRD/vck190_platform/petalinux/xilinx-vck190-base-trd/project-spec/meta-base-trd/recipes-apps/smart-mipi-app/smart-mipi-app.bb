SUMMARY = "Smart mipicam application"
DESCRIPTION = "Application for runing single/quad mipi sensor with Vitis AI/IVAS"
SECTION = "PETALINUX/apps"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=ba05a6f05e084b7a9f5a28a629077646"

SRC_URI = " \
    file://cmake \
    file://config \
    file://models \
    file://src \
    file://CMakeLists.txt \
    file://LICENSE \
    "

S = "${WORKDIR}"

DEPENDS = " glog ivas-accel-libs opencv"

RDEPENDS_${PN} += " \
	gst-perf \
	gstreamer1.0-plugins-bad-kms \
	gstreamer1.0-plugins-bad-mediasrcbin \
	gstreamer1.0-plugins-bad-videoparsersbad \
	gstreamer1.0-plugins-good-multifile \
	gstreamer1.0-plugins-good-video4linux2 \
	ivas-accel-libs \
	libdrm-tests \
	v4l-utils \
	alsa-utils \
	"

inherit cmake

EXTRA_OECMAKE += "-DCMAKE_BUILD_TYPE=Release"

FILES_${PN} += " \
    ${datadir} \
    "
