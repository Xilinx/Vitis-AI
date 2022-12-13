SUMMARY = "Vitis AI UNILOG"
DESCRIPTION = "Xilinx Vitis AI components - a wrapper for glog. Define unified log formats for vitis ai tools."

require recipes-vitis-ai/vitis-ai-library/vitisai.inc

BRANCH = "3.0"
SRC_URI = "git://gitenterprise.xilinx.com/VitisAI/unilog.git;protocol=https;branch=${BRANCH} \
	file://0001-fix-python-path-for-petalinux.patch \
"

FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRCREV = "476ce6ad48e3b8669c9131e4996d1fb827b8247a"
S = "${WORKDIR}/git"

DEPENDS = "glog boost"

PACKAGECONFIG[test] = "-DBUILD_TEST=ON,-DBUILD_TEST=OFF,,"
PACKAGECONFIG[python] = ",,,"

inherit cmake

EXTRA_OECMAKE += "-DCMAKE_BUILD_TYPE=Release"

# unilog contains only one shared lib and will therefore become subject to renaming
# by debian.bbclass. Prevent renaming in order to keep the package name consistent 
AUTO_LIBNAME_PKGS = ""

FILES_SOLIBSDEV = ""
INSANE_SKIP:${PN} += "dev-so"
FILES:${PN} += "${libdir}/*.so"
