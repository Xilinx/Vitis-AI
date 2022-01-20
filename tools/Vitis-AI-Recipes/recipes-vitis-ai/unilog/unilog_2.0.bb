SUMMARY = "Vitis AI UNILOG"
DESCRIPTION = "Xilinx Vitis AI components - a wrapper for glog. Define unified log formats for vitis ai tools."

require recipes-vitis-ai/vitis-ai-library/vitisai.inc

SRC_URI = "git://gitenterprise.xilinx.com/Vitis/vitis-ai-staging;protocol=https;branch=2.0 \
	file://0001-fix-python-path-for-petalinux.patch \
"

FILESEXTRAPATHS_prepend := "${THISDIR}/files:"

SRCREV = "700297f6e45c7fddfd4450adf1703ce12de4ae97"
S = "${WORKDIR}/git/tools/Vitis-AI-Runtime/VART/unilog"

DEPENDS = "glog boost"

PACKAGECONFIG[test] = "-DBUILD_TEST=ON,-DBUILD_TEST=OFF,,"
PACKAGECONFIG[python] = ",,,"

inherit cmake

EXTRA_OECMAKE += "-DCMAKE_BUILD_TYPE=Release"

# unilog contains only one shared lib and will therefore become subject to renaming
# by debian.bbclass. Prevent renaming in order to keep the package name consistent 
AUTO_LIBNAME_PKGS = ""

FILES_SOLIBSDEV = ""
INSANE_SKIP_${PN} += "dev-so"
FILES_${PN} += "${libdir}/*.so"
