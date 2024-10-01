SUMMARY = "Packages For VAI Vitis-AI-library and VART Examples Test."

LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

BRANCH = "master"
SRC_URI = "git://github.com/Xilinx/Vitis-AI.git;protocol=https;branch=${BRANCH}"
SRC_URI += "file://0001-add-CMakeLists.txt.patch"

SRCREV = "bf7cca32b0f45144531a77a9a2a6fa8e4bcb6c4b"
S = "${WORKDIR}/git/examples"

DEPENDS:append = " vitis-ai-library"
DEPENDS:append = " protobuf-native pkgconfig-native"
RDEPENDS:${PN} += " python3-pillow python3-imageio python3-protobuf"

inherit cmake

VAI_DEMO_INSTALL_PATH = "/home/root/Vitis-AI/examples"

EXTRA_OECMAKE += " -DINSTALL_PATH=${VAI_DEMO_INSTALL_PATH} -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSROOT=${STAGING_DIR_HOST}"

do_install:append() {

	mkdir -p ${D}/${VAI_DEMO_INSTALL_PATH}

	# should copy source code into ${VAI_DEMO_INSTALL_PATH} for target build
	cp ${S}/vai_runtime/ ${D}/${VAI_DEMO_INSTALL_PATH} -r
	cp ${S}/custom_operator/ ${D}/${VAI_DEMO_INSTALL_PATH} -r
	rm -rf ${S}/vai_library/vai_benchmark
	cp ${S}/vai_library/ ${D}/${VAI_DEMO_INSTALL_PATH} -r

	# delete useless files
	find ${D}/${VAI_DEMO_INSTALL_PATH} -type f -name "CMakeLists.txt" -delete
}

FILES:${PN} += "${VAI_DEMO_INSTALL_PATH}"
