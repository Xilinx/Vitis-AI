SUMMARY = "Test for benchmark"
DESCRIPTION = "Tools For Xilinx Vitis AI benchmark test"

LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

SRC_URI = "git://github.com/Xilinx/Vitis-AI.git;protocol=https;branch=master"
SRCREV = "b7953a2a9f60e23efdfced5c186328dd1449665c"

S = "${WORKDIR}/git/examples/vai_library/vai_benchmark"

inherit setuptools3

