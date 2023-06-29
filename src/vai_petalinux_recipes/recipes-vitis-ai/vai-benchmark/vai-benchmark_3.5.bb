SUMMARY = "Test for benchmark"
DESCRIPTION = "Tools For Xilinx Vitis AI benchmark test"

LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

SRC_URI = "git://gitenterprise.xilinx.com/Vitis/vitis-ai-staging.git;protocol=https;branch=vai-3.5-update"
SRCREV = "78c7d7f09769e49de38984819451975f13e01d8a"

S = "${WORKDIR}/git/examples/vai_library/vai_benchmark"

inherit setuptools3

