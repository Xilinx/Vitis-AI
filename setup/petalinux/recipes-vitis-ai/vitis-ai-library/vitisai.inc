LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

BRANCH = "master"
SRC_URI = "git://github.com/Xilinx/Vitis-AI.git;protocol=https;branch=${BRANCH}"
SRCREV = "c26eae36f034d5a2f9b2a7bfe816b8c43311a4f8"

PACKAGECONFIG:append = " python"
PACKAGECONFIG:append = " test"
