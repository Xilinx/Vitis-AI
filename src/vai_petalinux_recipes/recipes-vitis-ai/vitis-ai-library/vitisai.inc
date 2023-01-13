LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

BRANCH = "master"
SRC_URI = "git://github.com/Xilinx/Vitis-AI.git;protocol=https;branch=${BRANCH}"

SRCREV = "c5d2bd43d951c174185d728b8e5bcda3869e0b39"

PACKAGECONFIG:append = " python"
PACKAGECONFIG:append = " test"
