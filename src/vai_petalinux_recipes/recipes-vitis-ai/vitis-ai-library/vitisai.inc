LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

BRANCH = "master"
SRC_URI = "git://github.com/Xilinx/Vitis-AI.git;protocol=https;branch=${BRANCH}"
SRCREV = "b7953a2a9f60e23efdfced5c186328dd1449665c"

PACKAGECONFIG:append = " python"
PACKAGECONFIG:append = " test"
