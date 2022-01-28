LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

BRANCH = "master"
SRC_URI = "git://github.com/Xilinx/Vitis-AI.git;protocol=https;branch=${BRANCH}"

SRCREV = "d02dcb6041663dbc7ecbc0c6af9fafa087a789de"

PACKAGECONFIG_append = " python"
PACKAGECONFIG_append = " test"
