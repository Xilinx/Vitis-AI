LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

BRANCH = "vai-3.5-update"
SRC_URI = "git://gitenterprise.xilinx.com/Vitis/vitis-ai-staging.git;protocol=https;branch=${BRANCH}"
SRCREV = "3e916873f010b1b6592f86f2a1f633fc6bcfbed5"

PACKAGECONFIG:append = " python"
PACKAGECONFIG:append = " test"
