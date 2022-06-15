LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

BRANCH = "2.5"
SRC_URI = "git://gitenterprise.xilinx.com/Vitis/vitis-ai-staging;protocol=https;branch=${BRANCH}"
SRCREV = "da716d28aa31c5371ec9759bf8bcc5ca11176fdf"

PACKAGECONFIG:append = " python"
PACKAGECONFIG:append = " test"
