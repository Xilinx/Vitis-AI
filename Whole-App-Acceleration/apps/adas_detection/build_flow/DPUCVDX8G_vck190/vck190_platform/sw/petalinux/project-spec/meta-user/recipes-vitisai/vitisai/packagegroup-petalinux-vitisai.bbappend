DESCRIPTION = "PetaLinux Vitis AI packages"

inherit packagegroup

# Since dnndk is SOC_FAMILY specific, this package must be also
PACKAGE_ARCH = "${SOC_FAMILY_ARCH}"

COMPATIBLE_MACHINE = "^$"
COMPATIBLE_MACHINE_zynqmp = ".*"
COMPATIBLE_MACHINE_versal = ".*"

RDEPENDS_${PN} = "\
    glog \
    googletest \
    json-c \
    protobuf \
    python3-pip \
    opencv \
    python3-pybind11 \
    "

RDEPENDS_${PN}-dev += "\
    protobuf-c \
    libeigen-dev \
    "
