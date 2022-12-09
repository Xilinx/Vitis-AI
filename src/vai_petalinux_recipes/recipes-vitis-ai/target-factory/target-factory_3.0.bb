SUMMARY = "Target Factory"
DESCRIPTION = "A factory to manage DPU target description infos. Register targets and then you can get infos by name or fingerprint."

require recipes-vitis-ai/vitis-ai-library/vitisai.inc

BRANCH = "3.0"
SRC_URI = "git://gitenterprise.xilinx.com/VitisAI/target_factory.git;protocol=https;branch=${BRANCH}"

SRCREV = "c787d60164549b08512affc5179ba4e4f173996e"

S = "${WORKDIR}/git"

DEPENDS = "unilog protobuf-native protobuf-c"

PACKAGECONFIG[test] = "-DBUILD_TEST=ON,-DBUILD_TEST=OFF,,"
PACKAGECONFIG[python] = ",,,"

inherit cmake

EXTRA_OECMAKE += "-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON"

# target-factory contains only one shared lib and will therefore become subject to renaming
# by debian.bbclass. Prevent renaming in order to keep the package name consistent 
AUTO_LIBNAME_PKGS = ""

FILES_SOLIBSDEV = ""
INSANE_SKIP:${PN} += "dev-so"
FILES:${PN} += "${libdir}/*.so"
