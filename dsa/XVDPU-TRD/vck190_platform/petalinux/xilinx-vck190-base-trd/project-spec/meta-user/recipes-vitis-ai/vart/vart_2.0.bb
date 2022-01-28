SUMMARY = "Vitis AI RunTime"
DESCRIPTION = "Runner is an application level runtime interface for DPU IPs based on XRT. It use XIR subgraph as input, and run it on different targets. There are also emulators implemented with same interface to make debuging eaiser."

require recipes-vitis-ai/vitis-ai-library/vitisai.inc

S = "${WORKDIR}/git/tools/Vitis-AI-Runtime/VART/vart"

DEPENDS = "json-c xir target-factory"

# By default, Vart enables vitis-flow which depends on xrt. Please remove this line if use vivado-flow
PACKAGECONFIG_append = " vitis"

PACKAGECONFIG[test] = "-DBUILD_TEST=ON,-DBUILD_TEST=OFF,opencv,"
PACKAGECONFIG[python] = "-DBUILD_PYTHON=ON -DPYTHON_INSTALL_DIR=${PYTHON_DIR},-DBUILD_PYTHON=OFF,,python3-core"
PACKAGECONFIG[vitis] = ",,xrt,"

inherit cmake python3-dir

EXTRA_OECMAKE += "-DENABLE_CPU_RUNNER=OFF -DENABLE_SIM_RUNNER=OFF -DENABLE_DPU_RUNNER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSROOT=${STAGING_DIR_HOST}"

do_configure_prepend() {

	rm -rf ${STAGING_DIR_HOST}/usr/share/cmake/XRT/
}

# Vart uses dl_open, so package the .so files in the runtime package
FILES_SOLIBSDEV = ""
INSANE_SKIP_${PN} += "dev-so"
FILES_${PN} += " \
	${libdir}/*.so \
	${PYTHON_SITEPACKAGES_DIR} \
"

SYSROOT_DIRS += "${bindir}"
