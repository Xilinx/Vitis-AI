#
# This is the jupyter-lab startup daemon
#

SUMMARY = "Start Jupyter-lab server at system boot"

SRC_URI = " \
	file://jupyterlab-server \
	file://jupyter_notebook_config.py \
	"

LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

JUPYTER_STARTUP_PACKAGES += " \
	python3-jupyterlab \
	bash \
	procps \
	"

RDEPENDS_${PN} = " ${JUPYTER_STARTUP_PACKAGES}"

inherit update-rc.d

INITSCRIPT_NAME = "jupyterlab-server"
INITSCRIPT_PARAMS = "start 99 3 5 . stop 20 0 1 2 6 ."

S = "${WORKDIR}"

do_install() {
    install -d ${D}${sysconfdir}/init.d/
    install -m 0755 ${WORKDIR}/jupyterlab-server ${D}${sysconfdir}/init.d/jupyterlab-server
 
    install -d ${D}${sysconfdir}/jupyter/
    install -m 0600 ${WORKDIR}/jupyter_notebook_config.py ${D}${sysconfdir}/jupyter
}
