FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}:"

SRC_URI += "file://0001-recipes-utils-jupyterlab-server.patch"

do_install() {

    install -d ${D}${datadir}/jupyter/lab/settings
    install -m 0644 ${WORKDIR}/overrides.json ${D}${datadir}/jupyter/lab/settings/

    install -d ${D}${sysconfdir}/init.d/
    install -m 0755 ${WORKDIR}/jupyterlab-server ${D}${sysconfdir}/init.d/jupyterlab-server

    install -d ${D}${sysconfdir}/jupyter/
    install -m 0600 ${WORKDIR}/jupyter_lab_config.py ${D}${sysconfdir}/jupyter
}
