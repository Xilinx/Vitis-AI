FILESEXTRAPATHS_prepend := "${THISDIR}/files:"

SRC_URI += "file://overrides.json"

do_install_append () {
	install -d ${D}${datadir}/jupyter/lab/settings
	install -m 0644 ${WORKDIR}/overrides.json ${D}${datadir}/jupyter/lab/settings/
}

FILES_${PN}_append = "${datadir}/jupyter/lab/settings"
