FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}:"

do_install_append () {
	echo "alias ls='ls --color=auto'" >> ${D}${sysconfdir}/profile
	echo "alias ll='ls --color=auto -l'" >> ${D}${sysconfdir}/profile
	echo "alias la='ls --color=auto -la'" >> ${D}${sysconfdir}/profile
	echo "alias modetest='modetest -M xlnx'" >> ${D}${sysconfdir}/profile
	echo "export TERM=xterm" >> ${D}${sysconfdir}/profile
}
