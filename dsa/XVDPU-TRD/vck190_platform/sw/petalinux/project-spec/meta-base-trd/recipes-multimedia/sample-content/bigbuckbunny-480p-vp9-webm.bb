SUMMARY = "Big Buck Bunny - 480p VP9 webm"
LICENSE = "CC-BY-3.0"
LIC_FILES_CHKSUM = "file://${COREBASE}/meta/files/common-licenses/CC-BY-3.0;md5=dfa02b5755629022e267f10b9c0a2ab7"

SRC_URI = "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.480p.vp9.webm"
SRC_URI[md5sum] = "e0b252523770ca098692803ad52f95d6"
SRC_URI[sha256sum] = "6d6c5ab5862cadec8c61cdba8044526bff3350b715657c7c3a9e575c491769d0"

inherit allarch

do_install() {
    install -d ${D}${datadir}/movies
    install -m 0644 ${WORKDIR}/Big_Buck_Bunny_4K.webm.480p.vp9.webm ${D}${datadir}/movies/
}

FILES_${PN} += "${datadir}/movies"
