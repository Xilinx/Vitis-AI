SUMMARY = "Big Buck Bunny - 360p VP9 webm"
LICENSE = "CC-BY-3.0"
LIC_FILES_CHKSUM = "file://${COREBASE}/meta/files/common-licenses/CC-BY-3.0;md5=dfa02b5755629022e267f10b9c0a2ab7"

SRC_URI = "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.360p.vp9.webm"
SRC_URI[md5sum] = "3f5b3b10f69a1676b17317530ff9e1cb"
SRC_URI[sha256sum] = "492f23f851afe0c570e5d2c436a9085b5c718d5bec5d31d4e66ebe308336ef00"

inherit allarch

do_install() {
    install -d ${D}${datadir}/movies
    install -m 0644 ${WORKDIR}/Big_Buck_Bunny_4K.webm.360p.vp9.webm ${D}${datadir}/movies/
}

FILES_${PN} += "${datadir}/movies"
