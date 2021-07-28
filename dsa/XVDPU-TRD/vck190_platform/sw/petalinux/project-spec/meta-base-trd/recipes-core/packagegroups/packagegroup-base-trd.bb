DESCRIPTION = "Base TRD related Packages"

inherit packagegroup

MOVIE_PACKAGES = " \
	bigbuckbunny-360p-vp9-webm \
	bigbuckbunny-480p-vp9-webm \
	"

BASE_TRD_PACKAGES = " \
	packagegroup-petalinux-vitisai \
	packagegroup-core-tools-debug \
	packagegroup-petalinux-display-debug \
	packagegroup-petalinux-gstreamer \
	packagegroup-petalinux-opencv \
	packagegroup-petalinux-python-modules \
	packagegroup-petalinux-self-hosted \
	packagegroup-petalinux-v4lutils \
	packagegroup-petalinux-xrt \
	packagegroup-python3-jupyter \
	alsa-utils \
	base-trd-notebooks \
	cmake \	
	custom-edid \
	git \
	glog \
	gstreamer1.0-python \
	jupyter-startup \
	kernel-module-hdmi \
	ldd \
	libxapm-python \
	nodejs \
	nodejs-npm \
	ntp \
	python3-dev \
	python3-periphery \
	resize-part \
	tcf-agent \
	trd-files \
	tree \
	ttf-bitstream-vera \
	tzdata \
	${MOVIE_PACKAGES} \
	"

RDEPENDS_${PN} = "${BASE_TRD_PACKAGES}"

RDEPENDS_${PN}-dev += " \
	protobuf-c \
	libeigen-dev \
	glog-dev \
	googletest-dev \
	protobuf-dev \
	json-c-dev \
	opencv-dev \
	"

