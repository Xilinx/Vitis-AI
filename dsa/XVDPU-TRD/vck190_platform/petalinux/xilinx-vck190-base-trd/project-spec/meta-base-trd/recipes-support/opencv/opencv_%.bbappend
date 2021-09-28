PACKAGECONFIG_append_class-native = " \
    libav \
    dnn \
    freetype \
    text \
    "

do_install_append() {
    sed -i '/TARGET correspondence/d' ${D}${libdir}/cmake/opencv4/OpenCVModules-release.cmake
    sed -i '/_IMPORT_CHECK_TARGETS correspondence/d' ${D}${libdir}/cmake/opencv4/OpenCVModules-release.cmake

    sed -i '/TARGET numeric/d' ${D}${libdir}/cmake/opencv4/OpenCVModules-release.cmake
    sed -i '/_IMPORT_CHECK_TARGETS numeric/d' ${D}${libdir}/cmake/opencv4/OpenCVModules-release.cmake

    sed -i '/TARGET multiview/d' ${D}${libdir}/cmake/opencv4/OpenCVModules-release.cmake
    sed -i '/_IMPORT_CHECK_TARGETS multiview/d' ${D}${libdir}/cmake/opencv4/OpenCVModules-release.cmake
}
