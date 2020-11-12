include(FindDoxygen)

# doxygen setting
set(DOXYGEN_CONFIG_DIR ${CMAKE_SOURCE_DIR}/doc/doxygen)
configure_file(${DOXYGEN_CONFIG_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc ALL
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
  COMMAND cp ${DOXYGEN_CONFIG_DIR}/Logo.png ${CMAKE_BINARY_DIR}/
  # optional, only necessary if you refer some images in your document.
  # COMMAND cp -av ${DOXYGEN_CONFIG_DIR}/images ${CMAKE_BINARY_DIR}/c-doc/latex
  COMMAND make -C ${CMAKE_BINARY_DIR}/c-doc/latex
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Generating Documentation with Doxygen" VERBATIM
  )
install(FILES ${CMAKE_BINARY_DIR}/c-doc/latex/refman.pdf DESTINATION ${CMAKE_SOURCE_DIR}/doc
  RENAME xir-reference-manual.pdf)
