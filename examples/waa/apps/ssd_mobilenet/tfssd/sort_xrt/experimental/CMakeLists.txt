set(XRT_EXPERIMENTAL_HEADER_SRC
  xrt-next.h)


install (FILES ${XRT_EXPERIMENTAL_HEADER_SRC} DESTINATION ${XRT_INSTALL_INCLUDE_DIR}/experimental)

message("-- XRT experimental header files")
foreach (header ${XRT_EXPERIMENTAL_HEADER_SRC})
  message("-- ${header}")
endforeach()
