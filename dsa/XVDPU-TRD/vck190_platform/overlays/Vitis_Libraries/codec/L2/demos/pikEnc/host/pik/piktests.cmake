# Copyright 2019 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

set(PIK_TESTS
  ac_predictions_test
  adaptive_reconstruction_test
  ans_encode_test
  ans_test
  approx_cube_root_test
  bit_reader_test
  bits_test
  brotli_test
  byte_order_test
  chroma_from_luma_test
  codec_impl_test
  color_encoding_test
  color_management_test
  compressed_image_test
  convolve_test
  data_parallel_test
  dc_predictor_test
  dct_test
  dct_util_test
  deconvolve_test
  descriptive_statistics_test
  entropy_coder_test
  epf_test
  external_image_test
  fields_test
  gaborish_test
  gamma_correct_test
  gradient_test
  headers_test
  image_test
  linalg_test
  lossless8_test
  lossless16_test
  opsin_image_test
  opsin_inverse_test
  optimize_test
  padded_bytes_test
  pik_test
  quantizer_test
  rational_polynomial_test
  resample_test
  robust_statistics_test
  yuv_convert_test
  yuv_opsin_convert_test
)
foreach (TEST IN LISTS PIK_TESTS)
  add_executable("${TEST}" "${CMAKE_CURRENT_LIST_DIR}/${TEST}.cc")
  target_compile_definitions("${TEST}" PRIVATE -DTEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/third_party/testdata")
  target_link_libraries("${TEST}" pikcommon gmock gtest gtest_main)
  gtest_add_tests(TARGET "${TEST}")
endforeach ()
