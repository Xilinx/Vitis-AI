/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _XF_PARAMS_H_
#define _XF_PARAMS_H_

#ifndef __cplusplus
#error C++ is needed to use this file!
#endif

#include "ap_int.h"

#define __ABS(X) ((X) < 0 ? (-(X)) : (X))

// Channels of an image
enum _channel_extract {
    XF_EXTRACT_CH_0, // Used by formats with unknown channel types
    XF_EXTRACT_CH_1, // Used by formats with unknown channel types
    XF_EXTRACT_CH_2, // Used by formats with unknown channel types
    XF_EXTRACT_CH_3, // Used by formats with unknown channel types
    XF_EXTRACT_CH_R, // Used to extract the RED channel
    XF_EXTRACT_CH_G, // Used to extract the GREEN channel
    XF_EXTRACT_CH_B, // Used to extract the BLUE channel
    XF_EXTRACT_CH_A, // Used to extract the ALPHA channel
    XF_EXTRACT_CH_Y, // Used to extract the LUMA channel
    XF_EXTRACT_CH_U, // Used to extract the Cb/U channel
    XF_EXTRACT_CH_V  // Used to extract the Cr/V/Value channel
};
typedef _channel_extract XF_channel_extract_e;

// Conversion Policy for fixed point arithmetic
enum _convert_policy { XF_CONVERT_POLICY_SATURATE, XF_CONVERT_POLICY_TRUNCATE };
typedef _convert_policy XF_convert_policy_e;

// Bit-depth conversion types
enum _convert_bit_depth {
    // Down-convert
    XF_CONVERT_16U_TO_8U,
    XF_CONVERT_16S_TO_8U,
    XF_CONVERT_32S_TO_8U,
    XF_CONVERT_32S_TO_16U,
    XF_CONVERT_32S_TO_16S,
    // Up-convert
    XF_CONVERT_8U_TO_16U,
    XF_CONVERT_8U_TO_16S,
    XF_CONVERT_8U_TO_32S,
    XF_CONVERT_16U_TO_32S,
    XF_CONVERT_16S_TO_32S
};
typedef _convert_bit_depth XF_convert_bit_depth_e;

// Thresholding types
enum _threshold_type {
    XF_THRESHOLD_TYPE_BINARY = 0,
    XF_THRESHOLD_TYPE_BINARY_INV = 1,
    XF_THRESHOLD_TYPE_TRUNC = 2,
    XF_THRESHOLD_TYPE_TOZERO = 3,
    XF_THRESHOLD_TYPE_TOZERO_INV = 4,
};
typedef _threshold_type XF_threshold_type_e;

// Thresholding types
enum _ccm_type {
    XF_CCM_bt2020_bt709 = 0,
    XF_CCM_bt709_bt2020 = 1,
    XF_CCM_rgb_yuv_601 = 2,
    XF_CCM_rgb_yuv_709 = 3,
    XF_CCM_rgb_yuv_2020 = 4,
    XF_CCM_yuv_rgb_601 = 5,
    XF_CCM_yuv_rgb_709 = 6,
    XF_CCM_yuv_rgb_2020 = 7,
    XF_CCM_full_from_16_235 = 8,
    XF_CCM_full_to_16_235 = 9,

};
typedef _ccm_type XF_ccm_type_e;

// Comparision types
enum _comparison_op { XF_CMP_EQ = 0, XF_CMP_GT = 1, XF_CMP_GE = 2, XF_CMP_LT = 3, XF_CMP_LE = 4, XF_CMP_NE = 5 };
typedef _comparison_op _comparison_op_e;

// Comparision types
enum _reduction_op { REDUCE_SUM = 0, REDUCE_AVG = 1, REDUCE_MAX = 2, REDUCE_MIN = 3 };
typedef _reduction_op _reduction_op_e;

// Pixel Per Cycle
enum _pixel_per_cycle {
    XF_NPPC1 = 1,
    XF_NPPC2 = 2,
    XF_NPPC4 = 4,
    XF_NPPC8 = 8,
    XF_NPPC16 = 16,
    XF_NPPC32 = 32,
    XF_NPPC64 = 64
};
typedef _pixel_per_cycle XF_nppc_e;

// Pixel types
enum _pixel_type {
    XF_8UP = 0,
    XF_8SP = 1,
    XF_16UP = 2,
    XF_16SP = 3,
    XF_32UP = 4,
    XF_32SP = 5,
    XF_19SP = 6,
    XF_32FP = 7,
    XF_35SP = 8,
    XF_24SP = 9,
    XF_20SP = 10,
    XF_48SP = 11,
    XF_2UP = 12,
    XF_9SP = 13,
    XF_9UP = 14,
    XF_24UP = 15,
    XF_64UP = 16,
    XF_10UP = 17,
    XF_12UP = 18,
    XF_40UP = 19,
    XF_48UP = 20,
    XF_30UP = 21,
    XF_36UP = 22,
    XF_96FP = 23
};
typedef _pixel_type XF_pixel_type_e;

// Word width
enum _word_width {
    XF_2UW = 0,
    XF_8UW = 1,
    XF_9UW = 2,
    XF_10UW = 3,
    XF_12UW = 4,
    XF_16UW = 5,
    XF_19SW = 6,
    XF_20UW = 7,
    XF_22UW = 8,
    XF_24UW = 9,
    XF_24SW = 10,
    XF_30UW = 11,
    XF_32UW = 12,
    XF_32FW = 13,
    XF_35SW = 14,
    XF_36UW = 15,
    XF_40UW = 16,
    XF_48UW = 17,
    XF_48SW = 18,
    XF_60UW = 19,
    XF_64UW = 20,
    XF_72UW = 21,
    XF_80UW = 22,
    XF_96UW = 23,
    XF_96SW = 24,
    XF_120UW = 25,
    XF_128UW = 26,
    XF_144UW = 27,
    XF_152SW = 28,
    XF_160UW = 29,
    XF_160SW = 30,
    XF_176UW = 31,
    XF_192UW = 32,
    XF_192SW = 33,
    XF_240UW = 34,
    XF_256UW = 35,
    XF_280SW = 36,
    XF_288UW = 37,
    XF_304SW = 38,
    XF_320UW = 39,
    XF_352UW = 40,
    XF_384UW = 41,
    XF_384SW = 42,
    XF_512UW = 43,
    XF_560SW = 44,
    XF_576UW = 45,
    XF_96FW = 46,
    XF_192FW = 47,
    XF_384FW = 48,
    XF_768FW = 49,
    XF_1536FW = 50
};
typedef _word_width XF_word_width_e;

// Filter size
enum _filter_size { XF_FILTER_3X3 = 3, XF_FILTER_5X5 = 5, XF_FILTER_7X7 = 7 };
typedef _filter_size XF_filter_size_e;

// Radius size for Non Maximum Suppression
enum _nms_radius { XF_NMS_RADIUS_1 = 1, XF_NMS_RADIUS_2 = 2, XF_NMS_RADIUS_3 = 3 };
typedef _nms_radius XF_nms_radius_e;

// Image Pyramid Parameters
enum _image_pyramid_params {
    XF_PYRAMID_TYPE_GXFSSIAN = 0,
    XF_PYRAMID_TYPE_LAPLACIAN = 1,
    XF_PYRAMID_SCALE_HALF = 2,
    XF_PYRAMID_SCALE_ORB = 3,
    XF_PYRAMID_SCALE_DOUBLE = 4
};
typedef _image_pyramid_params XF_image_pyramid_params_e;

// Magnitude computation
enum _normalisation_params { XF_L1NORM = 0, XF_L2NORM = 1 };
typedef _normalisation_params XF_normalisation_params_e;

enum _border_type {
    XF_BORDER_CONSTANT = 0,
    XF_BORDER_REPLICATE = 1,
    XF_BORDER_REFLECT = 2,
    XF_BORDER_WRAP = 3,
    XF_BORDER_REFLECT_101 = 4,
    XF_BORDER_TRANSPARENT = 5,
    XF_BORDER_REFLECT101 = XF_BORDER_REFLECT_101,
    XF_BORDER_DEFAULT = XF_BORDER_REFLECT_101,
    XF_BORDER_ISOLATED = 16,
};
typedef _border_type XF_border_type_e;

enum _structuring_element_shape {
    XF_SHAPE_RECT = 0,
    XF_SHAPE_ELLIPSE = 1,
    XF_SHAPE_CROSS = 2,

};
enum _wb_type {
    XF_WB_GRAY = 0,
    XF_WB_SIMPLE = 1,
};

// Phase computation
enum _phase_params { XF_RADIANS = 0, XF_DEGREES = 1 };
typedef _phase_params XF_phase_params_e;

// Types of Interpolaton techniques used in resize, affine and perspective
enum _interpolation_types { XF_INTERPOLATION_NN = 0, XF_INTERPOLATION_BILINEAR = 1, XF_INTERPOLATION_AREA = 2 };
typedef _interpolation_types _interpolation_types_e;

// loop dependent variables used in image pyramid
enum _loop_dependent_vars { XF_GXFSSIANLOOP = 8, XF_BUFSIZE = 12 };
typedef _loop_dependent_vars loop_dependent_vars_e;

// loop dependent variables used in image pyramid
enum _image_size { XF_SDIMAGE = 0, XF_HDIMAGE = 1 };
typedef _image_size image_size_e;

// enumerations for HOG feature descriptor
enum _input_image_type { XF_GRAY = 1, XF_RGB = 3 };
typedef _input_image_type input_image_type_e;

// enumerations for HOG feature descriptor
enum _HOG_output_type { XF_HOG_RB = 0, XF_HOG_NRB = 1 };
typedef _HOG_output_type HOG_output_type_e;

enum use_model { XF_STANDALONE = 0, XF_PIPELINE = 1 };
typedef use_model use_model_e;

// enumerations for HOG feature descriptor
enum _HOG_type { XF_DHOG = 0, XF_SHOG = 1 };
typedef _HOG_type HOG_type_e;

// enumerations for Stereo BM
enum XF_stereo_prefilter_type { XF_STEREO_PREFILTER_SOBEL_TYPE, XF_STEREO_PREFILTER_NORM_TYPE };
/****************************new************************/
// enumerations for Demosaicing
enum XF_demosaicing {
    XF_BAYER_BG,
    XF_BAYER_GB,
    XF_BAYER_GR,
    XF_BAYER_RG,
};
// typedef XF_stereo_prefilter_type XF_stereo_pre_filter_type_e;
// enum _pixel_percycle
//{
//	XF_NPPC1  = 0,
//	XF_NPPC8  = 3,
//	XF_NPPC16 = 4
//};
// typedef _pixel_percycle XF_nppc_e;

// enumerations for Architecture
enum _ARCH_type {
    XF_STREAM = 0,
    XF_MEMORYMAPPED = 1

};
typedef _ARCH_type _ARCH_type_e;

enum _pixeltype {
    XF_8UC1 = 0,
    XF_16UC1 = 1,
    XF_16SC1 = 2,
    XF_32UC1 = 3,
    XF_32FC1 = 4,
    XF_32SC1 = 5,
    XF_8UC2 = 6,
    XF_8UC4 = 7,
    XF_2UC1 = 8,
    XF_8UC3 = 9,
    XF_16UC3 = 10,
    XF_16SC3 = 11,
    XF_16UC4 = 12,
    XF_10UC1 = 13,
    XF_10UC4 = 14,
    XF_12UC1 = 15,
    XF_12UC4 = 16,
    XF_10UC3 = 17,
    XF_12UC3 = 18,
    XF_32FC3 = 19
};
typedef _pixeltype XF_npt_e;

enum _ramtype {
    RAM_1P_BRAM = 0,
    RAM_1P_LUTRAM = 1,
    RAM_1P_URAM = 2,
    RAM_2P_BRAM = 3,
    RAM_2P_LUTRAM = 4,
    RAM_2P_URAM = 5,
    RAM_S2P_BRAM = 6,
    RAM_S2P_LUTRAM = 7,
    RAM_S2P_URAM = 8,
    RAM_T2P_BRAM = 9,
    RAM_T2P_URAM = 10
};
typedef _ramtype XF_ramtype_e;

#endif //_XF_PARAMS_H_
