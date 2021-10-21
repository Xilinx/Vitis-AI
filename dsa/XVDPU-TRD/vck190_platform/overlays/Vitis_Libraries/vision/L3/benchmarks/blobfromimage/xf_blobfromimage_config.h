/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef _XF_BLOBFROMIMAGE_CONFIG_
#define _XF_BLOBFROMIMAGE_CONFIG_

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "dnn/xf_preprocess.hpp"
#include "imgproc/xf_crop.hpp"
#include "imgproc/xf_cvt_color.hpp"
#include "imgproc/xf_cvt_color_1.hpp"
#include "imgproc/xf_duplicateimage.hpp"
#include "imgproc/xf_resize.hpp"
#include "xf_config_params.h"
#include <ap_int.h>
#include <hls_stream.h>

#define _XF_SYNTHESIS_ 1

#endif
