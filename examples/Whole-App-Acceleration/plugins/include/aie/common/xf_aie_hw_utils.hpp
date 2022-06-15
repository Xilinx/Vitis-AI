/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef _XF_AIE_HW_UTILS_H_
#define _XF_AIE_HW_UTILS_H_

#include <aie_api/aie.hpp>
#include <common/xf_aie_const.hpp>
#include <common/xf_aie_utils.hpp>

namespace xf {
namespace cv {
namespace aie {

// Utility functions which can be used only inside kernel programs
inline void xfCopyMetaData(void* img_in_ptr, void* img_out_ptr) {
    ::aie::store_v(((metadata_elem_t*)img_out_ptr), ::aie::load_v<METADATA_ELEMENTS>(((metadata_elem_t*)img_in_ptr)));
    return;
}
}
}
}
#endif
