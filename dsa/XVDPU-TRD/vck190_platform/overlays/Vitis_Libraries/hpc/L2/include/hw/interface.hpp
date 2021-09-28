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

#ifndef XF_HPC_INTERFACE_HPP
#define XF_HPC_INTERFACE_HPP

#define PRAGMA_HLS(x) _Pragma(#x)

#define SCALAR(NAME) PRAGMA_HLS(HLS INTERFACE s_axilite port = NAME bundle = control)

#define POINTER(NAME, BUNDLE)                                                  \
    PRAGMA_HLS(HLS INTERFACE m_axi port = NAME bundle = BUNDLE offset = slave) \
    SCALAR(NAME)

#define AXIS(NAME) PRAGMA_HLS(HLS INTERFACE axis port = NAME)
#define AP_CTRL_NONE(NAME) PRAGMA_HLS(HLS INTERFACE ap_ctrl_none port = NAME)

#endif
