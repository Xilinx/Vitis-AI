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
#ifndef _RE_ENGINE_KERNEL_HPP_
#define _RE_ENGINE_KERNEL_HPP_

#include "xf_data_analytics/text/re_engine.hpp"

#define PU_NM 16
#define INSTR_DEPTH (1 << 12)
#define CCLASS_NM 128
#define CPGP_NM 512
#define MSG_LEN 512
#define STACK_SIZE (1 << 13)

extern "C" void reEngineKernel(ap_uint<64>* cfg_buff,
                               ap_uint<64>* msg_buff,
                               ap_uint<16>* len_buff,
                               ap_uint<32>* out_buff);
#endif
