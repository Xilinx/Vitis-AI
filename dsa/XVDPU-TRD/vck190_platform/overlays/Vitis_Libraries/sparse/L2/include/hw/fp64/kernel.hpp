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

/**
 * @file kernel.hpp
 * @brief general definitions used by L2 kernels.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_KERNEL_HPP
#define XF_SPARSE_KERNEL_HPP

#ifndef __cplusplus
#error "SPARSE Library only works with C++."
#endif

#include <cstdint>
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#ifndef __SYNTHESIS__
#include <iostream>
#include <fstream>
#include <cstring>
#endif
#include "xf_sparse_fp64.hpp"
#include "hpc/L2/include/hw/interface.hpp"

typedef ap_uint<SPARSE_hbmMemBits> HBM_InfTyp;
typedef hls::stream<ap_uint<SPARSE_hbmMemBits> > HBM_StrTyp;
typedef hls::stream<uint32_t> ParamStrTyp;
typedef ap_uint<32 * SPARSE_hbmChannels> WideParamTyp;
typedef hls::stream<ap_uint<32 * SPARSE_hbmChannels> > WideParamStrTyp;
typedef hls::stream<ap_uint<SPARSE_indexBits> > IdxStrTyp;
typedef hls::stream<ap_uint<SPARSE_dataBits> > DatStrTyp;

#endif
