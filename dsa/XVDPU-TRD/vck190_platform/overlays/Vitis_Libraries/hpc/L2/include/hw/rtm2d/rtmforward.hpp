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
#include "xf_blas.hpp"
#include "rtm.hpp"
#include "streamOps.hpp"

using namespace xf::hpc;
using namespace xf::hpc::rtm;
typedef RTM2D<RTM_dataType, RTM_order, RTM_maxDim, RTM_MaxB, RTM_nPE> RTM_TYPE;
typedef RTM_TYPE::t_PairInType RTM_pairType;
typedef RTM_TYPE::t_InType RTM_vtType;
typedef RTM_TYPE::t_UpbInType RTM_upbType;

typedef xf::blas::WideType<RTM_dataType, RTM_parEntries> RTM_wideType;
typedef RTM_wideType::t_TypeInt RTM_interface;

void forward(RTM_TYPE s[RTM_numFSMs],
             const unsigned int p_t,
             const RTM_dataType* p_src,
             const RTM_interface* p_v2dt2,
             RTM_upbType* p_upb,
             RTM_interface* p_p0,
             RTM_interface* p_p1) {
    const int l_width = s[0].getX();
    const int l_entries = s[0].getArea();
    const int l_size = s[0].getX() * s[0].getZ();
    const int l_sizeVt = l_size / RTM_parEntries;
    const int l_sizeP = l_size / RTM_parEntries * 2;
    const unsigned int RTM_pairTypeWidth = RTM_TYPE::t_PairType::t_TypeWidth;
    const unsigned int RTM_vtTypeWidth = RTM_TYPE::t_WideType::t_TypeWidth;
    const unsigned int RTM_multi = RTM_wideType::t_TypeWidth / RTM_pairTypeWidth;

    hls::stream<RTM_vtType> l_vt[RTM_numFSMs + 1];
#pragma HLS ARRAY_PARTITION variable = l_vt complete dim = 1
#pragma HLS stream depth = RTM_TYPE::t_FifoDepth variable = l_vt
#pragma HLS RESOURCE variable = l_vt core = RAM_2P_URAM

    hls::stream<RTM_upbType> l_upb[RTM_numFSMs];
#pragma HLS ARRAY_PARTITION variable = l_upb complete dim = 1

    hls::stream<RTM_interface> l_pin;
#pragma HLS stream variable = l_pin depth = 4
    hls::stream<RTM_interface> l_pout;
#pragma HLS stream variable = l_pout depth = 4
    hls::stream<RTM_interface> l_v2dt2;
#pragma HLS stream variable = l_v2dt2 depth = 4

    hls::stream<RTM_pairType> l_p[RTM_numFSMs + 1];
#pragma HLS ARRAY_PARTITION variable = l_p complete dim = 1

#pragma HLS DATAFLOW

    xf::blas::mem2stream(l_sizeP, p_p0, l_pin);
    wide2stream<RTM_pairTypeWidth, RTM_multi>(l_sizeP, l_pin, l_p[0]);

    xf::blas::mem2stream(l_sizeVt, p_v2dt2, l_v2dt2);
    wide2stream<RTM_vtTypeWidth, RTM_multi << 1>(l_sizeVt, l_v2dt2, l_vt[0]);

    for (int i = 0; i < RTM_numFSMs; i++) {
#pragma HLS UNROLL
        s[i].forward(p_src[i], l_upb[i], l_vt[i], l_vt[i + 1], l_p[i], l_p[i + 1]);
    }

    xf::hpc::dataConsumer(l_entries, l_vt[RTM_numFSMs]);

    RTM_TYPE::saveUpb<RTM_numFSMs>(l_width, p_t, l_upb, p_upb);
    stream2wide<RTM_pairTypeWidth, RTM_multi>(l_sizeP, l_p[RTM_numFSMs], l_pout);
    xf::blas::stream2mem<RTM_interface>(l_sizeP, l_pout, p_p1);
}

/**
 * @brief rfmforward kernel function
 *
 * @param p_z is the number of grids along detecting depth
 * @param p_x is the number of grids along detecting width
 * @param p_t is the number of detecting time parititons
 *
 * @param p_srcz is the source z coordinate
 * @param p_srcx is the source x coordinate
 * @param p_src is the source wavefiled
 *
 * @param p_coefz is the laplacian z-direction coefficients
 * @param p_coefx is the laplacian x-direction coefficients
 * @param p_taperz is the absorbing factor along z
 * @param p_taperx is the absorbing factor along x
 *
 * @param p_v2dt2 is the velocity model v^2 * dt^2
 *
 * @param p_p0 is the first input memory of source wavefield
 * @param p_p1 is the second input memory of source wavefield
 * @param p_upb is the uppper bounday wavefiled
 */

extern "C" void rtmforward(const unsigned int p_z,
                           const unsigned int p_x,
                           const unsigned int p_t,
                           const unsigned int p_srcz,
                           const unsigned int p_srcx,
                           const RTM_dataType* p_src,
                           const RTM_dataType* p_coefz,
                           const RTM_dataType* p_coefx,
                           const RTM_dataType* p_taperz,
                           const RTM_dataType* p_taperx,
                           const RTM_interface* p_v2dt2,
                           RTM_interface* p_p0,
                           RTM_interface* p_p1,
                           RTM_upbType* p_upb);
