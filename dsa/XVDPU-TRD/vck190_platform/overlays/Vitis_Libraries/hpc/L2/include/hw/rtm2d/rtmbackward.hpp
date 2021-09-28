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

void backward(RTM_TYPE sf[RTM_numBSMs],
              RTM_TYPE sr[RTM_numBSMs],
              const unsigned int p_t,
              const unsigned int p_T,
              const RTM_interface* p_v2dt2,
              const RTM_dataType* p_rec,
              const RTM_upbType* p_upb,
              RTM_interface* p_p0,
              RTM_interface* p_p1,
              RTM_interface* p_r0,
              RTM_interface* p_r1,
              RTM_interface* p_i0,
              RTM_interface* p_i1) {
    const unsigned int l_num = sr[0].getArea();
    const unsigned int l_size = sr[0].getX() * sr[0].getZ();
    const unsigned int l_upbSize = sr[0].getX();
    const unsigned int l_recSize = sr[0].getX() - 2 * sr[0].getXB();
    const unsigned int l_imgSize = l_recSize * (sr[0].getZ() - 2 * sr[0].getZB());

    const int l_sizeVt = l_size / RTM_parEntries;
    const int l_sizeI = l_imgSize / RTM_parEntries;
    const int l_sizeP = l_size / RTM_parEntries * 2;
    const int l_t = p_t * RTM_numBSMs;

    const unsigned int RTM_pairTypeWidth = RTM_TYPE::t_PairType::t_TypeWidth;
    const unsigned int RTM_vtTypeWidth = RTM_TYPE::t_WideType::t_TypeWidth;
    const unsigned int RTM_multi = RTM_wideType::t_TypeWidth / RTM_pairTypeWidth;

#pragma HLS DATAFLOW
    hls::stream<RTM_vtType> l_cp[RTM_numBSMs], l_cr[RTM_numBSMs];
#pragma HLS ARRAY_PARTITION variable = l_cp dim = 1 complete
#pragma HLS stream variable = l_cp depth = 64
#pragma HLS ARRAY_PARTITION variable = l_cr dim = 1 complete
#pragma HLS stream variable = l_cr depth = 64

    // Velocity model
    hls::stream<RTM_interface> l_v2dt2;
#pragma HLS stream variable = l_v2dt2 depth = 4
    hls::stream<RTM_vtType> l_vt, l_pvt[RTM_numBSMs + 1], l_rvt[RTM_numBSMs + 1];
#pragma HLS ARRAY_PARTITION variable = l_pvt dim = 1 complete
#pragma HLS stream depth = RTM_TYPE::t_FifoDepth variable = l_pvt
#pragma HLS RESOURCE variable = l_pvt core = RAM_2P_URAM
#pragma HLS ARRAY_PARTITION variable = l_rvt dim = 1 complete
#pragma HLS stream depth = RTM_TYPE::t_FifoDepth variable = l_rvt
#pragma HLS RESOURCE variable = l_rvt core = RAM_2P_URAM
    xf::blas::mem2stream<RTM_interface>(l_sizeVt, p_v2dt2, l_v2dt2);
    wide2stream<RTM_vtTypeWidth, RTM_multi << 1>(l_sizeVt, l_v2dt2, l_vt);
    duplicate(l_num, l_vt, l_pvt[0], l_rvt[0]);

    // Direct Wave
    hls::stream<RTM_interface> l_pin;
#pragma HLS stream variable = l_pin depth = 4
    hls::stream<RTM_interface> l_pout;
#pragma HLS stream variable = l_pout depth = 4
    hls::stream<RTM_pairType> l_p[RTM_numBSMs + 1];
#pragma HLS ARRAY_PARTITION variable = l_p dim = 1 complete
    hls::stream<RTM_upbType> l_upb[RTM_numBSMs];
#pragma HLS ARRAY_PARTITION variable = l_upb dim = 1 complete

    RTM_TYPE::loadUpb<RTM_numBSMs>(l_upbSize, p_t, l_upb, p_upb);

    xf::blas::mem2stream<RTM_interface>(l_sizeP, p_p0, l_pin);
    wide2stream<RTM_pairTypeWidth, RTM_multi>(l_sizeP, l_pin, l_p[0]);
    for (int i = 0; i < RTM_numBSMs; i++) {
#pragma HLS UNROLL
        sf[i].backwardF(l_upb[i], l_pvt[i], l_pvt[i + 1], l_p[i], l_p[i + 1], l_cp[i],
                        (l_t + RTM_numBSMs - i) == p_T || p_T - 1 == (l_t + RTM_numBSMs - i));
    }
    xf::hpc::dataConsumer(l_num, l_pvt[RTM_numBSMs]);
    stream2wide<RTM_pairTypeWidth, RTM_multi>(l_sizeP, l_p[RTM_numBSMs], l_pout);
    xf::blas::stream2mem<RTM_interface>(l_sizeP, l_pout, p_p1);

    // Receiver Wave

    hls::stream<RTM_pairType> l_r[RTM_numBSMs + 1];
#pragma HLS ARRAY_PARTITION variable = l_r dim = 1 complete
    hls::stream<RTM_dataType> l_rec[RTM_numBSMs];
#pragma HLS ARRAY_PARTITION variable = l_rec dim = 1 complete
    hls::stream<RTM_interface> l_rin;
#pragma HLS stream variable = l_rin depth = 4
    hls::stream<RTM_interface> l_rout;
#pragma HLS stream variable = l_rout depth = 4
    RTM_TYPE::loadUpb<RTM_numBSMs>(l_recSize, p_t, l_rec, p_rec);
    xf::blas::mem2stream<RTM_interface>(l_sizeP, p_r0, l_rin);
    wide2stream<RTM_pairTypeWidth, RTM_multi>(l_sizeP, l_rin, l_r[0]);
    for (int i = 0; i < RTM_numBSMs; i++) {
#pragma HLS UNROLL
        sr[i].backwardR(l_rec[i], l_rvt[i], l_rvt[i + 1], l_r[i], l_r[i + 1], l_cr[i]);
    }

    xf::hpc::dataConsumer(l_num, l_rvt[RTM_numBSMs]);
    stream2wide<RTM_pairTypeWidth, RTM_multi>(l_sizeP, l_r[RTM_numBSMs], l_rout);
    xf::blas::stream2mem<RTM_interface>(l_sizeP, l_rout, p_r1);

    // Image reBuild
    hls::stream<RTM_interface> l_in;
#pragma HLS stream variable = l_in depth = 4
    hls::stream<RTM_interface> l_out;
#pragma HLS stream variable = l_out depth = 4
    hls::stream<RTM_vtType> l_i[RTM_numBSMs + 1];
#pragma HLS ARRAY_PARTITION variable = l_i dim = 1 complete
#pragma HLS stream depth = RTM_TYPE::t_FifoDepth variable = l_i
#pragma HLS RESOURCE variable = l_i core = RAM_2P_URAM
    xf::blas::mem2stream<RTM_interface>(l_sizeI, p_i0, l_in);
    wide2stream<RTM_vtTypeWidth, RTM_multi << 1>(l_sizeI, l_in, l_i[0]);
    for (int i = 0; i < RTM_numBSMs; i++) {
#pragma HLS UNROLL
        sr[i].crossCorrelation(l_cp[i], l_cr[i], l_i[i], l_i[i + 1]);
    }
    stream2wide<RTM_vtTypeWidth, RTM_multi << 1>(l_sizeI, l_i[RTM_numBSMs], l_out);
    xf::blas::stream2mem<RTM_interface>(l_sizeI, l_out, p_i1);
}

/**
 * @brief rfmbackward kernel function
 *
 * @param p_z is the number of grids along detecting depth
 * @param p_x is the number of grids along detecting width
 * @param p_t is the number of detecting time parititons
 *
 * @param p_recz is the z coordinates of all receivers
 * @param p_rec  is the receiver data wavefileds
 *
 * @param p_coefz is the laplacian z-direction coefficients
 * @param p_coefx is the laplacian x-direction coefficients
 * @param p_taperz is the absorbing factor along z
 * @param p_taperx is the absorbing factor along x
 *
 * @param p_v2dt2 is the velocity model v^2 * dt^2
 *
 * @param p_upb is the uppper bounday wavefiled
 *
 * @param p_p0 is the first input memory of source wavefield
 * @param p_p1 is the second input memory of source wavefield
 *
 * @param p_r0 is the first input memory of receiver wavefield
 * @param p_r1 is the second input memory of receiver wavefield
 *
 * @param p_i0 is the first input memory of cross-correlation images
 * @param p_i1 is the second input memory of cross-correlation images
 */

extern "C" void rtmbackward(const unsigned int p_z,
                            const unsigned int p_x,
                            const unsigned int p_t,
                            const unsigned int p_recz,
                            const RTM_dataType* p_rec,
                            const RTM_dataType* p_coefz,
                            const RTM_dataType* p_coefx,
                            const RTM_dataType* p_taperz,
                            const RTM_dataType* p_taperx,
                            const RTM_interface* p_v2dt2,
                            RTM_interface* p_p0,
                            RTM_interface* p_p1,
                            RTM_interface* p_r0,
                            RTM_interface* p_r1,
                            RTM_interface* p_i0,
                            RTM_interface* p_i1,
                            RTM_upbType* p_upb);
