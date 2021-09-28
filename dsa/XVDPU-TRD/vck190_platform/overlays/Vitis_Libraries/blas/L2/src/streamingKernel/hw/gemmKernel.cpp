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
#include "blasInstr.hpp"
#include "gemmStoreKernel.hpp"
#include "gemmLoadKernel.hpp"
#include "gemmTimerKernel.hpp"

#if GemmStreamSeq

#include "gemmMuls.hpp"
#include "gemmMulsSink.hpp"
#include "gemmAdds.hpp"
#include "gemmTags.hpp"
#include "gemmCPlusX.hpp"
#include "gemmSystolicArray.hpp"
#include "gemmMerge.hpp"

extern "C" void gemmKernel(ap_uint<BLAS_ddrMemBits>* p_rdPtr, ap_uint<BLAS_ddrMemBits>* p_wrPtr) {
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_aStr;
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_bStr;
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_xStr;
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_cStr;
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_aOut;
    hls::stream<ap_uint<2> > l_tagOut;
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_tagB;
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_bMulIn[BLAS_parEntries];
    hls::stream<BLAS_dataType> l_aSysOut[BLAS_parEntries];
    hls::stream<ap_uint<2> > l_tagSysOut[BLAS_parEntries];
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_mul[BLAS_parEntries];
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_add[BLAS_parEntries];
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_sumStr;
    hls::stream<ap_uint<32> > l_resStr;
    hls::stream<unsigned int> l_blocks[BLAS_parEntries + 1];
    hls::stream<ap_uint<16> > l_opCodeStr;
#pragma HLS DATAFLOW
    xf::blas::gemmLoad<BLAS_maxNumInstrs, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                       BLAS_nParWords, BLAS_ddrMemBits>(p_rdPtr, l_aStr, l_bStr, l_xStr, l_opCodeStr);

    xf::blas::gemmTags<BLAS_dataType, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                       BLAS_nParWords, BLAS_ddrMemBits>(l_aStr, l_bStr, l_aOut, l_tagOut, l_tagB, l_blocks);

    xf::blas::gemmSystolicArray<BLAS_dataType, BLAS_parEntries, BLAS_kParWords, BLAS_ddrMemBits>(
        l_aOut, l_tagOut, l_tagB, l_bMulIn[0], l_aSysOut, l_tagSysOut);
    for (int i = 0; i < BLAS_parEntries - 1; i++) {
#pragma HLS UNROLL
        xf::blas::gemmMuls<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(l_aSysOut[i], l_tagSysOut[i], l_bMulIn[i],
                                                                           l_bMulIn[i + 1], l_mul[i]);
    }
    xf::blas::gemmMulsSink<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(
        l_aSysOut[BLAS_parEntries - 1], l_tagSysOut[BLAS_parEntries - 1], l_bMulIn[BLAS_parEntries - 1],
        l_mul[BLAS_parEntries - 1]);
    for (int i = 0; i < BLAS_parEntries; i++) {
#pragma HLS UNROLL
        xf::blas::gemmAdds<BLAS_dataType, BLAS_parEntries, BLAS_kParWords>(l_mul[i], l_add[i], l_blocks[i]);
    }

    xf::blas::gemmMerge<BLAS_ddrMemBits, BLAS_parEntries>(l_add, l_sumStr, l_blocks[BLAS_parEntries]);

    xf::blas::gemmCPlusX<BLAS_dataType, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                         BLAS_nParWords, BLAS_ddrMemBits>(l_xStr, l_sumStr, l_cStr);
    xf::blas::gemmLdStTimer(l_opCodeStr, l_resStr);
    xf::blas::gemmStore<BLAS_resOffsetBytes, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                        BLAS_nParWords, BLAS_ddrMemBits>(l_cStr, l_resStr, p_wrPtr);
}

#else
#include "gemmStreamKernel.hpp"
extern "C" void gemmKernel(ap_uint<BLAS_ddrMemBits>* p_rdPtr, ap_uint<BLAS_ddrMemBits>* p_wrPtr) {
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_aStr;
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_bStr;
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_xStr;
    hls::stream<ap_uint<BLAS_ddrMemBits> > l_cStr;
    hls::stream<ap_uint<32> > l_resStr;
    hls::stream<ap_uint<16> > l_opCodeStr;
#pragma HLS DATAFLOW
    xf::blas::gemmLoad<BLAS_maxNumInstrs, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                       BLAS_nParWords, BLAS_ddrMemBits>(p_rdPtr, l_aStr, l_bStr, l_xStr, l_opCodeStr);
    xf::blas::gemmStream<BLAS_dataType, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                         BLAS_nParWords, BLAS_ddrMemBits>(l_aStr, l_bStr, l_xStr, l_cStr);
    xf::blas::gemmLdStTimer(l_opCodeStr, l_resStr);
    xf::blas::gemmStore<BLAS_resOffsetBytes, BLAS_memWordsPerInstr, BLAS_parEntries, BLAS_mParWords, BLAS_kParWords,
                        BLAS_nParWords, BLAS_ddrMemBits>(l_cStr, l_resStr, p_wrPtr);
}
#endif
