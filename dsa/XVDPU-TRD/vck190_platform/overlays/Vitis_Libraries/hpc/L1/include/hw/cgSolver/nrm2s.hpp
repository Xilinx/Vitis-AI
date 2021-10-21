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
#ifndef XF_HPC_CG_NRM2S_HPP
#define XF_HPC_CG_NRM2S_HPP
namespace xf {
namespace hpc {
namespace cg {

template <typename t_DataType, unsigned int t_ParEntries, typename t_IndexType = unsigned int>
void square(unsigned int p_n,
            hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_x,
            hls::stream<typename xf::blas::WideType<t_DataType, t_ParEntries>::t_TypeInt>& p_res) {
#ifndef __SYNTHESIS__
    assert(p_n % t_ParEntries == 0);
#endif
    t_IndexType l_numParEntries = p_n / t_ParEntries;
    for (t_IndexType i = 0; i < l_numParEntries; ++i) {
#pragma HLS PIPELINE
        xf::blas::WideType<t_DataType, t_ParEntries> l_valX;
        xf::blas::WideType<t_DataType, t_ParEntries> l_valRes;
        l_valX = p_x.read();
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            l_valRes[j] = l_valX[j] * l_valX[j];
        }
        p_res.write(l_valRes);
    }
}
template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
void nrm2s(unsigned int p_n,
           hls::stream<typename xf::blas::WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
           t_DataType& p_res) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
#pragma HLS DATAFLOW
    hls::stream<typename xf::blas::WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt> l_mulStr;
    square<t_DataType, 1 << t_LogParEntries, t_IndexType>(p_n, p_x, l_mulStr);
    xf::blas::sum<t_DataType, t_LogParEntries, t_IndexType>(p_n, l_mulStr, p_res);
}
template <typename t_DataType, unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
void nrm2s(unsigned int p_n,
           hls::stream<typename xf::blas::WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt>& p_x,
           hls::stream<typename xf::blas::WideType<t_DataType, 1>::t_TypeInt>& p_res) {
#ifndef __SYNTHESIS__
    assert(p_n % (1 << t_LogParEntries) == 0);
#endif
#pragma HLS DATAFLOW
    hls::stream<typename xf::blas::WideType<t_DataType, 1 << t_LogParEntries>::t_TypeInt> l_mulStr;
    square<t_DataType, 1 << t_LogParEntries, t_IndexType>(p_n, p_x, l_mulStr);
    xf::blas::sum<t_DataType, t_LogParEntries, t_IndexType>(p_n, l_mulStr, p_res, 1);
}
}
}
}
#endif
