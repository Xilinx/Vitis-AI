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

#ifndef XF_HPC_RTM_DATAMOVER_HPP
#define XF_HPC_RTM_DATAMOVER_HPP

/**
 * @file dataMover.hpp
 * @brief datamovers are defined here, including memory to stream,
 * stream to stream and stream to memory
 */
namespace xf {
namespace hpc {
namespace rtm {

/**
 * @brief memSelStream reads data alternatively from two memory addresses to a stream
 *
 * @tparam t_InterfaceType is the datatype in memory
 * @tparam t_DataType is the datatype in of the stream
 *
 * @param p_n is the number of data to be read
 * @param p_k is the selection
 * @param p_mem0 is the first memory port
 * @param p_mem1 is the second memory port
 * @param p_str is the output stream
 */
template <typename t_InterfaceType, typename t_DataType>
void memSelStream(unsigned int p_n,
                  unsigned int p_k,
                  t_InterfaceType* p_mem0,
                  t_InterfaceType* p_mem1,
                  hls::stream<t_DataType>& p_str) {
    switch (p_k) {
        case 0:
            for (int i = 0; i < p_n; i++) {
#pragma HLS PIPELINE
                t_DataType l_in = p_mem0[i];
                p_str.write(l_in);
            }
            break;
        case 1:
            for (int i = 0; i < p_n; i++) {
#pragma HLS PIPELINE
                t_DataType l_in = p_mem1[i];
                p_str.write(l_in);
            }
            break;
    }
}

/**
 * @brief streamSelMem reads write alternatively to two memory addresses from a stream
 *
 * @tparam t_InterfaceType is the datatype in memory
 * @tparam t_DataType is the datatype in of the stream
 *
 * @param p_n is the number of data to be read
 * @param p_k is the selection
 * @param p_mem0 is the first memory port
 * @param p_mem1 is the second memory port
 * @param p_str is the input stream
 */
template <typename t_InterfaceType, typename t_DataType>
void streamSelMem(unsigned int p_n,
                  unsigned int p_k,
                  t_InterfaceType* p_mem0,
                  t_InterfaceType* p_mem1,
                  hls::stream<t_DataType>& p_str) {
    switch (p_k) {
        case 0:
            for (int i = 0; i < p_n; i++) {
#pragma HLS PIPELINE
                p_mem0[i] = p_str.read();
            }
            break;
        case 1:
            for (int i = 0; i < p_n; i++) {
#pragma HLS PIPELINE
                p_mem1[i] = p_str.read();
            }
            break;
    }
}
}
}
}
#endif
