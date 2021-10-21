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

#ifndef _XF_GRAPH_DENSE_SIMILARITY_KERNEL_HPP_
#define _XF_GRAPH_DENSE_SIMILARITY_KERNEL_HPP_

#include "similarity/dense_similarity_int.hpp"
#include "similarity/sort_top_k.hpp"
#include "similarity/enums.hpp"

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#define W_DATA 32
#define CHANNEL_NUMBER 8
#define PU_NUMBER 4
#define RAM_SIZE 15
#define MAX_K 101
#define EXT_MEM_SZ 32 << 10

template <typename I, typename F>
inline F bitsToF(I in) {
    union {
        I __I;
        F __F;
    } __T;
    __T.__I = in;
    return __T.__F;
}

template <typename F, typename I>
inline I fToBits(F in) {
    union {
        I __I;
        F __F;
    } __T;
    __T.__F = in;
    return __T.__I;
}

template <typename T>
struct is_float {
    operator bool() { return false; }
};

template <>
struct is_float<float> {
    operator bool() { return true; }
};

extern "C" void denseSimilarityKernel(ap_int<32>* config,
                                      ap_int<32>* sourceWeight,

                                      ap_int<32 * CHANNEL_NUMBER>* dataIn00,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn01,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn02,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn03,

                                      ap_int<32 * CHANNEL_NUMBER>* dataIn10,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn11,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn12,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn13,

                                      ap_int<32 * CHANNEL_NUMBER>* dataIn20,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn21,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn22,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn23,

                                      ap_int<32 * CHANNEL_NUMBER>* dataIn30,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn31,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn32,
                                      ap_int<32 * CHANNEL_NUMBER>* dataIn33,

                                      ap_int<32>* resultID,
                                      float* similarity);
#endif
