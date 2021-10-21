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
 * @file pagerank.hpp
 * @brief  This files contains implementation of Strongly Connected Components
 */

#ifndef XF_SOLVER_PSQRT_H
#define XF_SOLVER_PSQRT_H

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#include "../../../L2/include/hw/MatrixDecomposition/potrf.hpp"
#include "hls_math.h"
#include <hls_stream.h>

namespace xf {
namespace solver {
namespace internalPSQRT {
template <typename DT>
union DTConvert {
    uint64_t fixed;
    DT fp;
};
}

#ifndef __SYNTHESIS__
template <typename T, int rowTemplate, int unrollNm>
void pseudosqrt(int nrows, T* matrix, T* outMat) {
#else
template <typename T, int rowTemplate, int unrollNm>
void pseudosqrt(int nrows, T matrix[rowTemplate * rowTemplate], T outMat[rowTemplate * rowTemplate]) {
#endif
    int info;
    xf::solver::potrf<T, rowTemplate, unrollNm>(nrows, matrix, nrows, info);
    for (int i = 0; i < nrows; ++i) {
#pragma HLS loop_tripcount min = rowTemplate max = rowTemplate
        for (int j = 0; j < nrows; ++j) {
#pragma HLS loop_tripcount min = rowTemplate max = rowTemplate
            if (j <= i) {
                outMat[i * nrows + j] = matrix[i * nrows + j];
            } else {
                outMat[i * nrows + j] = 0;
            }
        }
    }
}
template <typename T, int rowTemplate, int unrollNm, int TLen, int TO>
void pseudosqrtStrm(int nrows, hls::stream<ap_uint<TLen * TO> >& matrix, hls::stream<ap_uint<TLen * TO> >& outMat) {
    if (nrows == 1) {
        ap_uint<TLen* TO> tmp = matrix.read();
        internalPSQRT::DTConvert<T> tmp0;
        tmp0.fixed = tmp.range(TLen - 1, 0);
        tmp0.fp = hls::sqrt(tmp0.fp);
        tmp.range(TLen - 1, 0) = tmp0.fixed;
        outMat.write(tmp);
    } else {
        static T matA[unrollNm][(rowTemplate + unrollNm - 1) / unrollNm][rowTemplate];
#pragma HLS array_partition variable = matA cyclic factor = unrollNm
#pragma HLS resource variable = matA core = XPM_MEMORY uram

        int size0 = nrows;
        int size = (size0 + TO - 1) / TO;
        ap_uint<TLen* TO> tmp = 0;
    Loop_read:
        for (int c = 0; c < nrows; ++c) {
#pragma HLS loop_tripcount min = rowTemplate max = rowTemplate
            for (int i = 0; i < size; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = rowTemplate/TO max = rowTemplate/TO
                // clang-format on
                for (int j = 0; j < TO; ++j) {
#pragma HLS loop_tripcount min = TO max = TO
#pragma HLS pipeline II = 1
                    if (j == 0) {
                        tmp = matrix.read();
                    }
                    int index = i * TO + j;
                    if (index < size0) {
                        int r = index % unrollNm;
                        int l = index / unrollNm;
                        internalPSQRT::DTConvert<T> tmp0;
                        tmp0.fixed = tmp.range(TLen * (j + 1) - 1, TLen * j);
                        matA[r][l][c] = tmp0.fp;
                    }
                }
            }
        }

        xf::solver::internal::cholesky_core<T, rowTemplate, unrollNm>(nrows, matA);
        tmp = 0;
    Loop_write:
        for (int c = 0; c < nrows; ++c) {
#pragma HLS loop_tripcount min = rowTemplate max = rowTemplate
            for (int i = 0; i < size; ++i) {
// clang-format off
#pragma HLS loop_tripcount min = rowTemplate/TO max = rowTemplate/TO
                // clang-format on
                for (int j = 0; j < TO; ++j) {
#pragma HLS loop_tripcount min = TO max = TO
#pragma HLS pipeline II = 1
                    int index = i * TO + j;
                    if (index < size0) {
                        if (index <= c) {
                            internalPSQRT::DTConvert<T> tmp0;
                            tmp0.fp = matA[c % unrollNm][c / unrollNm][index];
#ifndef __SYNTHESIS__
#ifdef _DEBUG_SOLVER_
                            std::cout << "index = " << c * nrows + index << "\t val = " << tmp0.fp << std::endl;
#endif
#endif
                            tmp.range(TLen * (j + 1) - 1, TLen * j) = tmp0.fixed;
                        } else {
                            tmp.range(TLen * (j + 1) - 1, TLen * j) = 0;
#ifndef __SYNTHESIS__
#ifdef _DEBUG_SOLVER_
                            std::cout << "index = " << c * nrows + index << "\t val = " << 0 << std::endl;
#endif
#endif
                        }
                    }
                    if (j == TO - 1) {
                        outMat.write(tmp);
                    }
                }
            }
        }
    }
}

} // namespace solver
} // namespace xf
#endif //#ifndef XF_SOLVER_PSQRT_H
