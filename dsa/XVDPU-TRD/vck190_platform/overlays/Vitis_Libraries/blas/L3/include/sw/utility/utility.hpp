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

#ifndef XF_BLAS_UTILITY_HPP
#define XF_BLAS_UTILITY_HPP

using namespace std;

namespace xf {

namespace blas {

typedef enum {
    XFBLAS_STATUS_SUCCESS,         // 0
    XFBLAS_STATUS_NOT_INITIALIZED, // 1
    XFBLAS_STATUS_INVALID_VALUE,   // 2
    XFBLAS_STATUS_ALLOC_FAILED,    // 3
    XFBLAS_STATUS_NOT_SUPPORTED,   // 4
    XFBLAS_STATUS_NOT_PADDED,      // 5
    XFBLAS_STATUS_MEM_ALLOCATED,   // 6
    XFBLAS_STATUS_INVALID_OP,      // 7
    XFBLAS_STATUS_INVALID_FILE,    // 8
    XFBLAS_STATUS_INVALID_PROGRAM  // 9
} xfblasStatus_t;

typedef enum { XFBLAS_ENGINE_GEMM, XFBLAS_ENGINE_GEMV, XFBLAS_ENGINE_FCN } xfblasEngine_t;

typedef enum { XFBLAS_OP_N, XFBLAS_OP_T, XFBLAS_OP_C } xfblasOperation_t;

} // namespace blas

} // namespace xf

#endif
