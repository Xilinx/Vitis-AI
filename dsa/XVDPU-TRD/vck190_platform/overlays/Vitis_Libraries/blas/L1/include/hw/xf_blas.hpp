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
 * @file xf_blas.hpp
 * @brif Top-level header for Vitis BLAS Library.
 *
 */

#ifndef XF_BLAS_HPP
#define XF_BLAS_HPP

// shared modules
#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas/helpers.hpp"

// BLAS L1 function modules

#include "xf_blas/amax.hpp"
#include "xf_blas/amin.hpp"
#include "xf_blas/asum.hpp"
#include "xf_blas/axpy.hpp"
#include "xf_blas/copy.hpp"
#include "xf_blas/dot.hpp"
#include "xf_blas/scal.hpp"
#include "xf_blas/swap.hpp"
#include "xf_blas/nrm2.hpp"

// BLAS L2 function modules

#include "xf_blas/gemv.hpp"
#include "xf_blas/gbmv.hpp"
#include "xf_blas/symv.hpp"
#include "xf_blas/trmv.hpp"
/* TODO
 *
 */

// BLAS L3 function modules

#include "xf_blas/gemm.hpp"

/* TODO
 *
 *
*/

#endif
