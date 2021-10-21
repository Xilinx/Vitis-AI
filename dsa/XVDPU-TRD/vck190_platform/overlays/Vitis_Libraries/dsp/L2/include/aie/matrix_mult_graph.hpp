/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DSPLIB_MATRIX_MULT_GRAPH_HPP_
#define _DSPLIB_MATRIX_MULT_GRAPH_HPP_

// This file holds the definition of the matrix mult graph class
/**
 * @file matrix_mult_graph.hpp
 *
 **/

#include <adf.h>
#include <vector>

#include "matrix_mult.hpp"
#include "matrix_mult_untiler.hpp"
#include "matrix_mult_tiler.hpp"
#include "matrix_mult_tile_widget.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace blas {
namespace matrix_mult {

using namespace adf;
//// Kernel creation, static coefficients
// template <int dim, typename TT_DATA_A, typename TT_DATA_B, unsigned int ...Args >
// class create_casc_kernel<2, TT_DATA_A, TT_DATA_B, Args... > {
//    public:
//    static void create(kernel (&matMultKernels)[TP_CASC_LEN]) {
//        matMultKernels[0] = kernel::create_object<matrix_mult<TT_DATA_A, TT_DATA_B, Args...>>();
//    }
//}

/**
  * @endcond
  */

/**
 * @brief matrix_mult performs a GEneral Matrix Multiply (GEMM), taking two input matrices of configurable dimensions
 *and data type.
 *
 * These are the templates to configure the Matrix Multiply graph class.
 * @tparam TT_DATA_A describes the type of individual data samples input of
 *         Matrix A to the gemm function. This is a typename and must be one
 *         of the following:
 *         int16, cint16, int32, cint32, float, cfloat.
 * @tparam TT_DATA_B describes the type of individual data samples input of
 *         Matrix B to the gemm function. This is a typename and must be one
 *         of the following:
 *         int16, cint16, int32, cint32, float, cfloat.
 *         The following rules apply:
 *         - must be an integer type if TT_DATA_A is an integer type
 *         - must be a float type if TT_DATA_A is a float type.
 * @tparam TP_DIM_A is an unsigned integer which describes the number of elements
 *          along the unique dimension (rows) of Matrix A.
 * @tparam TP_DIM_AB is an unsigned integer which describes the number of elements
 *          along the common dimension of Matrix A (columns) and Matrix B (rows).
 * @tparam TP_DIM_B is an unsigned integer which describes the number of elements
 *          along the unique dimension (columns) of Matrix B.
 * @tparam TP_SHIFT is describes power of 2 shift down applied to the accumulation of
 *         product terms before each output. TP_SHIFT must be in the range 0 to 61.
 * @tparam TP_RND describes the selection of rounding to be applied during the
 *         shift down stage of processing. TP_RND must be in the range 0 to 7
 *         where
 *         0 = floor (truncate) eg. 3.8 Would become 3.
 *         1 = ceiling e.g. 3.2 would become 4.
 *         2 = round to positive infinity.
 *         3 = round to negative infinity.
 *         4 = round symmetrical to infinity.
 *         5 = round symmetrical to zero.
 *         6 = round convergent to even.
 *         7 = round convergent to odd.
 *         Modes 2 to 7 round to the nearest integer. They differ only in how
 *         they round for values of 0.5.
 * @tparam TP_DIM_A_LEADING describes the scheme in which the data should be stored
 *         in memory. ROW_MAJOR = 0, COL_MAJOR = 1. Note, a COL_MAJOR matrix can be
 *         transposed to become a ROW_MAJOR matrix.
 * @tparam TP_DIM_B_LEADING describes the scheme in which the data should be stored
 *         in memory. ROW_MAJOR = 0, COL_MAJOR = 1.
 * @tparam TP_DIM_OUT_LEADING describes the scheme in which the data should be stored
 *         in memory. ROW_MAJOR = 0, COL_MAJOR = 1.
 * @tparam TP_ADD_TILING_A describes wether or not to add an additional kernel to
 *          rearrange the matrix samples into their required position. Setting this
 *          option to 0 indicates that the re-arrangement will be done externally to
 *          the AIE matrix multiply graph.
 * @tparam TP_ADD_TILING_B describes wether or not to add an additional kernel to
 *          rearrange the matrix samples into their required position. Setting this
 *          option to 0 indicates that the re-arrangement will be done externally to
 *          the AIE matrix multiply graph.
 * @tparam TP_ADD_DETILING_OUT describes wether or not to add an additional kernel to
 *          rearrange the matrix samples into their required position. Setting this
 *          option to 0 indicates that the re-arrangement will be done externally to
 *          the AIE matrix multiply graph.
 * @tparam TP_INPUT_WINDOW_VSIZE_A describes the number of samples in the window API
 *         used for input to Matrix A. It must be of size TP_DIM_A*TP_DIM_AB*N.
 *         Typical use has N=1, however N>1 can be utilised to minimise overhead of
 *         window API. This parameter is optional and has a default value of
 *         TP_DIM_A*TP_DIM_AB (N=1).
 * @tparam TP_INPUT_WINDOW_VSIZE_B describes the number of samples in the window API
 *         used for input to Matrix B. It must be of size TP_DIM_B*TP_DIM_AB*M.
 *         Typical use has M=1, however M>1 can be utilised to minimise overhead of
 *         window API.  This parameter is optional and has a default value of
 *         TP_DIM_B*TP_DIM_AB (M=1).
 *         Note, the output window will be of size:
 *           (TP_INPUT_WINDOW_VSIZE_A/TP_DIM_AB * TP_INPUT_WINDOW_VSIZE_B/TP_DIM_AB).
 *          When N and M is 1, output window size will be TP_DIM_A * TP_DIM_B.
 * @tparam TP_CASC_LEN describes the number of AIE Tiles to split the GEMM operation into.
 *         TP_CASC_LEN splits the operation over TP_DIM_AB, where each kernel
 *         utilises the cascade stream to pass partial accumulation results to
 *         the next kernel. In effect, dot(A,B) + C.
 *         Note, it is also possible to tile the operation over multiple AIE tiles
 *         by instantiating multiple GEMM graphs with smaller dimensions.
 *
**/

template <typename TT_DATA_A,
          typename TT_DATA_B,
          unsigned int TP_DIM_A,
          unsigned int TP_DIM_AB,
          unsigned int TP_DIM_B,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_DIM_A_LEADING = ROW_MAJOR,
          unsigned int TP_DIM_B_LEADING = COL_MAJOR,
          unsigned int TP_DIM_OUT_LEADING = ROW_MAJOR,
          unsigned int TP_ADD_TILING_A = 1,
          unsigned int TP_ADD_TILING_B = 1,
          unsigned int TP_ADD_DETILING_OUT = 1,
          unsigned int TP_INPUT_WINDOW_VSIZE_A = TP_DIM_A* TP_DIM_AB,
          unsigned int TP_INPUT_WINDOW_VSIZE_B = TP_DIM_B* TP_DIM_AB,
          unsigned int TP_CASC_LEN = 1>
class matrix_mult_graph : public graph {
   public:
    /**
     * The input data to the function. This input is two windows of samples of
     * TT_DATA_A and TT_DATA_B type. The number of samples in the window is
     * described by TP_INPUT_WINDOW_VSIZE_A and TP_INPUT_WINDOW_VSIZE_B, which are
     * derived from TP_DIM_A, TP_DIM_AB and TP_DIM_B.
     **/
    port<input> inA[TP_CASC_LEN];
    port<input> inB[TP_CASC_LEN];
    /**
     * A window API of
     * TP_INPUT_WINDOW_VSIZE_A/TP_DIM_AB * TP_INPUT_WINDOW_VSIZE_B/TP_DIM_AB samples,
     * or simply TP_DIM_A * TP_DIM_B samples of a derived output type.
     **/
    port<output> out;
    /**
      * @cond NOCOMMENTS
      */
    kernel m_MatmultKernels[TP_CASC_LEN];
    kernel untiler;
    kernel tilerA[TP_CASC_LEN];
    kernel tilerB[TP_CASC_LEN];
    // Access function
    /**
      * @endcond
      */

    /**
     * Access function to get pointer to kernel (or first kernel in a chained configuration).
     **/

    kernel* getKernels() { return m_MatmultKernels; };

    // Empty type for a fallback to avoid redundant instantiations from the compiler in x86
    struct no_kernel {};

    template <bool cascIn, bool cascOut>
    using matMultCasc = matrix_mult<TT_DATA_A,
                                    TT_DATA_B,
                                    TP_DIM_A,
                                    (TP_DIM_AB / TP_CASC_LEN),
                                    TP_DIM_B,
                                    TP_SHIFT,
                                    TP_RND,
                                    TP_DIM_A_LEADING,
                                    TP_DIM_B_LEADING,
                                    TP_DIM_OUT_LEADING,
                                    (TP_INPUT_WINDOW_VSIZE_A / TP_CASC_LEN),
                                    (TP_INPUT_WINDOW_VSIZE_B / TP_CASC_LEN),
                                    cascIn,
                                    cascOut>;
    // avoid redundant instances of unsued templates -- Fixes x86sim linker error without resorting to recursive
    // template metaprogramming
    using onlyMatMult = typename std::conditional<(TP_CASC_LEN == 1), matMultCasc<false, false>, no_kernel>::type;
    using firstMatMult = typename std::conditional<(TP_CASC_LEN > 1), matMultCasc<false, true>, onlyMatMult>::type;
    using lastMatMult = typename std::conditional<(TP_CASC_LEN > 1), matMultCasc<true, false>, firstMatMult>::type;
    using middleMatMult = typename std::conditional<(TP_CASC_LEN > 2), matMultCasc<true, true>, lastMatMult>::type;

    // Todo structured binding instead (C++17)
    // AIE_API tiling scheme in use - single configuration for each data type.
    // Tiling scheme doesn't change vs cascade or not.
    static constexpr typename middleMatMult::tilingStruct tilingScheme = middleMatMult::getTilingScheme();

    // Forward compatible for batch window processing.
    static constexpr unsigned int dimAPerKernel = (TP_INPUT_WINDOW_VSIZE_A / TP_DIM_AB);
    static constexpr unsigned int dimBPerKernel = (TP_INPUT_WINDOW_VSIZE_B / TP_DIM_AB);

    using TilerClassA = tilerKernelClass<tilingScheme.Atile,
                                         tilingScheme.ABtile,
                                         dimAPerKernel,
                                         (TP_DIM_AB / TP_CASC_LEN),
                                         TP_DIM_A_LEADING,
                                         TT_DATA_A>;
    using TilerClassB = tilerKernelClass<tilingScheme.ABtile,
                                         tilingScheme.Btile,
                                         (TP_DIM_AB / TP_CASC_LEN),
                                         dimBPerKernel,
                                         TP_DIM_B_LEADING,
                                         TT_DATA_B>;
    using DetilerClassOut = untilerKernelClass<tilingScheme.Atile,
                                               tilingScheme.Btile,
                                               dimAPerKernel,
                                               dimBPerKernel,
                                               TP_DIM_OUT_LEADING,
                                               outType_t<TT_DATA_A, TT_DATA_B> >;

    static constexpr bool isRedundantTilerA =
        (((TP_DIM_AB / TP_CASC_LEN) <= tilingScheme.ABtile) && (TP_DIM_A_LEADING == ROW_MAJOR));
    static constexpr bool isRedundantTilerB =
        ((dimBPerKernel <= tilingScheme.Btile) && (TP_DIM_B_LEADING == ROW_MAJOR));
    static constexpr bool isRedundantTilerOut =
        ((dimBPerKernel <= tilingScheme.Btile) && (TP_DIM_OUT_LEADING == ROW_MAJOR));

    using TileAConditional = ConditionalWidget<isRedundantTilerA ? 0 : TP_ADD_TILING_A,
                                               (TP_INPUT_WINDOW_VSIZE_A / TP_CASC_LEN) * sizeof(TT_DATA_A),
                                               TilerClassA>;
    using TileBConditional = ConditionalWidget<isRedundantTilerB ? 0 : TP_ADD_TILING_B,
                                               (TP_INPUT_WINDOW_VSIZE_B / TP_CASC_LEN) * sizeof(TT_DATA_B),
                                               TilerClassB>;
    using DetileOutConditional =
        ConditionalWidget<isRedundantTilerOut ? 0 : TP_ADD_DETILING_OUT,
                          dimAPerKernel * dimBPerKernel * sizeof(outType_t<TT_DATA_A, TT_DATA_B>),
                          DetilerClassOut>;
    /**
     * @brief This is the constructor function for the Matric Multiply graph.
     **/
    matrix_mult_graph() {
        if (isRedundantTilerA && TP_ADD_TILING_A == 1) {
            printf(
                "WARNING: TP_ADD_TILING_A is true, but P_DIM_AB is small enough that tiling is not nessecary for this "
                "configuration. TP_ADD_TILING_A will be ignored. \n");
        }
        if (isRedundantTilerB && TP_ADD_TILING_B == 1) {
            printf(
                "WARNING: TP_ADD_TILING_B is true, but P_DIM_B is small enough that tiling is not nessecary for this "
                "configuration. TP_ADD_TILING_B will be ignored. \n");
        }
        if (isRedundantTilerOut && TP_ADD_DETILING_OUT == 1) {
            printf(
                "WARNING: TP_ADD_DETILING_OUT is true, but P_DIM_B is small enough that detiling is not nessecary for "
                "this configuration. TP_ADD_DETILING_OUT will be ignored. \n");
        }
        printf("\n");

        // make input connections
        for (int i = 0; i < TP_CASC_LEN; i++) {
            if (i >= 1 && i < (TP_CASC_LEN - 1)) {
                // both casccade
                m_MatmultKernels[i] = kernel::create_object<middleMatMult>();

            } else if (i >= 1 && i >= (TP_CASC_LEN - 1)) {
                // last kernel
                m_MatmultKernels[i] = kernel::create_object<lastMatMult>();
            } else {
                // first kernel
                m_MatmultKernels[i] = kernel::create_object<firstMatMult>();
            }
            if (i >= 1) {
                connect<cascade>(m_MatmultKernels[i - 1].out[0], m_MatmultKernels[i].in[2]);
            }
            // TODO, different window sizes for end kernel if the window size doesn't evenly split by CASC_LEN.

            tilerA[i] = TileAConditional::create(inA[i], m_MatmultKernels[i].in[0]);
            tilerB[i] = TileBConditional::create(inB[i], m_MatmultKernels[i].in[1]);
            // Specify mapping constraints - Can be overriden in parent graph.
            runtime<ratio>(m_MatmultKernels[i]) = 0.8;
            runtime<ratio>(tilerA[i]) = 0.4;
            runtime<ratio>(tilerB[i]) = 0.4;
            // Source files
            source(m_MatmultKernels[i]) = "matrix_mult.cpp";
            source(tilerA[i]) = "matrix_mult_tiler.cpp";
            source(tilerB[i]) = "matrix_mult_tiler.cpp";
        }

        untiler = DetileOutConditional::create(m_MatmultKernels[(TP_CASC_LEN - 1)].out[0], out);
        runtime<ratio>(untiler) = 0.4;
        source(untiler) = "matrix_mult_untiler.cpp";
    }
};
/**
  * @cond NOCOMMENTS
  */

// Specialized template for Cascade length of 1
template <typename TT_DATA_A,
          typename TT_DATA_B,
          unsigned int TP_DIM_A,
          unsigned int TP_DIM_AB,
          unsigned int TP_DIM_B,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_DIM_A_LEADING,
          unsigned int TP_DIM_B_LEADING,
          unsigned int TP_DIM_OUT_LEADING,
          unsigned int TP_ADD_TILING_A,
          unsigned int TP_ADD_TILING_B,
          unsigned int TP_ADD_DETILING_OUT,
          unsigned int TP_INPUT_WINDOW_VSIZE_A,
          unsigned int TP_INPUT_WINDOW_VSIZE_B>
class matrix_mult_graph<TT_DATA_A,
                        TT_DATA_B,
                        TP_DIM_A,
                        TP_DIM_AB,
                        TP_DIM_B,
                        TP_SHIFT,
                        TP_RND,
                        TP_DIM_A_LEADING,
                        TP_DIM_B_LEADING,
                        TP_DIM_OUT_LEADING,
                        TP_ADD_TILING_A,
                        TP_ADD_TILING_B,
                        TP_ADD_DETILING_OUT,
                        TP_INPUT_WINDOW_VSIZE_A,
                        TP_INPUT_WINDOW_VSIZE_B,
                        1> : public graph {
   public:
    port<input> inA[1];
    port<input> inB[1];
    port<output> out;

    kernel m_matMult;
    kernel untiler;
    kernel tilerA;
    kernel tilerB;
    // Access function for AIE synthesizer. Note, m_matMult is a public member anyway.
    kernel* getKernels() { return &m_matMult; };
    // individual matrix mult kernels don't need to know casc_len or add_tiling.
    using MatMult = matrix_mult<TT_DATA_A,
                                TT_DATA_B,
                                TP_DIM_A,
                                TP_DIM_AB,
                                TP_DIM_B,
                                TP_SHIFT,
                                TP_RND,
                                TP_DIM_A_LEADING,
                                TP_DIM_B_LEADING,
                                TP_DIM_OUT_LEADING,
                                TP_INPUT_WINDOW_VSIZE_A,
                                TP_INPUT_WINDOW_VSIZE_B>;

    // Todo structured binding instead (C++17)
    // AIE_API tiling scheme in use - single configuration for each data type.
    static constexpr typename MatMult::tilingStruct tilingScheme = MatMult::getTilingScheme();

    using TilerClassA = tilerKernelClass<tilingScheme.Atile,
                                         tilingScheme.ABtile,
                                         TP_INPUT_WINDOW_VSIZE_A / TP_DIM_AB,
                                         TP_DIM_AB,
                                         TP_DIM_A_LEADING,
                                         TT_DATA_A>;
    using TilerClassB = tilerKernelClass<tilingScheme.ABtile,
                                         tilingScheme.Btile,
                                         TP_DIM_AB,
                                         TP_INPUT_WINDOW_VSIZE_B / TP_DIM_AB,
                                         TP_DIM_B_LEADING,
                                         TT_DATA_B>;
    using DetilerClassOut = untilerKernelClass<tilingScheme.Atile,
                                               tilingScheme.Btile,
                                               TP_INPUT_WINDOW_VSIZE_A / TP_DIM_AB,
                                               TP_INPUT_WINDOW_VSIZE_B / TP_DIM_AB,
                                               TP_DIM_OUT_LEADING,
                                               outType_t<TT_DATA_A, TT_DATA_B> >;

    static constexpr bool isRedundantTilerA = ((TP_DIM_AB <= tilingScheme.ABtile) && (TP_DIM_A_LEADING == ROW_MAJOR));
    static constexpr bool isRedundantTilerB =
        ((TP_INPUT_WINDOW_VSIZE_B / TP_DIM_AB <= tilingScheme.Btile) && (TP_DIM_B_LEADING == ROW_MAJOR));
    static constexpr bool isRedundantTilerOut =
        ((TP_INPUT_WINDOW_VSIZE_B / TP_DIM_AB <= tilingScheme.Btile) && (TP_DIM_OUT_LEADING == ROW_MAJOR));

    using TileAConditional = ConditionalWidget<isRedundantTilerA ? 0 : TP_ADD_TILING_A,
                                               TP_INPUT_WINDOW_VSIZE_A * sizeof(TT_DATA_A),
                                               TilerClassA>;
    using TileBConditional = ConditionalWidget<isRedundantTilerB ? 0 : TP_ADD_TILING_B,
                                               TP_INPUT_WINDOW_VSIZE_B * sizeof(TT_DATA_B),
                                               TilerClassB>;
    using DetileOutConditional = ConditionalWidget<isRedundantTilerOut ? 0 : TP_ADD_DETILING_OUT,
                                                   TP_INPUT_WINDOW_VSIZE_A / TP_DIM_AB * TP_INPUT_WINDOW_VSIZE_B /
                                                       TP_DIM_AB * sizeof(outType_t<TT_DATA_A, TT_DATA_B>),
                                                   DetilerClassOut>;
    // constructor
    matrix_mult_graph() {
        if (isRedundantTilerA && TP_ADD_TILING_A == 1) {
            printf(
                "WARNING: TP_ADD_TILING_A is true, but P_DIM_AB is small enough that tiling is not nessecary for this "
                "configuration. TP_ADD_TILING_A will be ignored. \n");
        }
        if (isRedundantTilerB && TP_ADD_TILING_B == 1) {
            printf(
                "WARNING: TP_ADD_TILING_B is true, but P_DIM_B is small enough that tiling is not nessecary for this "
                "configuration. TP_ADD_TILING_B will be ignored. \n");
        }
        if (isRedundantTilerOut && TP_ADD_DETILING_OUT == 1) {
            printf(
                "WARNING: TP_ADD_DETILING_OUT is true, but P_DIM_B is small enough that detiling is not nessecary for "
                "this configuration. TP_ADD_DETILING_OUT will be ignored. \n");
        }
        printf("\n");

        m_matMult = kernel::create_object<MatMult>();

        tilerA = TileAConditional::create(inA[0], m_matMult.in[0]);
        tilerB = TileBConditional::create(inB[0], m_matMult.in[1]);
        untiler = DetileOutConditional::create(m_matMult.out[0], out);

        // Specify mapping constraints - overriden in parent graph
        runtime<ratio>(m_matMult) = 0.8;
        runtime<ratio>(tilerA) = 0.4;
        runtime<ratio>(tilerB) = 0.4;
        runtime<ratio>(untiler) = 0.4;
        // Source files
        source(tilerA) = "matrix_mult_tiler.cpp";
        source(tilerB) = "matrix_mult_tiler.cpp";
        source(untiler) = "matrix_mult_untiler.cpp";
        source(m_matMult) = "matrix_mult.cpp";
    }
};

/**
  * @endcond
  */
}
}
}
}
}

#endif //_DSPLIB_MATRIX_MULT_GRAPH_HPP_
