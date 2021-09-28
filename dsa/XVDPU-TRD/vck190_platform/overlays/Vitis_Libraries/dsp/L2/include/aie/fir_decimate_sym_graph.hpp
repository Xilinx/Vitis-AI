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
#ifndef _DSPLIB_FIR_DECIMATE_SYM_GRAPH_HPP_
#define _DSPLIB_FIR_DECIMATE_SYM_GRAPH_HPP_

/**
 * @file fir_decimate_sym_graph.hpp
 **/

// This file captures the definition of the 'L2' graph level class for the Symmetrical Decimation FIR library element.

#include <adf.h>
#include <vector>
#include "fir_decimate_sym.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace fir {
namespace decimate_sym {
using namespace adf;

//---------------------------------------------------------------------------------------------------
// create_casc_kernel_recur
// Where the FIR function is split over multiple processors to increase throughput, recursion
// is used to generate the multiple kernels, rather than a for loop due to constraints of
// c++ template handling.
// For each such kernel, only a splice of the full array of coefficients is processed.
//---------------------------------------------------------------------------------------------------
/**
  * @cond NOCOMMENTS
  */
template <int dim,
          typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP = 0,
          unsigned int TP_USE_COEFF_RELOAD = 0>
class create_casc_kernel_recur {
   private:
    static constexpr unsigned int kDualIpEn =
        0; // cascaded kernels do not support dual inputs, unless input interface is a stream
   public:
    static void create(kernel (&firKernels)[TP_CASC_LEN], const std::vector<TT_COEFF>& taps) {
        firKernels[dim - 1] = kernel::create_object<
            fir_decimate_sym<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND, TP_INPUT_WINDOW_VSIZE,
                             true, true, fnFirRangeSym<TP_FIR_LEN, TP_CASC_LEN, dim - 1>(), dim - 1, TP_CASC_LEN,
                             kDualIpEn, USE_COEFF_RELOAD_FALSE, 1> >(taps);
        create_casc_kernel_recur<dim - 1, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                                 TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, TP_DUAL_IP,
                                 USE_COEFF_RELOAD_FALSE>::create(firKernels, taps);
    }
};

// Middle kernel(s) in cascade, reloadable coefficients
template <int dim,
          typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class create_casc_kernel_recur<dim,
                               TT_DATA,
                               TT_COEFF,
                               TP_FIR_LEN,
                               TP_DECIMATE_FACTOR,
                               TP_SHIFT,
                               TP_RND,
                               TP_INPUT_WINDOW_VSIZE,
                               TP_CASC_LEN,
                               TP_DUAL_IP,
                               USE_COEFF_RELOAD_TRUE> {
   private:
    static constexpr unsigned int kDualIpEn =
        0; // cascaded kernels do not support dual inputs, unless input interface is a stream
   public:
    static void create(kernel (&firKernels)[TP_CASC_LEN]) {
        firKernels[dim - 1] = kernel::create_object<
            fir_decimate_sym<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND, TP_INPUT_WINDOW_VSIZE,
                             true, true, fnFirRangeSym<TP_FIR_LEN, TP_CASC_LEN, dim - 1>(), dim - 1, TP_CASC_LEN,
                             kDualIpEn, USE_COEFF_RELOAD_TRUE, 1> >();
        create_casc_kernel_recur<dim - 1, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                                 TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, TP_DUAL_IP,
                                 USE_COEFF_RELOAD_TRUE>::create(firKernels);
    }
};

// First kernel in cascade, last of recursion, static coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class create_casc_kernel_recur<1,
                               TT_DATA,
                               TT_COEFF,
                               TP_FIR_LEN,
                               TP_DECIMATE_FACTOR,
                               TP_SHIFT,
                               TP_RND,
                               TP_INPUT_WINDOW_VSIZE,
                               TP_CASC_LEN,
                               TP_DUAL_IP,
                               USE_COEFF_RELOAD_FALSE> {
   public:
    static void create(kernel (&firKernels)[TP_CASC_LEN], const std::vector<TT_COEFF>& taps) {
        firKernels[0] = kernel::create_object<fir_decimate_sym<
            TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND, TP_INPUT_WINDOW_VSIZE, false, true,
            fnFirRangeSym<TP_FIR_LEN, TP_CASC_LEN, 0>(), 0, TP_CASC_LEN, TP_DUAL_IP, USE_COEFF_RELOAD_FALSE, 1> >(taps);
    }
};

// First kernel in cascade, last of recursion, reloadable coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP>
class create_casc_kernel_recur<1,
                               TT_DATA,
                               TT_COEFF,
                               TP_FIR_LEN,
                               TP_DECIMATE_FACTOR,
                               TP_SHIFT,
                               TP_RND,
                               TP_INPUT_WINDOW_VSIZE,
                               TP_CASC_LEN,
                               TP_DUAL_IP,
                               USE_COEFF_RELOAD_TRUE> {
   public:
    static void create(kernel (&firKernels)[TP_CASC_LEN]) {
        firKernels[0] = kernel::create_object<fir_decimate_sym<
            TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND, TP_INPUT_WINDOW_VSIZE, false, true,
            fnFirRangeSym<TP_FIR_LEN, TP_CASC_LEN, 0>(), 0, TP_CASC_LEN, TP_DUAL_IP, USE_COEFF_RELOAD_TRUE, 1> >();
    }
};

// Last kernel in cascade, first of recursion, static coefficients
template <int dim,
          typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP = 0,
          unsigned int TP_USE_COEFF_RELOAD = 0,
          unsigned int TP_NUM_OUTPUTS = 1>
class create_casc_kernel {
   private:
    static constexpr unsigned int kDualIpEn =
        0; // cascaded kernels do not support dual inputs, unless input interface is a stream
   public:
    static void create(kernel (&firKernels)[TP_CASC_LEN], const std::vector<TT_COEFF>& taps) {
        firKernels[dim - 1] = kernel::create_object<
            fir_decimate_sym<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND, TP_INPUT_WINDOW_VSIZE,
                             true, false, fnFirRangeRemSym<TP_FIR_LEN, TP_CASC_LEN, dim - 1>(), dim - 1, TP_CASC_LEN,
                             kDualIpEn, USE_COEFF_RELOAD_FALSE, TP_NUM_OUTPUTS> >(taps);
        create_casc_kernel_recur<dim - 1, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                                 TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, TP_DUAL_IP,
                                 USE_COEFF_RELOAD_FALSE>::create(firKernels, taps);
    }
};

// Last kernel in cascade, first of recursion, reloadable coefficients
template <int dim,
          typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DUAL_IP,
          unsigned int TP_NUM_OUTPUTS>
class create_casc_kernel<dim,
                         TT_DATA,
                         TT_COEFF,
                         TP_FIR_LEN,
                         TP_DECIMATE_FACTOR,
                         TP_SHIFT,
                         TP_RND,
                         TP_INPUT_WINDOW_VSIZE,
                         TP_CASC_LEN,
                         TP_DUAL_IP,
                         USE_COEFF_RELOAD_TRUE,
                         TP_NUM_OUTPUTS> {
   private:
    static constexpr unsigned int kDualIpEn =
        0; // cascaded kernels do not support dual inputs, unless input interface is a stream
   public:
    static void create(kernel (&firKernels)[TP_CASC_LEN]) {
        firKernels[dim - 1] = kernel::create_object<
            fir_decimate_sym<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND, TP_INPUT_WINDOW_VSIZE,
                             true, false, fnFirRangeRemSym<TP_FIR_LEN, TP_CASC_LEN, dim - 1>(), dim - 1, TP_CASC_LEN,
                             kDualIpEn, USE_COEFF_RELOAD_TRUE, TP_NUM_OUTPUTS> >();
        create_casc_kernel_recur<dim - 1, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                                 TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, TP_DUAL_IP,
                                 USE_COEFF_RELOAD_TRUE>::create(firKernels);
    }
};

// Only kernel in cascade, static coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_DUAL_IP,
          unsigned int TP_NUM_OUTPUTS>
class create_casc_kernel<1,
                         TT_DATA,
                         TT_COEFF,
                         TP_FIR_LEN,
                         TP_DECIMATE_FACTOR,
                         TP_SHIFT,
                         TP_RND,
                         TP_INPUT_WINDOW_VSIZE,
                         1,
                         TP_DUAL_IP,
                         USE_COEFF_RELOAD_FALSE,
                         TP_NUM_OUTPUTS> {
   public:
    static constexpr unsigned int dim = 1;
    static constexpr unsigned int TP_CASC_LEN = 1;
    static void create(kernel (&firKernels)[TP_CASC_LEN], const std::vector<TT_COEFF>& taps) {
        firKernels[dim - 1] = kernel::create_object<
            fir_decimate_sym<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND, TP_INPUT_WINDOW_VSIZE,
                             false, false, fnFirRangeRemSym<TP_FIR_LEN, TP_CASC_LEN, dim - 1>(), dim - 1, TP_CASC_LEN,
                             TP_DUAL_IP, USE_COEFF_RELOAD_FALSE, TP_NUM_OUTPUTS> >(taps);
    }
};

// Only kernel in cascade, reloadable coefficients
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_DUAL_IP,
          unsigned int TP_NUM_OUTPUTS>
class create_casc_kernel<1,
                         TT_DATA,
                         TT_COEFF,
                         TP_FIR_LEN,
                         TP_DECIMATE_FACTOR,
                         TP_SHIFT,
                         TP_RND,
                         TP_INPUT_WINDOW_VSIZE,
                         1,
                         TP_DUAL_IP,
                         USE_COEFF_RELOAD_TRUE,
                         TP_NUM_OUTPUTS> {
   public:
    static constexpr unsigned int dim = 1;
    static constexpr unsigned int TP_CASC_LEN = 1;
    static void create(kernel (&firKernels)[TP_CASC_LEN]) {
        firKernels[dim - 1] = kernel::create_object<
            fir_decimate_sym<TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND, TP_INPUT_WINDOW_VSIZE,
                             false, false, fnFirRangeRemSym<TP_FIR_LEN, TP_CASC_LEN, dim - 1>(), dim - 1, TP_CASC_LEN,
                             TP_DUAL_IP, USE_COEFF_RELOAD_TRUE, TP_NUM_OUTPUTS> >();
    }
};
/**
  * @endcond
  */
/**
 * @brief fir_decimate_sym is a Symmetrical Decimation FIR filter
 *
 * These are the templates to configure the symmetrical decimator FIR class.
 * @tparam TT_DATA describes the type of individual data samples input to and
 *         output from the filter function. This is a typename and must be one
 *         of the following:
 *         int16, cint16, int32, cint32, float, cfloat.
 * @tparam TT_COEFF describes the type of individual coefficients of the filter
 *         taps. It must be one of the same set of types listed for TT_DATA
 *         and must also satisfy the following rules:
 *         - Complex types are only supported when TT_DATA is also complex.
 *         - 32 bit types are only supported when TT_DATA is also a 32 bit type,
 *         - TT_COEFF must be an integer type if TT_DATA is an integer type
 *         - TT_COEFF must be a float type if TT_DATA is a float type.
 * @tparam TP_FIR_LEN is an unsigned integer which describes the number of taps
 *         in the filter. TP_FIR_LEN must be in the range 4 to 240 and
 *         must be an integer multiple of the TP_DECIMATE_FACTOR value.
 * @tparam TP_DECIMATE_FACTOR is an unsigned integer which describes the
 *         decimation factor of the filter, the ratio of input to output samples.
 *         TP_DECIMATE_FACTOR must be in the range 2 to 3. For larger factors, use
 *         the fir_decimate_asym library element.
 * @tparam TP_SHIFT is describes power of 2 shift down applied to the accumulation of
 *         FIR terms before output. TP_SHIFT must be in the range 0 to 61.
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
 * @tparam TP_INPUT_WINDOW_VSIZE describes the number of samples in the window API
 *         used for input to the filter function.
 *         The number of values in the output window will be TP_INPUT_WINDOW_VSIZE
 *         divided by TP_DECIMATE_FACTOR by virtue the decimation factor.
 *         TP_INPUT_WINDOW_VSIZE must be an integer multiple of TP_DECIMATE_FACTOR
 *         The resulting output window size must be a multiple of 256bits.
 * @tparam TP_CASC_LEN describes the number of AIE processors to split the operation
 *         over. This allows resource to be traded for higher performance.
 *         TP_CASC_LEN must be in the range 1 (default) to 9.
 * @tparam TP_DUAL_IP is an implementation trade-off between performance and data
 *         bank resource. When set to 0, the FIR performance may be limited
 *         by load contention. When set to 1, two ram banks are used for input.
 * @tparam TP_USE_COEFF_RELOAD allows the user to select if runtime coefficient
 *         reloading should be used. This currently is only available for single
 *         kernel filters. When defining the parameter:
 *         - 0 = static coefficients, defined in filter constructor
 *         - 1 = reloadable coefficients, passed as argument to runtime function
 *
 *         Note: Margin size should not be included in TP_INPUT_WINDOW_VSIZE.
 **/
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN = 1,
          unsigned int TP_DUAL_IP = 0,
          unsigned int TP_USE_COEFF_RELOAD = 0, // 1 = use coeff reload, 0 = don't use coeff reload
          unsigned int TP_NUM_OUTPUTS = 1>
class fir_decimate_sym_graph : public graph {
   private:
// Parameter value defensive and legality checks
#include "fir_decimate_sym_check_params.hpp"

   public:
    /**
     * The input data to the function. This input is a window API of
     * samples of TT_DATA type. The number of samples in the window is
     * described by TP_INPUT_WINDOW_VSIZE.
     * Note: Margin is added internally to the graph, when connecting input port
     * with kernel port. Therefore, margin should not be added when connecting
     * graph to a higher level design unit.
     * Margin size (in Bytes) equals to TP_FIR_LEN rounded up to a nearest
     * multiple of 32 bytes.
     **/
    port<input> in;
    /**
     * A window API of TP_INPUT_WINDOW_VSIZE/TP_DECIMATE_FACTOR samples of TT_DATA type.
     **/
    port<output> out;
    /**
     * @cond NOCOMMENTS
     */
    kernel m_firKernels[TP_CASC_LEN];
    /**
      * @endcond
      */

    /**
     * Access function to get pointer to kernel (or first kernel in a chained configuration).
     **/

    kernel* getKernels() { return m_firKernels; };

    static_assert(TP_CASC_LEN < 9, "ERROR: Unsupported Cascade length");

    /**
     * @brief This is the constructor function for the symmetrical decimator FIR graph.
     * @param[in] taps - a pointer to the array of taps values of type TT_COEFF.
     *                   Due to symmetry, only the first half of the supplied taps
     *                   array (plus any odd centre tap) is used by this class.
     *                   The remaining coefficients are inferred by symmetry.
     **/
    fir_decimate_sym_graph(const std::vector<TT_COEFF>& taps) {
        // Create FIR class
        create_casc_kernel<TP_CASC_LEN, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                           TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, TP_DUAL_IP, TP_USE_COEFF_RELOAD,
                           TP_NUM_OUTPUTS>::create(m_firKernels, taps);

        // Make input connections
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernels[0].in[0]);
        for (int i = 1; i < TP_CASC_LEN; i++) {
            single_buffer(m_firKernels[i].in[0]);
            connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) +
                           fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(async(m_firKernels[i - 1].out[1]),
                                                                                   async(m_firKernels[i].in[0]));
        }
        // make cascade connections
        for (int i = 1; i < TP_CASC_LEN; i++) {
            connect<cascade>(m_firKernels[i - 1].out[0], m_firKernels[i].in[1]);
        }

        // make output connections
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[0], out);

        for (int i = 0; i < TP_CASC_LEN; i++) {
            // Specify mapping constraints
            runtime<ratio>(m_firKernels[i]) = 0.8;
            // Source files
            source(m_firKernels[i]) = "fir_decimate_sym.cpp";
        }
    };
}; // end of class

/**
 * @cond NOCOMMENTS
 */

// Specialized template for multiple kernels, single  input and static coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_sym_graph<TT_DATA,
                             TT_COEFF,
                             TP_FIR_LEN,
                             TP_DECIMATE_FACTOR,
                             TP_SHIFT,
                             TP_RND,
                             TP_INPUT_WINDOW_VSIZE,
                             TP_CASC_LEN,
                             DUAL_IP_SINGLE,
                             USE_COEFF_RELOAD_FALSE,
                             2> : public graph {
   public:
    port<input> in;
    port<output> out;
    /**
     * @endcond
     */
    /**
     * A second output window API of TP_INPUT_WINDOW_VSIZE/TP_DECIMATE_FACTOR samples of TT_DATA type.
     **/
    port<output> out2;
    /**
     * @cond NOCOMMENTS
     */
    kernel m_firKernels[TP_CASC_LEN];
    // Access function for AIE synthesizer
    kernel* getKernels() { return m_firKernels; };

    fir_decimate_sym_graph(const std::vector<TT_COEFF>& taps) {
        // Create FIR class
        create_casc_kernel<TP_CASC_LEN, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                           TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, DUAL_IP_SINGLE, USE_COEFF_RELOAD_FALSE,
                           2>::create(m_firKernels, taps);

        // Make input connections
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernels[0].in[0]);
        for (int i = 1; i < TP_CASC_LEN; i++) {
            single_buffer(m_firKernels[i].in[0]);
            connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) +
                           fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(async(m_firKernels[i - 1].out[1]),
                                                                                   async(m_firKernels[i].in[0]));
        }

        // make cascade connections
        for (int i = 1; i < TP_CASC_LEN; i++) {
            connect<cascade>(m_firKernels[i - 1].out[0], m_firKernels[i].in[1]);
        }

        // make output connections
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[0], out);
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[1], out2);

        for (int i = 0; i < TP_CASC_LEN; i++) {
            // Specify mapping constraints
            runtime<ratio>(m_firKernels[i]) = 0.8;
            // Source files
            source(m_firKernels[i]) = "fir_decimate_sym.cpp";
        }
    };
}; // end of class

// Specialized template for multiple kernels, dual input and static coefficients, single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_sym_graph<TT_DATA,
                             TT_COEFF,
                             TP_FIR_LEN,
                             TP_DECIMATE_FACTOR,
                             TP_SHIFT,
                             TP_RND,
                             TP_INPUT_WINDOW_VSIZE,
                             TP_CASC_LEN,
                             DUAL_IP_DUAL,
                             USE_COEFF_RELOAD_FALSE,
                             1> : public graph {
   public:
    port<input> in;
    port<input> in2;
    port<output> out;

    kernel m_firKernels[TP_CASC_LEN];
    // Access function for AIE synthesizer
    kernel* getKernels() { return m_firKernels; };

    // constructor
    fir_decimate_sym_graph(const std::vector<TT_COEFF>& taps) {
        create_casc_kernel<TP_CASC_LEN, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                           TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, DUAL_IP_DUAL, USE_COEFF_RELOAD_FALSE,
                           1>::create(m_firKernels, taps);

        // make input connections
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernels[0].in[0]);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in2, m_firKernels[0].in[1]);
        for (int i = 1; i < TP_CASC_LEN; i++) {
            single_buffer(m_firKernels[i].in[0]);
            connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) +
                           fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(async(m_firKernels[i - 1].out[1]),
                                                                                   async(m_firKernels[i].in[0]));
        }

        // make cascade connections
        for (int i = 1; i < TP_CASC_LEN; i++) {
            connect<cascade>(m_firKernels[i - 1].out[0], m_firKernels[i].in[1]);
        }

        // make output connectionsB
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[0], out);

        // Specify mapping constraints
        for (int i = 0; i < TP_CASC_LEN; i++) {
            // Specify mapping constraints
            runtime<ratio>(m_firKernels[i]) = 0.8; // ignore this figure. The real value is a function of parameters.
            // Source files
            source(m_firKernels[i]) = "fir_decimate_sym.cpp";
        }
    }
};

// Specialized template for multiple kernels, dual input and static coefficients, dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_sym_graph<TT_DATA,
                             TT_COEFF,
                             TP_FIR_LEN,
                             TP_DECIMATE_FACTOR,
                             TP_SHIFT,
                             TP_RND,
                             TP_INPUT_WINDOW_VSIZE,
                             TP_CASC_LEN,
                             DUAL_IP_DUAL,
                             USE_COEFF_RELOAD_FALSE,
                             2> : public graph {
   public:
    port<input> in;
    port<input> in2;
    port<output> out;
    port<output> out2;

    kernel m_firKernels[TP_CASC_LEN];
    // Access function for AIE synthesizer
    kernel* getKernels() { return m_firKernels; };

    // constructor
    fir_decimate_sym_graph(const std::vector<TT_COEFF>& taps) {
        create_casc_kernel<TP_CASC_LEN, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                           TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, DUAL_IP_DUAL, USE_COEFF_RELOAD_FALSE,
                           2>::create(m_firKernels, taps);

        // make input connections
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernels[0].in[0]);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in2, m_firKernels[0].in[1]);
        for (int i = 1; i < TP_CASC_LEN; i++) {
            single_buffer(m_firKernels[i].in[0]);
            connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) +
                           fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(async(m_firKernels[i - 1].out[1]),
                                                                                   async(m_firKernels[i].in[0]));
        }

        // make cascade connections
        for (int i = 1; i < TP_CASC_LEN; i++) {
            connect<cascade>(m_firKernels[i - 1].out[0], m_firKernels[i].in[1]);
        }

        // make output connectionsB
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[0], out);
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[1], out2);

        // Specify mapping constraints
        for (int i = 0; i < TP_CASC_LEN; i++) {
            // Specify mapping constraints
            runtime<ratio>(m_firKernels[i]) = 0.8; // ignore this figure. The real value is a function of parameters.
            // Source files
            source(m_firKernels[i]) = "fir_decimate_sym.cpp";
        }
    }
};

// Placeholder space for specializations:
// multiple kernels, single input with reloadable coefficients
// multiple kernels, dual input with reloadable coefficients.

// Specialized template for multi-kernel design with single input and reloadable coefficients single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_sym_graph<TT_DATA,
                             TT_COEFF,
                             TP_FIR_LEN,
                             TP_DECIMATE_FACTOR,
                             TP_SHIFT,
                             TP_RND,
                             TP_INPUT_WINDOW_VSIZE,
                             TP_CASC_LEN,
                             DUAL_IP_SINGLE,
                             USE_COEFF_RELOAD_TRUE,
                             1> : public graph {
   public:
    port<input> in;
    port<output> out;
    /**
     * @endcond
     */
    /**
     * A Run-time Parameter API containing the set of coefficient values. A change to these values will be detected and
     *will cause a
     * reload of the coefficients within the kernel or kernels to be used on the next data window.
     * This port is present only when TP_USE_COEFF_RELOAD is set to 1.
     **/
    port<input> coeff;
    /**
     * @cond NOCOMMENTS
     */

    kernel m_firKernels[TP_CASC_LEN];

    // Access function for AIE synthesizer
    kernel* getKernels() { return m_firKernels; };

    // constructor
    fir_decimate_sym_graph() {
        // create kernels
        create_casc_kernel<TP_CASC_LEN, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                           TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, DUAL_IP_SINGLE, USE_COEFF_RELOAD_TRUE,
                           1>::create(m_firKernels);

        // make input connections
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernels[0].in[0]);
        for (int i = 1; i < TP_CASC_LEN; i++) {
            single_buffer(m_firKernels[i].in[0]);
            connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) +
                           fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(async(m_firKernels[i - 1].out[1]),
                                                                                   async(m_firKernels[i].in[0]));
        }

        // make cascade connections
        for (int i = 1; i < TP_CASC_LEN; i++) {
            connect<cascade>(m_firKernels[i - 1].out[0], m_firKernels[i].in[1]);
        }

        // make RTP connection
        connect<parameter>(coeff, async(m_firKernels[0].in[1]));

        // make output connections
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[0], out);

        for (int i = 0; i < TP_CASC_LEN; i++) {
            // Specify mapping constraints
            runtime<ratio>(m_firKernels[i]) = 0.8;
            // Source files
            source(m_firKernels[i]) = "fir_decimate_sym.cpp";
        }
    }
};

// Specialized template for multi-kernel design with single input and reloadable coefficients dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_sym_graph<TT_DATA,
                             TT_COEFF,
                             TP_FIR_LEN,
                             TP_DECIMATE_FACTOR,
                             TP_SHIFT,
                             TP_RND,
                             TP_INPUT_WINDOW_VSIZE,
                             TP_CASC_LEN,
                             DUAL_IP_SINGLE,
                             USE_COEFF_RELOAD_TRUE,
                             2> : public graph {
   public:
    port<input> in;
    port<output> out;
    port<output> out2;
    port<input> coeff;

    kernel m_firKernels[TP_CASC_LEN];

    // Access function for AIE synthesizer
    kernel* getKernels() { return m_firKernels; };

    // constructor
    fir_decimate_sym_graph() {
        // create kernels
        create_casc_kernel<TP_CASC_LEN, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                           TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, DUAL_IP_SINGLE, USE_COEFF_RELOAD_TRUE,
                           2>::create(m_firKernels);

        // make input connections
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernels[0].in[0]);
        for (int i = 1; i < TP_CASC_LEN; i++) {
            single_buffer(m_firKernels[i].in[0]);
            connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) +
                           fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(async(m_firKernels[i - 1].out[1]),
                                                                                   async(m_firKernels[i].in[0]));
        }

        // make cascade connections
        for (int i = 1; i < TP_CASC_LEN; i++) {
            connect<cascade>(m_firKernels[i - 1].out[0], m_firKernels[i].in[1]);
        }

        // make RTP connection
        connect<parameter>(coeff, async(m_firKernels[0].in[1]));

        // make output connections
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[0], out);
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[1], out2);

        for (int i = 0; i < TP_CASC_LEN; i++) {
            // Specify mapping constraints
            runtime<ratio>(m_firKernels[i]) = 0.8;
            // Source files
            source(m_firKernels[i]) = "fir_decimate_sym.cpp";
        }
    }
};

// Specialized template for multi-kernel design with dual input and reloadable coefficients. single output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_sym_graph<TT_DATA,
                             TT_COEFF,
                             TP_FIR_LEN,
                             TP_DECIMATE_FACTOR,
                             TP_SHIFT,
                             TP_RND,
                             TP_INPUT_WINDOW_VSIZE,
                             TP_CASC_LEN,
                             DUAL_IP_DUAL,
                             USE_COEFF_RELOAD_TRUE,
                             1> : public graph {
   public:
    port<input> in;
    port<input> in2;
    port<output> out;
    port<input> coeff;

    kernel m_firKernels[TP_CASC_LEN];

    // Access function for AIE synthesizer
    kernel* getKernels() { return m_firKernels; };

    // constructor
    fir_decimate_sym_graph() {
        // create kernels
        create_casc_kernel<TP_CASC_LEN, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                           TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, DUAL_IP_DUAL, USE_COEFF_RELOAD_TRUE,
                           1>::create(m_firKernels);

        // make input connections
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernels[0].in[0]);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in2, m_firKernels[0].in[1]);
        for (int i = 1; i < TP_CASC_LEN; i++) {
            single_buffer(m_firKernels[i].in[0]);
            connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) +
                           fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(async(m_firKernels[i - 1].out[1]),
                                                                                   async(m_firKernels[i].in[0]));
        }

        // make cascade connections
        for (int i = 1; i < TP_CASC_LEN; i++) {
            connect<cascade>(m_firKernels[i - 1].out[0], m_firKernels[i].in[1]);
        }

        // make RTP connection
        connect<parameter>(coeff, async(m_firKernels[0].in[2]));

        // make output connections
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[0], out);

        for (int i = 0; i < TP_CASC_LEN; i++) {
            // Specify mapping constraints
            runtime<ratio>(m_firKernels[i]) = 0.8;
            // Source files
            source(m_firKernels[i]) = "fir_decimate_sym.cpp";
        }
    }
};

// Specialized template for multi-kernel design with dual input and reloadable coefficients. dual output
template <typename TT_DATA,
          typename TT_COEFF,
          unsigned int TP_FIR_LEN,
          unsigned int TP_DECIMATE_FACTOR,
          unsigned int TP_SHIFT,
          unsigned int TP_RND,
          unsigned int TP_INPUT_WINDOW_VSIZE,
          unsigned int TP_CASC_LEN>
class fir_decimate_sym_graph<TT_DATA,
                             TT_COEFF,
                             TP_FIR_LEN,
                             TP_DECIMATE_FACTOR,
                             TP_SHIFT,
                             TP_RND,
                             TP_INPUT_WINDOW_VSIZE,
                             TP_CASC_LEN,
                             DUAL_IP_DUAL,
                             USE_COEFF_RELOAD_TRUE,
                             2> : public graph {
   public:
    port<input> in;
    port<input> in2;
    port<output> out;
    port<output> out2;
    port<input> coeff;

    kernel m_firKernels[TP_CASC_LEN];

    // Access function for AIE synthesizer
    kernel* getKernels() { return m_firKernels; };

    // constructor
    fir_decimate_sym_graph() {
        // create kernels
        create_casc_kernel<TP_CASC_LEN, TT_DATA, TT_COEFF, TP_FIR_LEN, TP_DECIMATE_FACTOR, TP_SHIFT, TP_RND,
                           TP_INPUT_WINDOW_VSIZE, TP_CASC_LEN, DUAL_IP_DUAL, USE_COEFF_RELOAD_TRUE,
                           2>::create(m_firKernels);

        // make input connections
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in, m_firKernels[0].in[0]);
        connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA), fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(
            in2, m_firKernels[0].in[1]);
        for (int i = 1; i < TP_CASC_LEN; i++) {
            single_buffer(m_firKernels[i].in[0]);
            connect<window<TP_INPUT_WINDOW_VSIZE * sizeof(TT_DATA) +
                           fnFirMargin<TP_FIR_LEN, TT_DATA>() * sizeof(TT_DATA)> >(async(m_firKernels[i - 1].out[1]),
                                                                                   async(m_firKernels[i].in[0]));
        }

        // make cascade connections
        for (int i = 1; i < TP_CASC_LEN; i++) {
            connect<cascade>(m_firKernels[i - 1].out[0], m_firKernels[i].in[1]);
        }

        // make RTP connection
        connect<parameter>(coeff, async(m_firKernels[0].in[2]));

        // make output connections
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[0], out);
        connect<window<TP_INPUT_WINDOW_VSIZE / TP_DECIMATE_FACTOR * sizeof(TT_DATA)> >(
            m_firKernels[TP_CASC_LEN - 1].out[1], out2);

        for (int i = 0; i < TP_CASC_LEN; i++) {
            // Specify mapping constraints
            runtime<ratio>(m_firKernels[i]) = 0.8;
            // Source files
            source(m_firKernels[i]) = "fir_decimate_sym.cpp";
        }
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
#endif // _DSPLIB_FIR_DECIMATE_SYM_GRAPH_HPP_
