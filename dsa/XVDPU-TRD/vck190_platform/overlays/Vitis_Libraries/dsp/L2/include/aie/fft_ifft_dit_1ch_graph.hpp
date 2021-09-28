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
#ifndef _DSPLIB_FFT_IFFT_DIT_1CH_GRAPH_HPP_
#define _DSPLIB_FFT_IFFT_DIT_1CH_GRAPH_HPP_

#include <adf.h>
#include <vector>
#include "fft_ifft_dit_1ch.hpp"
#define CEIL(x, y) (((x + y - 1) / y) * y)

using namespace adf;

#ifndef __X86SIM__
#include "fft_bufs.h" //Extern declarations of all twiddle factor stores and rank-temporary sample stores.
#endif                //_X86SIM__

namespace xf {
namespace dsp {
namespace aie {
namespace fft {
namespace dit_1ch {
/**
 * @cond NOCOMMENTS
 */

//---------start of recursive kernel creation code.
// Recursive kernel creation, static coefficients
template <int dim,
          typename TT_DATA,
          typename TT_INT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_CASC_LEN,
          unsigned int TP_END_RANK,
          unsigned int TP_RANKS_PER_KERNEL,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE = TP_POINT_SIZE>
class create_casc_kernel_recur {
   public:
    static void create(kernel (&fftKernels)[TP_CASC_LEN]) {
        static constexpr int kRawStartRank = (int)TP_END_RANK - (int)TP_RANKS_PER_KERNEL;
        static constexpr unsigned int TP_START_RANK = kRawStartRank < 0 ? 0 : kRawStartRank;

        fftKernels[dim - 1] = kernel::create_object<
            fft_ifft_dit_1ch<TT_INT_DATA, TT_INT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT, TP_START_RANK,
                             TP_END_RANK, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE> >();
        create_casc_kernel_recur<dim - 1, TT_DATA, TT_INT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT,
                                 TP_CASC_LEN, TP_START_RANK, TP_RANKS_PER_KERNEL, TP_DYN_PT_SIZE,
                                 TP_WINDOW_VSIZE>::create(fftKernels);
    }
};
// Recursive kernel creation, static coefficients
template <typename TT_DATA,
          typename TT_INT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_CASC_LEN,
          unsigned int TP_END_RANK,
          unsigned int TP_RANKS_PER_KERNEL,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
class create_casc_kernel_recur<1,
                               TT_DATA,
                               TT_INT_DATA,
                               TT_TWIDDLE,
                               TP_POINT_SIZE,
                               TP_FFT_NIFFT,
                               TP_SHIFT,
                               TP_CASC_LEN,
                               TP_END_RANK,
                               TP_RANKS_PER_KERNEL,
                               TP_DYN_PT_SIZE,
                               TP_WINDOW_VSIZE> {
   public:
    static void create(kernel (&fftKernels)[TP_CASC_LEN]) {
        static constexpr unsigned int TP_START_RANK = 0;
        fftKernels[0] = kernel::create_object<
            fft_ifft_dit_1ch<TT_DATA, TT_INT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT, TP_START_RANK,
                             TP_END_RANK, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE> >();
    }
};
// Kernel creation, entry to recursion, also end of cascade. For integer types
template <int dim,
          typename TT_DATA, // type for I/O
          typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE = TP_POINT_SIZE>
class create_casc_kernel {
   public:
    typedef typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;
    static void create(kernel (&fftKernels)[TP_CASC_LEN]) {
        static constexpr unsigned int TP_END_RANK =
            (TP_POINT_SIZE == 4096)
                ? 12
                : (TP_POINT_SIZE == 2048)
                      ? 11
                      : (TP_POINT_SIZE == 1024)
                            ? 10
                            : (TP_POINT_SIZE == 512)
                                  ? 9
                                  : (TP_POINT_SIZE == 256)
                                        ? 8
                                        : (TP_POINT_SIZE == 128)
                                              ? 7
                                              : (TP_POINT_SIZE == 64)
                                                    ? 6
                                                    : (TP_POINT_SIZE == 32)
                                                          ? 5
                                                          : (TP_POINT_SIZE == 16) ? 4
                                                                                  : 0; // 0 is an error trap effectively
        static_assert(fnCheckCascLen<TT_DATA, TP_END_RANK, TP_CASC_LEN>(), "Error: TP_CASC_LEN is invalid");
        static_assert(
            fnCheckCascLen2<TT_DATA, TP_POINT_SIZE, TP_CASC_LEN>(),
            "Error: 16 point float FFT does not support cascade"); // due to need for ping/pang/pong complication
        static constexpr int kRawRanksPerKernel =
            std::is_same<TT_DATA, cfloat>::value ? (TP_END_RANK / TP_CASC_LEN) : (TP_END_RANK / TP_CASC_LEN / 2) * 2;
        static constexpr unsigned int TP_RANKS_PER_KERNEL = std::is_same<TT_DATA, cfloat>::value
                                                                ? kRawRanksPerKernel
                                                                : (kRawRanksPerKernel < 2 ? 2 : kRawRanksPerKernel);
        static constexpr unsigned int TP_START_RANK = (int)TP_END_RANK - (int)TP_RANKS_PER_KERNEL;
        fftKernels[dim - 1] = kernel::create_object<
            fft_ifft_dit_1ch<T_internalDataType, TT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT,
                             TP_START_RANK, TP_END_RANK, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE> >();
        create_casc_kernel_recur<dim - 1, TT_DATA, T_internalDataType, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT,
                                 TP_SHIFT, TP_CASC_LEN, TP_START_RANK, TP_RANKS_PER_KERNEL, TP_DYN_PT_SIZE,
                                 TP_WINDOW_VSIZE>::create(fftKernels);
    }
};
// Kernel creation, entry to recursion, also end of cascade. For integer types - for single kernel
template <typename TT_DATA, // type for I/O
          typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
class create_casc_kernel<1,
                         TT_DATA,
                         TT_TWIDDLE,
                         TP_POINT_SIZE,
                         TP_FFT_NIFFT,
                         TP_SHIFT,
                         1,
                         TP_DYN_PT_SIZE,
                         TP_WINDOW_VSIZE> {
   public:
    static void create(kernel (&fftKernels)[1]) {
        static constexpr unsigned int TP_END_RANK =
            (TP_POINT_SIZE == 4096)
                ? 12
                : (TP_POINT_SIZE == 2048)
                      ? 11
                      : (TP_POINT_SIZE == 1024)
                            ? 10
                            : (TP_POINT_SIZE == 512)
                                  ? 9
                                  : (TP_POINT_SIZE == 256)
                                        ? 8
                                        : (TP_POINT_SIZE == 128)
                                              ? 7
                                              : (TP_POINT_SIZE == 64)
                                                    ? 6
                                                    : (TP_POINT_SIZE == 32)
                                                          ? 5
                                                          : (TP_POINT_SIZE == 16) ? 4
                                                                                  : 0; // 0 is an error trap effectively
        static constexpr unsigned int TP_START_RANK = 0;
        fftKernels[0] =
            kernel::create_object<fft_ifft_dit_1ch<TT_DATA, TT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT,
                                                   TP_START_RANK, TP_END_RANK, TP_DYN_PT_SIZE, TP_WINDOW_VSIZE> >();
    }
};
//---------End of recursive code.

/**
  * @endcond
  */

//--------------------------------------------------------------------------------------------------
// fft_dit_1ch template
//--------------------------------------------------------------------------------------------------
/**
 * @brief fft_dit_1ch is a single-channel, decimation-in-time, fixed point size FFT
 *
 * These are the templates to configure the single-channel decimation-in-time class.
 * @tparam TT_DATA describes the type of individual data samples input to and
 *         output from the transform function. This is a typename and must be one
 *         of the following:
 *         int16, cint16, int32, cint32, float, cfloat.
 * @tparam TT_TWIDDLE describes the type of twiddle factors of the transform.
 *         It must be one of the following: cint16, cint32, cfloat
 *         and must also satisfy the following rules:
 *         - 32 bit types are only supported when TT_DATA is also a 32 bit type,
 *         - TT_TWIDDLE must be an integer type if TT_DATA is an integer type
 *         - TT_TWIDDLE must be cfloat type if TT_DATA is a float type.
 * @tparam TP_POINT_SIZE is an unsigned integer which describes the number of point
 *         size of the transform. This must be 2^N where N is an integer in the range
 *         4 to 16 inclusive. When TP_DYN_PT_SIZE is set, TP_POINT_SIZE describes the maximum
 *         point size possible.
 * @tparam TP_FFT_NIFFT selects whether the transform to perform is an FFT (1) or IFFT (0).
 * @tparam TP_SHIFT selects the power of 2 to scale the result by prior to output.
 * @tparam TP_CASC_LEN selects the number of kernels the FFT will be divided over in series
 *         to improve throughput
 * @tparam TP_DYN_PT_SIZE selects whether (1) or not (0) to use run-time point size determination.
 *         When set, each frame of data must be preceeded, in the window, by a 256 bit header.
 *         The output frame will also be preceeded by a 256 bit vector which is a copy of the input
 *         vector, but for the top byte, which is 0 to indicate a legal frame or 1 to indicate an illegal
 *         frame.
 *         The lowest significance byte of the input header field describes forward (non-zero) or
 *         inverse(0) direction.
 *         The second least significant byte  8 bits of this field describe the Radix 2 power of the following
 *         frame. e.g. for a 512 point size, this field would hold 9, as 2^9 = 512. Any value below 4 or
 *         greater than log2(TP_POINT_SIZE) is considered illegal. When this occurs the top byte of the
 *         output header will be set to 1 and the output samples will be set to 0 for a frame of TP_POINT_SIZE
 * @tparam TP_WINDOW_VSIZE is an unsigned intered which describes the number of samples in the input window.
 *         By default, TP_WINDOW_SIZE is set ot match TP_POINT_SIZE.
 *         TP_WINDOW_SIZE may be set to be an integer multiple of the TP_POINT_SIZE, in which case
 *         multiple FFT iterations will be performed on a given input window, resulting in multiple
 *         iterations of output samples, reducing the numer of times the kernel needs to be triggered to
 *         process a given number of input data samples.
 *         As a result, the overheads inferred during kernel triggering are reduced and overall performance
 *         is increased.
  **/
template <typename TT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_CASC_LEN = 1,
          unsigned int TP_DYN_PT_SIZE = 0, // backwards compatible default
          unsigned int TP_WINDOW_VSIZE = TP_POINT_SIZE>
/**
 * This is the base class for the Single channel DIT FFT graph - one or many cascaded kernels for higher throughput
 **/
class fft_ifft_dit_1ch_base_graph : public graph {
   public:
    /**
     * The input data to the function. This input is a window API of
     * samples of TT_DATA type. The number of samples in the window is
     * described by TP_POINT_SIZE.
     **/
    port<input> in;
    /**
     * A window API of TP_POINT_SIZE samples of TT_DATA type.
     **/
    port<output> out;

    // declare FFT Kernel array
    kernel m_fftKernels[TP_CASC_LEN];
    kernel* getKernels() { return m_fftKernels; };

#ifndef __X86SIM__
    parameter fft_buf1;

    // twiddle table
    parameter fft_lut1, fft_lut1a, fft_lut1b; // large twiddle tables
    parameter fft_lut2, fft_lut2a, fft_lut2b;
    parameter fft_lut3, fft_lut3a, fft_lut3b;
    parameter fft_lut4, fft_lut4a, fft_lut4b;

    // interrank store
    parameter fft_buf4096;
    parameter fft_buf2048;
    parameter fft_buf1024;
    parameter fft_buf512;
    parameter fft_buf256;
    parameter fft_buf128;
#endif //__X86SIM__
       /**
        * @brief This is the constructor function for the Single channel DIT FFT graph.
        **/
    // Constructor
    fft_ifft_dit_1ch_base_graph() {
        typedef
            typename std::conditional<std::is_same<TT_DATA, cint16>::value, cint32_t, TT_DATA>::type T_internalDataType;

        // Create kernel class(s)
        create_casc_kernel<TP_CASC_LEN, TT_DATA, TT_TWIDDLE, TP_POINT_SIZE, TP_FFT_NIFFT, TP_SHIFT, TP_CASC_LEN,
                           TP_DYN_PT_SIZE, TP_WINDOW_VSIZE>::create(m_fftKernels);

        // Make data connections
        // Size of window is in Bytes.
        connect<window<TP_WINDOW_VSIZE * sizeof(TT_DATA) + TP_DYN_PT_SIZE * 32> >(in, m_fftKernels[0].in[0]);
        for (int k = 0; k < TP_CASC_LEN - 1; ++k) {
            connect<window<TP_WINDOW_VSIZE * sizeof(T_internalDataType) + TP_DYN_PT_SIZE * 32> >(
                m_fftKernels[k].out[0], m_fftKernels[k + 1].in[0]);
        }
        connect<window<TP_WINDOW_VSIZE * sizeof(TT_DATA) + TP_DYN_PT_SIZE * 32> >(m_fftKernels[TP_CASC_LEN - 1].out[0],
                                                                                  out);

#ifndef __X86SIM__
        // TODO - at present all twiddles connect to all kernels in a cascade, but this could be optimized since each
        // twiddle table need only go to one kernel
        // Connect twiddle Lookups
        // Note that this switch statement does NOT use break statements. That is deliberate. The top case executes all
        // cases.
        if (std::is_same<TT_DATA, cfloat>::value) {
            if (TP_POINT_SIZE == 4096) {
                fft_lut1 = parameter::array(fft_lut_tw2048_cfloat);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut1, m_fftKernels[k]);
                }
            }
            if (TP_POINT_SIZE >= 2048) {
                fft_lut2 = parameter::array(fft_lut_tw1024_cfloat);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut2, m_fftKernels[k]);
                }
            }
            if (TP_POINT_SIZE >= 1024) {
                fft_lut3 = parameter::array(fft_lut_tw512_cfloat);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut3, m_fftKernels[k]);
                }
            }
            if (TP_POINT_SIZE >= 512) {
                fft_lut4 = parameter::array(fft_lut_tw256_cfloat);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut4, m_fftKernels[k]);
                }
            }
        } else {
            if (TP_POINT_SIZE == 4096) {
                fft_lut1 = parameter::array(fft_lut_tw2048_half);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut1, m_fftKernels[k]);
                }
                fft_lut2 = parameter::array(fft_lut_tw1024);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut2, m_fftKernels[k]);
                }
                fft_lut3 = parameter::array(fft_lut_tw512);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut3, m_fftKernels[k]);
                }
                fft_lut4 = parameter::array(fft_lut_tw256);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut4, m_fftKernels[k]);
                }
            }
            if (TP_POINT_SIZE == 2048) {
                fft_lut2 = parameter::array(fft_lut_tw1024_half);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut2, m_fftKernels[k]);
                }
                fft_lut3 = parameter::array(fft_lut_tw512);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut3, m_fftKernels[k]);
                }
                fft_lut4 = parameter::array(fft_lut_tw256);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut4, m_fftKernels[k]);
                }
            }
            if (TP_POINT_SIZE == 1024) {
                fft_lut3 = parameter::array(fft_lut_tw512_half);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut3, m_fftKernels[k]);
                }
                fft_lut4 = parameter::array(fft_lut_tw256);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut4, m_fftKernels[k]);
                }
            }
            if (TP_POINT_SIZE == 512) {
                fft_lut4 = parameter::array(fft_lut_tw256_half);
                for (int k = 0; k < TP_CASC_LEN; ++k) {
                    connect<>(fft_lut4, m_fftKernels[k]);
                }
            }
        }

        // Connect inter-rank temporary storage
        fft_buf4096 = parameter::array(fft_4096_tmp1);
        fft_buf2048 = parameter::array(fft_2048_tmp1);
        fft_buf1024 = parameter::array(fft_1024_tmp1);
        fft_buf512 = parameter::array(fft_512_tmp1);
        fft_buf256 = parameter::array(fft_256_tmp1);
        fft_buf128 = parameter::array(fft_128_tmp1);
        switch (TP_POINT_SIZE) {
            case 4096:
                fft_buf1 = fft_buf4096;
                break;
            case 2048:
                fft_buf1 = fft_buf2048;
                break;
            case 1024:
                fft_buf1 = fft_buf1024;
                break;
            case 512:
                fft_buf1 = fft_buf512;
                break;
            case 256:
                fft_buf1 = fft_buf256;
                break;
            default:
                fft_buf1 = fft_buf128;
                break;
        }
        for (int k = 0; k < TP_CASC_LEN; ++k) {
            connect<>(fft_buf1, m_fftKernels[k]);
        }
#endif //__X86SIM__
        for (int k = 0; k < TP_CASC_LEN; ++k) {
            // Specify mapping constraints
            runtime<ratio>(m_fftKernels[k]) = 0.94;

            // Source files
            source(m_fftKernels[k]) = "fft_ifft_dit_1ch.cpp";
            headers(m_fftKernels[k]) = {"fft_ifft_dit_1ch.hpp"};
        }
    };
};
/**
 * @cond NOCOMMENTS
 */

//------------------------------------------------
// inheritance - used because only cint16 requires two internal buffers for sample storage
template <typename TT_DATA,
          typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_CASC_LEN = 1,
          unsigned int TP_DYN_PT_SIZE = 0, // backwards compatible default
          unsigned int TP_WINDOW_VSIZE = TP_POINT_SIZE>
class fft_ifft_dit_1ch_graph : public fft_ifft_dit_1ch_base_graph<TT_DATA,
                                                                  TT_TWIDDLE,
                                                                  TP_POINT_SIZE,
                                                                  TP_FFT_NIFFT,
                                                                  TP_SHIFT,
                                                                  TP_CASC_LEN,
                                                                  TP_DYN_PT_SIZE,
                                                                  TP_WINDOW_VSIZE> {
    // This is the default, used by cint32 and cfloat, which requires no second internal pingpong buffer.
};

#ifndef __X86SIM__
// Default inheritance for cint16, PT_SIZE 128 or below- to include second temporary buffer.
template <typename TT_TWIDDLE,
          unsigned int TP_POINT_SIZE,
          unsigned int TP_FFT_NIFFT,
          unsigned int TP_SHIFT,
          unsigned int TP_CASC_LEN,
          unsigned int TP_DYN_PT_SIZE,
          unsigned int TP_WINDOW_VSIZE>
class fft_ifft_dit_1ch_graph<cint16,
                             TT_TWIDDLE,
                             TP_POINT_SIZE,
                             TP_FFT_NIFFT,
                             TP_SHIFT,
                             TP_CASC_LEN,
                             TP_DYN_PT_SIZE,
                             TP_WINDOW_VSIZE> : public fft_ifft_dit_1ch_base_graph<cint16,
                                                                                   TT_TWIDDLE,
                                                                                   TP_POINT_SIZE,
                                                                                   TP_FFT_NIFFT,
                                                                                   TP_SHIFT,
                                                                                   TP_CASC_LEN,
                                                                                   TP_DYN_PT_SIZE,
                                                                                   TP_WINDOW_VSIZE> {
   public:
    parameter fft_buf4096;
    parameter fft_buf2048;
    parameter fft_buf1024;
    parameter fft_buf512;
    parameter fft_buf256;
    parameter fft_buf128;
    parameter fft_buf2;
    fft_ifft_dit_1ch_graph() {
        fft_buf4096 = parameter::array(fft_4096_tmp2);
        fft_buf2048 = parameter::array(fft_2048_tmp2);
        fft_buf1024 = parameter::array(fft_1024_tmp2);
        fft_buf512 = parameter::array(fft_512_tmp2);
        fft_buf256 = parameter::array(fft_256_tmp2);
        fft_buf128 = parameter::array(fft_128_tmp2);
        switch (TP_POINT_SIZE) {
            case 4096:
                fft_buf2 = fft_buf4096;
                break;
            case 2048:
                fft_buf2 = fft_buf2048;
                break;
            case 1024:
                fft_buf2 = fft_buf1024;
                break;
            case 512:
                fft_buf2 = fft_buf512;
                break;
            case 256:
                fft_buf2 = fft_buf256;
                break;
            default:
                fft_buf2 = fft_buf128;
                break;
        }
        for (int k = 0; k < TP_CASC_LEN; k++) {
            connect<>(fft_buf2, this->m_fftKernels[k]);
        }
    }
};
#endif //__X86SIM__

/**
  * @endcond
  */
}
}
}
}
}

#endif // _DSPLIB_FFT_IFFT_DIT_1CH_GRAPH_HPP_
