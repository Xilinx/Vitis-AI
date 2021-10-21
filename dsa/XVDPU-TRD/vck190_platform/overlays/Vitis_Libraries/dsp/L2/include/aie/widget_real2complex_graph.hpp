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
#ifndef _DSPLIB_WIDGET_REAL2COMPLEX_GRAPH_HPP_
#define _DSPLIB_WIDGET_REAL2COMPLEX_GRAPH_HPP_
/*
The file captures the definition of the 'L2' graph level class for
the Real to Complex Widget library element.
*/
/**
 * @file widget_real2complex_graph.hpp
 *
 **/

#include <adf.h>
#include <vector>
#include <tuple>

#include "widget_real2complex.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace widget {
namespace real2complex {
using namespace adf;

//--------------------------------------------------------------------------------------------------
// widget_real2complex_graph template
//--------------------------------------------------------------------------------------------------
/**
 * @brief widget_real2complex is utility to convert real data to complex or vice versa
 *
 * These are the templates to configure the function.
 * @tparam TT_DATA describes the type of individual data samples input to the function.
 *         This is a typename and must be one of the following:
 *         int16, cint16, int32, cint32, float, cfloat.
 * @tparam TT_OUT_DATA describes the type of individual data samples output from the function.
 *         This is a typename and must be one of the following:
 *         int16, cint16, int32, cint32, float, cfloat.
 *         TT_OUT_DATA must also be the real or complex counterpart of TT_DATA, e.g.
 *         TT_DATA = int16 and TT_OUT_DATA = cint16 is valid,
 *         TT_DATA = cint16 and TT_OUT_DATA = int16 is valid, but
 *         TT_DATA = int16 and TT_OUT_DATA = cint32 is not valid.
 * @tparam TP_WINDOW_VSIZE describes the number of samples in the window API
 *         used if either input or output is a window.
 *         Note: Margin size should not be included in TP_INPUT_WINDOW_VSIZE.
 **/
template <typename TT_DATA, typename TT_OUT_DATA, unsigned int TP_WINDOW_VSIZE>
/**
 * This is the class for the Widget API Cast graph
 **/
class widget_real2complex_graph : public graph {
   public:
    static_assert((std::is_same<TT_DATA, cint16>::value && std::is_same<TT_OUT_DATA, int16>::value) ||
                      (std::is_same<TT_DATA, cint32>::value && std::is_same<TT_OUT_DATA, int32>::value) ||
                      (std::is_same<TT_DATA, cfloat>::value && std::is_same<TT_OUT_DATA, float>::value) ||
                      (std::is_same<TT_DATA, int16>::value && std::is_same<TT_OUT_DATA, cint16>::value) ||
                      (std::is_same<TT_DATA, int32>::value && std::is_same<TT_OUT_DATA, cint32>::value) ||
                      (std::is_same<TT_DATA, float>::value && std::is_same<TT_OUT_DATA, cfloat>::value),
                  "ERROR: TT_DATA and TT_OUT_DATA are not a real/complex pair");

    /**
     * The input data to the function. Window API is expected.
     * Data is read from here, converted and written to output.
     **/
    port<input> in;
    /**
     * An API of TT_DATA type.
     **/
    port<output> out;
    /**
      * @cond NOCOMMENTS
      */
    kernel m_kernel;

    // Access function for AIE synthesizer
    /**
      * @endcond
      */

    /**
     * Access function to get pointer to kernel (or first kernel in a chained configuration).
     **/

    kernel* getKernels() { return &m_kernel; };

    /**
     * @brief This is the constructor function for the Widget API Cast graph.
     **/
    widget_real2complex_graph() {
        m_kernel = kernel::create_object<widget_real2complex<TT_DATA, TT_OUT_DATA, TP_WINDOW_VSIZE> >();
        // Specify mapping constraints
        runtime<ratio>(m_kernel) = 0.1; // Nominal figure. The real figure requires knowledge of the sample rate.
        // Source files
        source(m_kernel) = "widget_real2complex.cpp";

        // make connections
        connect<window<TP_WINDOW_VSIZE * sizeof(TT_DATA)> >(in, m_kernel.in[0]);
        connect<window<TP_WINDOW_VSIZE * sizeof(TT_OUT_DATA)> >(m_kernel.out[0], out);
    }; // constructor
};
}
}
}
}
} // namespace braces

#endif //_DSPLIB_WIDGET_REAL2COMPLEX_GRAPH_HPP_
