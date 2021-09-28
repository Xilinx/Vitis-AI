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
#ifndef _DSPLIB_WIDGET_REAL2COMPLEX_HPP_
#define _DSPLIB_WIDGET_REAL2COMPLEX_HPP_

/*
Widget API Cast Kernel.
This file exists to capture the definition of the widget api cast kernel class.
The class definition holds defensive checks on parameter range and other
legality.
The constructor definition is held in this class because this class must be
accessible to graph level aie compilation.
The main runtime filter function is captured elsewhere as it contains aie
intrinsics which are not included in aie graph level
compilation.
*/

/* Coding conventions
   TT_      template type suffix
   TP_      template parameter suffix
*/

/* Design Notes

*/

#include <adf.h>
#include "widget_real2complex_traits.hpp"
#include <vector>

//#define _DSPLIB_WIDGET_REAL2COMPLEX_HPP_DEBUG_
#ifndef INLINE_DECL
#define INLINE_DECL inline __attribute__((always_inline))
#endif

namespace xf {
namespace dsp {
namespace aie {
namespace widget {
namespace real2complex {

//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA, typename TT_OUT_DATA, unsigned int TP_WINDOW_VSIZE>
class kernelClass {
   private:
    // Parameter value defensive and legality checks

   public:
    // Constructor
    kernelClass() {}

    void kernelClassMain(const TT_DATA* __restrict inBuff, TT_OUT_DATA* __restrict outBuff);
};

template <unsigned int TP_WINDOW_VSIZE>
class kernelClass<cint16, int16, TP_WINDOW_VSIZE> {
   private:
    // Parameter value defensive and legality checks

   public:
    // Constructor
    kernelClass() {}

    void kernelClassMain(const cint16* __restrict inBuff, int16* __restrict outBuff);
};

template <unsigned int TP_WINDOW_VSIZE>
class kernelClass<cint32, int32, TP_WINDOW_VSIZE> {
   private:
    // Parameter value defensive and legality checks

   public:
    // Constructor
    kernelClass() {}

    void kernelClassMain(const cint32* __restrict inBuff, int32* __restrict outBuff);
};

template <unsigned int TP_WINDOW_VSIZE>
class kernelClass<cfloat, float, TP_WINDOW_VSIZE> {
   private:
    // Parameter value defensive and legality checks

   public:
    // Constructor
    kernelClass() {}

    void kernelClassMain(const cfloat* __restrict inBuff, float* __restrict outBuff);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel base specialization. Used for single window to single window copy
template <typename TT_DATA, typename TT_OUT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_real2complex : public kernelClass<TT_DATA, TT_OUT_DATA, TP_WINDOW_VSIZE> {
   public:
    // Constructor
    widget_real2complex() : kernelClass<TT_DATA, TT_OUT_DATA, TP_WINDOW_VSIZE>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_real2complex::convertData); }

    // Main function
    void convertData(input_window<TT_DATA>* __restrict inWindow, output_window<TT_OUT_DATA>* __restrict outWindow0);
};
}
}
}
}
}

#endif // _DSPLIB_WIDGET_REAL2COMPLEX_HPP_
