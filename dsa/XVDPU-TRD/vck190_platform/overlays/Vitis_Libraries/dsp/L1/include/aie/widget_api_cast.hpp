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
#ifndef _DSPLIB_WIDGET_API_CAST_HPP_
#define _DSPLIB_WIDGET_API_CAST_HPP_

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
#include "widget_api_cast_traits.hpp"
#include <vector>

//#define _DSPLIB_WIDGET_API_CAST_HPP_DEBUG_
#ifndef INLINE_DECL
#define INLINE_DECL inline __attribute__((always_inline))
#endif

namespace xf {
namespace dsp {
namespace aie {
namespace widget {
namespace api_cast {

//-----------------------------------------------------------------------------------------------------
template <typename TT_DATA,
          unsigned int TP_IN_API,
          unsigned int TP_OUT_API,
          unsigned int TP_NUM_INPUTS,
          unsigned int TP_WINDOW_VSIZE,
          unsigned int TP_NUM_OUTPUT_CLONES>
class kernelClass {
   private:
    // Parameter value defensive and legality checks
    static_assert(TP_IN_API <= 2, "ERROR: Unsupported TP_IN_API value set. ");
    static_assert(TP_WINDOW_VSIZE * sizeof(TT_DATA) % 16 == 0, "ERROR: TP_WINDOW_VSIZE must be a multiple of 128bits");

   public:
    // Constructor
    kernelClass() {}

    void kernelClassMain(T_inputIF<TT_DATA, TP_IN_API> inInterface, T_outputIF<TT_DATA, TP_OUT_API> outInterface);
};

template <typename TT_DATA, unsigned int TP_NUM_INPUTS, unsigned int TP_WINDOW_VSIZE, unsigned int TP_NUM_OUTPUT_CLONES>
class kernelClass<TT_DATA, kStreamAPI, kWindowAPI, TP_NUM_INPUTS, TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES> {
   private:
   public:
    // Constructor
    kernelClass() {}

    void kernelClassMain(T_inputIF<TT_DATA, kStreamAPI> inInterface, T_outputIF<TT_DATA, kWindowAPI> outInterface);
};

template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE, unsigned int TP_NUM_OUTPUT_CLONES>
class kernelClass<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES> {
   private:
   public:
    // Constructor
    kernelClass() {}

    void kernelClassMain(T_inputIF<TT_DATA, kWindowAPI> inInterface, T_outputIF<TT_DATA, kStreamAPI> outInterface);
};

//-----------------------------------------------------------------------------------------------------
// Single kernel base specialization. Used for single window to single window copy
template <typename TT_DATA,
          unsigned int TP_IN_API,
          unsigned int TP_OUT_API,
          unsigned int TP_NUM_INPUTS,
          unsigned int TP_WINDOW_VSIZE,
          unsigned int TP_NUM_OUTPUT_CLONES>
class widget_api_cast
    : public kernelClass<TT_DATA, TP_IN_API, TP_OUT_API, TP_NUM_INPUTS, TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES> {
   public:
    // Constructor
    widget_api_cast()
        : kernelClass<TT_DATA, TP_IN_API, TP_OUT_API, TP_NUM_INPUTS, TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_window<TT_DATA>* __restrict inWindow, output_window<TT_DATA>* __restrict outWindow0);
};

// Specialization for single window in, dual window out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2>
    : public kernelClass<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_window<TT_DATA>* __restrict inWindow0,
                      output_window<TT_DATA>* __restrict outWindow0,
                      output_window<TT_DATA>* __restrict outWindow1);
};

// Specialization for single window in, triple window out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3>
    : public kernelClass<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_window<TT_DATA>* __restrict inWindow0,
                      output_window<TT_DATA>* __restrict outWindow0,
                      output_window<TT_DATA>* __restrict outWindow1,
                      output_window<TT_DATA>* __restrict outWindow2);
};

// stream to  window, 1 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 1>
    : public kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 1> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_stream<TT_DATA>* __restrict inStream0, output_window<TT_DATA>* __restrict outWindow0);
};

// stream to  window, 2 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2>
    : public kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_stream<TT_DATA>* __restrict inStream0,
                      output_window<TT_DATA>* __restrict outWindow0,
                      output_window<TT_DATA>* __restrict outWindow1);
};

// stream to  window, 3 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3>
    : public kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_stream<TT_DATA>* __restrict inStream0,
                      output_window<TT_DATA>* __restrict outWindow0,
                      output_window<TT_DATA>* __restrict outWindow1,
                      output_window<TT_DATA>* __restrict outWindow2);
};

// stream to  window, 4 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 4>
    : public kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 4> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 4>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_stream<TT_DATA>* __restrict inStream0,
                      output_window<TT_DATA>* __restrict outWindow0,
                      output_window<TT_DATA>* __restrict outWindow1,
                      output_window<TT_DATA>* __restrict outWindow2,
                      output_window<TT_DATA>* __restrict outWindow3);
};

// dual stream input
// stream to  window, 2 in 1 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 1>
    : public kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 1> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_stream<TT_DATA>* __restrict inStream0,
                      input_stream<TT_DATA>* __restrict inStream1,
                      output_window<TT_DATA>* __restrict outWindow0);
};

// stream to  window, 2 in 2 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 2>
    : public kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 2> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_stream<TT_DATA>* __restrict inStream0,
                      input_stream<TT_DATA>* __restrict inStream1,
                      output_window<TT_DATA>* __restrict outWindow0,
                      output_window<TT_DATA>* __restrict outWindow1);
};

// stream to  window, 2 in 3 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 3>
    : public kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 3> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 3>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_stream<TT_DATA>* __restrict inStream0,
                      input_stream<TT_DATA>* __restrict inStream1,
                      output_window<TT_DATA>* __restrict outWindow0,
                      output_window<TT_DATA>* __restrict outWindow1,
                      output_window<TT_DATA>* __restrict outWindow2);
};

// stream to  window, 2 in 4 out
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 4>
    : public kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 4> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 4>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_stream<TT_DATA>* __restrict inStream0,
                      input_stream<TT_DATA>* __restrict inStream1,
                      output_window<TT_DATA>* __restrict outWindow0,
                      output_window<TT_DATA>* __restrict outWindow1,
                      output_window<TT_DATA>* __restrict outWindow2,
                      output_window<TT_DATA>* __restrict outWindow3);
};

// Window to stream 1 to 1
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 1>
    : public kernelClass<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 1> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 1>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_window<TT_DATA>* __restrict inWindow0, output_stream<TT_DATA>* __restrict outStream0);
};

// Window to stream 1 to 2
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
class widget_api_cast<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 2>
    : public kernelClass<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 2> {
   public:
    // Constructor
    widget_api_cast() : kernelClass<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 2>() {}

    // Register Kernel Class
    static void registerKernelClass() { REGISTER_FUNCTION(widget_api_cast::transferData); }

    // Main function
    void transferData(input_window<TT_DATA>* __restrict inWindow0,
                      output_stream<TT_DATA>* __restrict outStream0,
                      output_stream<TT_DATA>* __restrict outStream1);
};
}
}
}
}
}

#endif // _DSPLIB_WIDGET_API_CAST_HPP_
