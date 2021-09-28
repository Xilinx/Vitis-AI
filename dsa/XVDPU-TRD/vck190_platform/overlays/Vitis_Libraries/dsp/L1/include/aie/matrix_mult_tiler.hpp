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
#ifndef _DSPLIB_MATRIX_MULT_TILER_HPP_
#define _DSPLIB_MATRIX_MULT_TILER_HPP_

#include <adf.h>

namespace xf {
namespace dsp {
namespace aie {
namespace blas {
namespace matrix_mult {
/*
* @brief Acts as a wrapper and the entry point from the graph.
*/
template <unsigned M, unsigned N, unsigned inRow, unsigned inCol, unsigned leadingDim, typename T_D>
class tilerKernelClass {
   public:
    void tile(input_window<T_D>* inWindow, output_window<T_D>* restrict outWindow);

    static void registerKernelClass() { REGISTER_FUNCTION(tilerKernelClass::tile); }
};
/*
  @brief Entry point from another kernel (using this as a function, instead of a subgraph.)
*/
// template<unsigned M, unsigned N, unsigned inRow, unsigned inCol, unsigned leadingDim, typename T_D >
// static void doTile(T_D* restrict inPtr, T_D* outPtr);
}
}
}
}
}

#endif