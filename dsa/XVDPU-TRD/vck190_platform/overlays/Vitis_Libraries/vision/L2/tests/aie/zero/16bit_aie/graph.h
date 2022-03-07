/*
 * Copyright 2021 Xilinx, Inc.
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

#ifndef ADF_GRAPH_H
#define ADF_GRAPH_H

#include <adf.h>
#include <common/xf_aie_const.hpp>
#include "kernels.h"

static constexpr int TILE_ELEMENTS = 4096;
static constexpr int TILE_WINDOW_SIZE = TILE_ELEMENTS * sizeof(int16_t) + xf::cv::aie::METADATA_SIZE;

using namespace adf;

/*
 * Cardano dataflow graph to compute weighted moving average of
 * the last 8 samples in a stream of numbers
 */

class zeroGraph : public adf::graph {
   private:
    kernel k1;

   public:
    port<input> in1;
    port<input> maxval;
    port<output> out;

    zeroGraph() {
        // create kernels

        k1 = kernel::create(zero);

        // create nets to connect kernels and IO ports
        connect<window<TILE_WINDOW_SIZE> >(in1, k1.in[0]);
        connect<window<TILE_WINDOW_SIZE> >(k1.out[0], out);
        //    connect< parameter > (maxval, async(k1.in[1]));

        // specify kernel sources
        source(k1) = "xf_zero.cc";

        // specify kernel run times
        runtime<ratio>(k1) = 0.5;
    }
};

#endif
