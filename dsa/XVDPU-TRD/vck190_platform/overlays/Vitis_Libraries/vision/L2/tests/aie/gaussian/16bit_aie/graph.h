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

#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <adf.h>
#include <common/xf_aie_const.hpp>
#include "kernels.h"

static constexpr int TILE_ELEMENTS = 2048;
static constexpr int TILE_WINDOW_SIZE = TILE_ELEMENTS * sizeof(int32_t) + xf::cv::aie::METADATA_SIZE;

using namespace adf;

class gaussianGraph : public graph {
   private:
    kernel gauss1;

   public:
    port<input> in;
    port<output> out;

    gaussianGraph() {
        gauss1 = kernel::create(gaussian);

        connect<adf::window<TILE_WINDOW_SIZE> >(in, gauss1.in[0]);
        connect<adf::window<TILE_WINDOW_SIZE> >(gauss1.out[0], out);

        source(gauss1) = "xf_gaussian.cc";

        runtime<ratio>(gauss1) = 0.5;
    };
};

#endif
