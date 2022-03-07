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
#include "kernels.h"
#include "config.h"

using namespace adf;

class myGraph : public adf::graph {
   private:
    kernel k1;

   public:
    port<input> inprt1;
    port<input> inprt2;
    port<input> alpha;
    port<output> outprt;

    myGraph() {
        k1 = kernel::create(accumulate_weighted);

        connect<window<TILE_WINDOW_SIZE> >(inprt1, k1.in[0]);
        connect<window<TILE_WINDOW_SIZE> >(inprt2, k1.in[1]);
        connect<parameter>(alpha, async(k1.in[2]));

        connect<window<TILE_WINDOW_SIZE> >(k1.out[0], outprt);

        source(k1) = "xf_accumulate_weighted.cc";
        // Initial mapping
        runtime<ratio>(k1) = 0.5;
    };
};

#endif
