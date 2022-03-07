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

class preprocessgraph : public adf::graph {
   private:
    kernel k1;

   public:
    port<input> in1;
    port<output> out;
    port<input> alpha;
#if OP_MODE == 1 || OP_MODE == 2
    port<input> beta;
#endif
#if OP_MODE == 2
    port<input> gamma;
#endif

    preprocessgraph() {
        k1 = kernel::create(pp_top);

        connect<adf::window<TILE_WINDOW_SIZE> >(in1, k1.in[0]);

        // blob window passsed directly to output
        connect<window<TILE_WINDOW_SIZE> >(k1.out[0], out);
        connect<parameter>(alpha, async(k1.in[1]));
#if OP_MODE == 1 || OP_MODE == 2
        connect<parameter>(beta, async(k1.in[2]));
#endif
#if OP_MODE == 2
        connect<parameter>(gamma, async(k1.in[3]));
#endif

        source(k1) = "xf_pp_top.cc";

        // Initial mapping
        runtime<ratio>(k1) = 0.5;
    };
};

#endif
