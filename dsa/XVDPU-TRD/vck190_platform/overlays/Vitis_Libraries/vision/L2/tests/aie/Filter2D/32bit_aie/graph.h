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
//#include "gauss2_stitcher.h"
//#include "gauss2_tiler.h"

using namespace adf;

class two_node_pipeline : public graph {
   private:
    // kernel tiler;
    // kernel stitcher;
    kernel gauss1;
    kernel gauss2;

   public:
    port<input> in;
    port<output> out;

    two_node_pipeline() {
        // tiler    = kernel::create(gauss2_tiler);
        gauss1 = kernel::create(filter2D);
        gauss2 = kernel::create(filter2D);
        // stitcher = kernel::create(gauss2_stitcher);

        // fabric<fpga>(tiler);
        // fabric<fpga>(stitcher);

        // connect< stream >(in,tiler.in[0]);

        // gauss1 processes 4096 32b blocks or 16384 byte blocks
        connect<adf::window<16384> >(in, gauss1.in[0]);
        // connect< stream, window<16384> >(tiler.out[0], gauss1.in[0]);

        // gauss1 window passsed directly to gauss2
        connect<window<16384> >(gauss1.out[0], gauss2.in[0]);
        // gauss2 window passed to output
        connect<window<16384>, stream>(gauss2.out[0], out);
        // connect< window<16384>, stream >(gauss2.out[0], stitcher.in[0]);

        // connect< stream >(stitcher.out[0],out);

        // Pull the source from previous lab
        // source(tiler)    = "kernels/gauss2_tiler.cpp";
        source(gauss1) = "xf_filter2d.cc";
        source(gauss2) = "xf_filter2d.cc";
        // source(stitcher) = "kernels/gauss2_stitcher.cpp";

        // Initial mapping
        runtime<ratio>(gauss1) = 0.5;
        runtime<ratio>(gauss2) = 0.5;
    };
};

#endif
