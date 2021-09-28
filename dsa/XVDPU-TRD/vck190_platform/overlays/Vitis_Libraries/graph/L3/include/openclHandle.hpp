/*
 * Copyright 2020 Xilinx, Inc.
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

#pragma once

#ifndef _XF_GRAPH_L3_OPENCLHANDLE_HPP_
#define _XF_GRAPH_L3_OPENCLHANDLE_HPP_

#include "xcl2.hpp"

namespace xf {
namespace graph {
namespace L3 {

class clHandle {
   public:
    cl::Device device;
    cl::Context context;
    cl::CommandQueue q;
    cl::Program::Binaries xclBins;
    cl::Program program;
    cl::Buffer* buffer;
    unsigned int deviceID;
    unsigned int cuID;
    unsigned int dupID;
};

} // L3
} // graph
} // xf

#endif
