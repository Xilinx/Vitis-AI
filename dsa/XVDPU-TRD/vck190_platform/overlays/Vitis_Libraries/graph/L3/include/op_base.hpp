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

#ifndef _XF_GRAPH_L3_OP_BASE_HPP_
#define _XF_GRAPH_L3_OP_BASE_HPP_

#include "common.hpp"
#include "task.hpp"

namespace xf {
namespace graph {
namespace L3 {

class opBase {
   public:
    std::array<queue, 1> task_queue;

    opBase(){};

    void initThread(class openXRM* xrm,
                    std::string kernelName,
                    std::string kernelAlias,
                    unsigned int requestLoad,
                    unsigned int deviceNeeded,
                    unsigned int cuNumber) {
        task_workers.emplace_back(std::thread(worker, std::ref(task_queue[0]), xrm, kernelName, kernelAlias,
                                              requestLoad, deviceNeeded, cuNumber));
    };

    void initThreadInt(class openXRM* xrm,
                       std::string kernelName,
                       std::string kernelAlias,
                       unsigned int requestLoad,
                       unsigned int deviceNeeded,
                       unsigned int cuNumber) {
        task_workers.emplace_back(std::thread(worker2, std::ref(task_queue[0]), xrm, kernelName, kernelAlias,
                                              requestLoad, deviceNeeded, cuNumber));
    };

    void join() {
        task_queue[0].stop();
        task_workers[0].join();
    };

   private:
    std::vector<std::thread> task_workers;
};

} // L3
} // graph
} // xf

#endif
