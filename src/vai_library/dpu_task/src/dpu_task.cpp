/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#include <iostream>
using namespace std;

#include "./dpu_task_imp.hpp"
#include "vitis/ai/dpu_task.hpp"
using namespace vitis::ai;

DpuTask::DpuTask() {}
DpuTask::~DpuTask() {}

std::unique_ptr<DpuTask> DpuTask::create(const std::string& model_name) {
  // supress coverity complain
  try{ 
    std::unique_ptr<DpuTask> ret = std::unique_ptr<DpuTask>(new DpuTaskImp(model_name));
    return ret;
  }catch(...){
    std::cerr <<"should never run here ...\n";
    abort();
  }
}
std::unique_ptr<DpuTask> DpuTask::create(const std::string& model_name, xir::Attrs *attrs) {
  // supress coverity complain
  try{ 
    std::unique_ptr<DpuTask> ret = std::unique_ptr<DpuTask>(new DpuTaskImp(model_name, attrs));
    return ret;
  }catch(...){
    std::cerr <<"should never run here ...\n";
    abort();
  }
}

