/*
 * Copyright 2019 Xilinx Inc.
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
#include "vart/tensor_buffer_allocator.hpp"

#include <dlfcn.h>
#include <glog/logging.h>

#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"
DEF_ENV_PARAM_2(XLNX_TENSOR_BUFFER_ALLOCATOR_PLUGIN, "libvart-mem-manager.so.1",
                std::string);
namespace vart {
TensorBufferAllocator::TensorBufferAllocator() {}

std::unique_ptr<TensorBufferAllocator> TensorBufferAllocator::create(
    const xir::Attrs* attrs) {
  typedef TensorBufferAllocator* (*INIT_FUN)(const xir::Attrs* attrs);
  auto lib = ENV_PARAM(XLNX_TENSOR_BUFFER_ALLOCATOR_PLUGIN).c_str();
  auto handle = dlopen(lib, RTLD_LAZY);
  CHECK(handle != NULL) << "cannot open library!"
                        << " lib=" << lib << ";error=" << dlerror();
  dlerror();
  auto init_fun = (INIT_FUN)dlsym(handle, "create_tensor_buffer_allocator");
  CHECK(init_fun != NULL) << "cannot load symbol 'create_runner'!"
                          << "! lib=" << lib << ";error=" << dlerror();
  return std::unique_ptr<TensorBufferAllocator>(init_fun(attrs));
}

}  // namespace vart
