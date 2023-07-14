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
#pragma once
#include <memory>

#include "xir/graph/graph.hpp"

namespace vitis {
namespace ai {
class XmodelJit {
 public:
 public:
  XmodelJit() = default;
  virtual ~XmodelJit() = default;
  XmodelJit(const XmodelJit& other) = delete;
  XmodelJit& operator=(const XmodelJit& rhs) = delete;

 public:
  static std::unique_ptr<XmodelJit> create(xir::Graph* graph);

 public:
  virtual int jit() = 0;
};
}  // namespace ai
}  // namespace vitis
