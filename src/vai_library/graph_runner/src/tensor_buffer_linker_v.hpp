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

#include "./tensor_buffer_linker.hpp"

class TensorBufferLinkerHostVirt : public TensorBufferLinker {
 public:
  explicit TensorBufferLinkerHostVirt(
      std::unique_ptr<vart::TensorBuffer>* master);
  virtual ~TensorBufferLinkerHostVirt();

 private:
  virtual void finalize() override;
  virtual void after_invoke_runner(const xir::Subgraph* subgraph) override;

 private:
  std::unique_ptr<vart::TensorBuffer>* replacement_=NULL;
  static constexpr int KEEP = 0;
  static constexpr int REPLACE = 1;
  static constexpr int THE_SELECTED = 2;
  std::vector<int> linker_decisions_;
};
