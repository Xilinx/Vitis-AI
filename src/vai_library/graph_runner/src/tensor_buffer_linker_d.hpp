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

#include "vart/tensor_buffer.hpp"
#include "xir/graph/subgraph.hpp"

class TensorBufferLinker {
 public:
  static std::unique_ptr<TensorBufferLinker> create(
      std::unique_ptr<vart::TensorBuffer>* tb);
  explicit TensorBufferLinker(std::unique_ptr<vart::TensorBuffer>* master);
  virtual ~TensorBufferLinker();

 public:
  void add_slave(std::unique_ptr<vart::TensorBuffer>* slave) {
    slaves_.push_back(slave);
  }
  std::string to_string();

  virtual void finalize();
  virtual void before_invoke_runner(const xir::Subgraph* subgraph);

 private:
  std::unique_ptr<vart::TensorBuffer>* master_;
  std::vector<std::unique_ptr<vart::TensorBuffer>*> slaves_;
};
