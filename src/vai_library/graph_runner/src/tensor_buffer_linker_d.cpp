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
#include <UniLog/UniLog.hpp>
#include "./tensor_buffer_linker.hpp"

#include <memory>
#include <sstream>

std::unique_ptr<TensorBufferLinker> TensorBufferLinker::create(
    std::unique_ptr<vart::TensorBuffer>* master) {
  return std::make_unique<TensorBufferLinker>(master);
}

TensorBufferLinker::TensorBufferLinker(
    std::unique_ptr<vart::TensorBuffer>* master)
    : master_{master} {}

TensorBufferLinker::~TensorBufferLinker() {}

std::string TensorBufferLinker::to_string() {
  std::ostringstream str;
  str << "linker{master=" << master_->get()->to_string() << "; slaves=[";
  int c = 0;
  for (auto s : slaves_) {
    if (c != 0) {
      str << ",";
    }
    str << s->get()->to_string();
    c++;
  }
  str << "]}";
  return str.str();
}

void TensorBufferLinker::finalize() {
  // LOG(ERROR) << " please override this function";
  UNI_LOG_ERROR(VAILIB_GRAPH_RUNNER_NOT_OVERRIDE)
      << " please override this function";
}

void TensorBufferLinker::before_invoke_runner(const xir::Subgraph* subgraph) {
  // LOG(ERROR) << " please override this function";
  UNI_LOG_ERROR(VAILIB_GRAPH_RUNNER_NOT_OVERRIDE)
      << " please override this function";
}
