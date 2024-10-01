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
#include "xir/dpu_controller.hpp"
#include <memory>
namespace {
typedef unsigned long task_handle_t;
class DpuControllerDnndk : public xir::DpuController {
public:
  DpuControllerDnndk();
  virtual ~DpuControllerDnndk();
  DpuControllerDnndk(const DpuControllerDnndk &other) = delete;
  DpuControllerDnndk &operator=(const DpuControllerDnndk &rhs) = delete;

public:
  virtual void run(const uint64_t code, const std::vector<uint64_t> &gen_reg,
                   int device_id = 0) override;

private:
  task_handle_t allocate_task_id();

private:
  int fd_;
  unsigned long task_id_;
};
} // namespace
