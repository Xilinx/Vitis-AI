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
#include "../../uio_driver/src/uio_driver.hpp"
#include "../dpu_op_controller.hpp"
#include "vitis/ai/buffer_object_map.hpp"

#include <memory>

class DpuControllerUio : public vitis::ai::DpuController {
public:
  static std::shared_ptr<vitis::ai::DpuController> getInstance();

public:
  DpuControllerUio();
  virtual ~DpuControllerUio();
  DpuControllerUio(const DpuControllerUio &other) = delete;
  DpuControllerUio &operator=(const DpuControllerUio &rhs) = delete;

public:
  virtual void run(const uint64_t workspace, const uint64_t code,
                   const uint64_t paramter) override;

private:
  struct DpuRegs {
    uint64_t code;
    uint64_t regs[8];
  };
  int RunDpu(int dpu_index, const DpuRegs &regs);

private:
  std::unique_ptr<UioDriver<DpuDriver>> dpu_driver_;
  std::unique_ptr<vitis::ai::buffer_object_map> dpu_reg_map_;
};
