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

#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/centerpoint.hpp>
#include <xir/attrs/attrs.hpp>
#include "middle_process.hpp"
#include "./preprocess.hpp"

namespace vitis { namespace ai{ 

class CenterPointImp:  public CenterPoint
{
public:
  CenterPointImp(const std::string &model_name_0, const std::string &model_name_1);
  CenterPointImp(const std::string &model_name_0, const std::string &model_name_1, xir::Attrs *attrs);
  virtual ~CenterPointImp();

  virtual std::vector<CenterPointResult> run(const std::vector<float>& input) override;
  virtual std::vector<std::vector<CenterPointResult>> run(
      const std::vector<std::vector<float>> &inputs);

  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;
  virtual size_t get_input_batch() const override;
private:
  uint32_t points_dim_;
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_0_;
  std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_1_;
};

}}

