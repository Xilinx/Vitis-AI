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
#include "./configurable_dpu_task_imp.hpp"

namespace vitis {
namespace ai {
ConfigurableDpuTask::ConfigurableDpuTask() {}
ConfigurableDpuTask::~ConfigurableDpuTask() {}

std::unique_ptr<ConfigurableDpuTask> ConfigurableDpuTask::create(
    const std::string& model_name, bool need_preprocess) {
  return std::unique_ptr<ConfigurableDpuTask>(
      new ConfigurableDpuTaskImp(model_name, need_preprocess));
}

std::unique_ptr<ConfigurableDpuTask> ConfigurableDpuTask::create(
    const std::string& model_name, xir::Attrs* attrs, bool need_preprocess) {
  return std::unique_ptr<ConfigurableDpuTask>(
      new ConfigurableDpuTaskImp(model_name, attrs, need_preprocess));
}

ConfigurableDpuTaskBase::ConfigurableDpuTaskBase(const std::string& model_name,
                                                 bool need_preprocess)
    : configurable_dpu_task_{
          ConfigurableDpuTask::create(model_name, need_preprocess)} {};
ConfigurableDpuTaskBase::ConfigurableDpuTaskBase(const std::string& model_name,
                                                 xir::Attrs* attrs,
                                                 bool need_preprocess)
    : configurable_dpu_task_{
          ConfigurableDpuTask::create(model_name, attrs, need_preprocess)} {};

ConfigurableDpuTaskBase::~ConfigurableDpuTaskBase(){};

int ConfigurableDpuTaskBase::get_input_width() const {
  return configurable_dpu_task_->getInputWidth();
}

int ConfigurableDpuTaskBase::get_input_height() const {
  return configurable_dpu_task_->getInputHeight();
}

size_t ConfigurableDpuTaskBase::get_input_batch() const {
  return configurable_dpu_task_->get_input_batch();
}

int ConfigurableDpuTaskBase::get_input_buffer_size() const {
  return configurable_dpu_task_->get_input_buffer_size();
}
/**
 * @brief return the vector of input offset for zero copy
 */
size_t ConfigurableDpuTaskBase::get_input_offset() const {
  return configurable_dpu_task_->get_input_offset();
}

int ConfigurableDpuTaskBase::get_input_fix_point() const {
  return configurable_dpu_task_->get_input_fix_point();
}

}  // namespace ai
}  // namespace vitis
