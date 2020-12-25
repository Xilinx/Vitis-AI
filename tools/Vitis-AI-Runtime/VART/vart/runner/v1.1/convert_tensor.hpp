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
#pragma once
#include <vitis/ai/tensor.hpp>
#include <xir/tensor/tensor.hpp>

vitis::ai::Tensor::DataType convert_data_type(xir::DataType data_type);

unique_ptr<vitis::ai::Tensor> convert_tensor(const xir::Tensor* xir_tensor);
unique_ptr<xir::Tensor> convert_tensor(const vitis::ai::Tensor* xir_tensor);

std::vector<unique_ptr<vitis::ai::Tensor>> convert_tensors(
    const std::vector<const xir::Tensor*>& xir_tensors);
