/*
 * Copyright 2019 xilinx Inc.
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
#include <cmath>
#include <iostream>
#include <vart/tensor_buffer.hpp>
#include <vector>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>
#include "../src/runner_helper.hpp"

int main() {
  std::vector<float> fin(1 * 10, 7.5f);
  auto fin_tensor = xir::Tensor::create(
      "test", {1, 10}, xir::DataType{xir::DataType::FLOAT, sizeof(float) * 8u});
  fin_tensor->set_attr<int>("fix_point", (int)2);
  auto tb_fin = vart::alloc_cpu_flat_tensor_buffer(fin_tensor.get());
  auto data_fin = tb_fin->data(std::vector<int32_t>{0, 0});
  float* data01 = (float*)data_fin.first;
  std::copy(fin.begin(),fin.end(),data01);
  std::cout << "tb_fin: " << tb_fin->to_string() << std::endl;

  auto iout_tensor = xir::Tensor::create(
      "test", {1, 10}, xir::DataType{xir::DataType::XINT, sizeof(int8_t) * 8u});
  auto tb_iout = vart::alloc_cpu_flat_tensor_buffer(iout_tensor.get());
  std::cout << "tb_iout: " << tb_iout->to_string() << std::endl;

  auto fout_tensor = xir::Tensor::create(
      "test", {1, 10}, xir::DataType{xir::DataType::FLOAT, sizeof(float) * 8u});
  fout_tensor->set_attr<int>("fix_point", (int)2);
  auto tb_fout = vart::alloc_cpu_flat_tensor_buffer(fout_tensor.get());
  auto data_fout = tb_fout->data(std::vector<int32_t>{0, 0});
  float* fout = (float*)data_fout.first;
  std::cout << "tb_fout: " << tb_fout->to_string() << std::endl;
  std::cout << std::endl;

  vart::TensorBuffer::copy_tensor_buffer(tb_fin.get(), tb_iout.get());
  vart::TensorBuffer::copy_tensor_buffer(tb_iout.get(), tb_fout.get());

  std::cout << "test copy tensor(FLOAT-XINT-FLOAT) result:" << std::endl;
  for (size_t i = 0; i < fin.size(); i++) {
    if (fin[i] != fout[i]) {
      std::cout << "error: " << fin[i] << " " << fout[i] << std::endl;
    } else {
      std::cout << "fin[" << i << "] = " << fin[i]  << "    ";
      std::cout << "test fin[" << i << "] ok"  << std::endl;
    }
  }
  std::cout << std::endl;


  fin[3] = 4.5f;
  std::copy(fin.begin(),fin.end(),data01);
  vart::TensorBuffer::copy_tensor_buffer(tb_fin.get(), tb_fout.get());

  std::cout << "test copy tensor(FLOAT-FLOAT) result:" << std::endl;
  for (size_t i = 0; i < fin.size(); i++) {
    if (fin[i] != fout[i]) {
      std::cout << "error: " << fin[i] << " " << fout[i] << std::endl;
    } else {
      std::cout << "fin[" << i << "] = " << fin[i]  << "    ";
      std::cout << "test fin[" << i << "] ok"  << std::endl;
    }
  }
  return 0;
}
