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
#include <xir/tensor/tensor.hpp>
struct my_tensor_t {
  my_tensor_t(const xir::Tensor* xir_tensor,
              std::unique_ptr<xir::Tensor> vitis_tensor, const size_t reg_id,
              const size_t ddr_addr, const size_t size,
              const size_t location  // ddr: 1, bank: 0
              )
      : xir_tensor_{xir_tensor},                 // per xir graph
        vitis_tensor_{std::move(vitis_tensor)},  // per runner
        reg_id_{reg_id},
        ddr_addr_{ddr_addr},
        size_{size},
        location_{location}  // ddr: 1, bank: ,
  {}

  // return the feature map size for one bactch
  inline size_t get_batch_size() const {
    return vitis_tensor_->get_shape().at(0);
  }

  inline size_t get_feature_map_size() const {
    return xir_tensor_->get_element_num() / xir_tensor_->get_shape().at(0);
  }
  size_t get_reg_id() const { return reg_id_; };
  size_t get_ddr_addr() const { return ddr_addr_; };
  size_t get_size() const { return size_; };
  size_t get_location() const { return location_; }
  std::string get_name() const { return vitis_tensor_->get_name(); }

  const xir::Tensor* get_tensor() const { return vitis_tensor_.get(); }
  const xir::Tensor* get_xir_tensor() const { return xir_tensor_; }

  friend std::ostream& operator<<(std::ostream& out,
                                  const my_tensor_t& my_tensor);

 private:
  const xir::Tensor* xir_tensor_;              // per xir graph
  std::unique_ptr<xir::Tensor> vitis_tensor_;  // per runner
  const size_t reg_id_;
  const size_t ddr_addr_;
  const size_t size_;
  const size_t location_;  // ddr: 1, bank: 0
};

inline size_t get_feature_map_size(const xir::Tensor& my_tensor) {
  return my_tensor.get_element_num() / my_tensor.get_shape().at(0);
}

inline size_t get_batch_size(const vart::TensorBuffer* tb) {
  return tb->get_tensor()->get_shape().at(0);
}
