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

#include "xir/attrs/attrs.hpp"
#include "xir/tensor/tensor.hpp"
#include "xir/util/data_type.hpp"

namespace xir {
class GraphImp;

class TensorImp : public Tensor {
 public:
  TensorImp(const std::string& name, const std::vector<std::int32_t>& dims,
            const DataType& data_type);
  TensorImp(TensorImp&&) = default;
  TensorImp() = delete;
  virtual ~TensorImp() = default;

 public:
  // get name of tensor
  const std::string get_name() const override;

  // get producer op
  const Op* get_producer() const override;
  Op* get_producer() override;

  const std::vector<std::int32_t> get_shape() const override;

  // TODO: legacy API
  const std::vector<std::int32_t> get_dims() const override;
  std::int32_t get_dim_num() const override;
  std::int32_t get_dim_size(std::int32_t idx) const override;

  // get the number of elements(data) in this tensor
  std::int64_t get_element_num() const override;

  // get the data type
  const DataType& get_data_type() const override;

  // TODO: legacy API
  std::int32_t get_bit_width() const override;

  // get the size of data in tensor, data_size = element_num * sizeof(datatype)
  std::uint64_t get_data_size() const override;

  // get attributes of this tensor
  std::unique_ptr<Attrs> get_attrs() const override;

  // set attributes of this tensor
  Tensor* set_attrs(std::unique_ptr<Attrs> attrs) override;

  bool has_attr(const std::string& key) const override;

  const xir::any get_attr(const std::string& key) const override;

  Tensor* set_attr(const std::string& key, const xir::any& value) override;

  Tensor* rename(const std::string& name) override;

  const std::string to_string(const std::string& delimiter = ",",     //
                              const std::string& left_bracket = "{",  //
                              const std::string& right_bracket = "}") const;

 private:
  std::string name_;
  std::vector<std::int32_t> shape_;
  DataType data_type_;
  std::unique_ptr<Attrs> attrs_;
  friend class c_api;
  // keep the producer_ op, default is nullptr
  Op* producer_{nullptr};
  friend class OpImp;
};

}  // namespace xir
