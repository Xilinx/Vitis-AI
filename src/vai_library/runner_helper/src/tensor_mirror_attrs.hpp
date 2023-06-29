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

#include "xir/tensor/tensor.hpp"
namespace vart {
class TensorMirrorAttrs : public xir::Tensor {
 public:
  static std::unique_ptr<TensorMirrorAttrs> create(
      const xir::Tensor* other, const std::vector<std::int32_t> shape = {},
      const xir::DataType data_type = xir::DataType{xir::DataType::UNKNOWN, 0});
  TensorMirrorAttrs(const xir::Tensor* other,
                    const std::vector<std::int32_t> shape = {},
                    const xir::DataType data_type = xir::DataType{
                        xir::DataType::UNKNOWN, 0});

 private:
  virtual const std::string get_name() const override;
  virtual const xir::Op* get_producer() const override;
  virtual xir::Op* get_producer() override;
  virtual const std::vector<std::int32_t> get_shape() const override;
  const std::vector<std::int32_t> get_dims() const override;
  const std::int32_t get_dim_num() const override;
  const std::int32_t get_dim_size(std::int32_t idx) const override;
  const std::int64_t get_element_num() const override;
  const xir::DataType& get_data_type() const override;
  const std::int32_t get_bit_width() const override;
  const std::uint64_t get_data_size() const override;
  std::unique_ptr<xir::Attrs> get_attrs() const override;
  TensorMirrorAttrs* set_attrs(std::unique_ptr<xir::Attrs> attrs) override;
  const bool has_attr(const std::string& key) const override;
  const xir::any get_attr(const std::string& key) const override;
  TensorMirrorAttrs* set_attr(const std::string& key,
                              const xir::any& value) override;
  TensorMirrorAttrs* rename(const std::string& name) override;
  const std::string to_string(
      const std::string& delimiter = ",",     //
      const std::string& left_bracket = "{",  //
      const std::string& right_bracket = "}") const override;

 public:
  virtual ~TensorMirrorAttrs();

 private:
  const xir::Tensor* other_;
  const std::vector<std::int32_t> shape_;
  const xir::DataType data_type_;
};

}  // namespace vart
