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

#include "xir/tensor/tensor_imp.hpp"
#include <algorithm>
#include <cmath>
#include "UniLog/UniLog.hpp"
#include "xir/graph/graph.hpp"
#include "xir/op/op_imp.hpp"
#include "xir/tensor/tensor.hpp"
#include "xir/util/tool_function.hpp"

namespace xir {

TensorImp::TensorImp(const std::string& name,
                     const std::vector<std::int32_t>& dims,
                     const DataType& data_type)
    : name_(name), shape_(dims), data_type_(data_type) {
  UNI_LOG_CHECK(this->data_type_.valid(), XIR_INVALID_DATA_TYPE)
      << this->to_string() << "'s data type, " << this->data_type_.to_string()
      << " is invalid.";
  for (int i = 0; i < static_cast<int>(shape_.size()); i++)
    UNI_LOG_CHECK((shape_[i] >= 0) | (shape_[i] == -1), XIR_MEANINGLESS_VALUE)
        << "The shape of each dim can only >= 0 or == -1, -1 means that this "
           "value has not been specified yet. But tensor \""
        << name << "\"'s dims are " << xir::to_string(dims) << ".";
};

const std::string TensorImp::get_name() const { return name_; }

const Op* TensorImp::get_producer() const {
  return const_cast<const Op*>(this->producer_);
}

Op* TensorImp::get_producer() { return this->producer_; }

const std::vector<std::int32_t> TensorImp::get_shape() const {
  return this->shape_;
}

// TODO: legacy API
const std::vector<std::int32_t> TensorImp::get_dims() const {
  return this->shape_;
}

const std::int32_t TensorImp::get_dim_num() const {
  return this->shape_.size();
}
const std::int32_t TensorImp::get_dim_size(std::int32_t idx) const {
  return this->shape_.at(idx);
}

const std::int64_t TensorImp::get_element_num() const {
  if (shape_.size() == 0)
    return 0;
  else {
    std::int64_t size = 1;
    for (auto dim : this->shape_) {
      UNI_LOG_CHECK(dim != -1, XIR_OUT_OF_RANGE)
          << "the shape of each dimension has not been specified, so you "
             "cannot get the number of elements in this tensor.";
      size *= dim;
    }
    return size;
  }
}

const DataType& TensorImp::get_data_type() const { return data_type_; }

const std::int32_t TensorImp::get_bit_width() const {
  return this->data_type_.bit_width;
}

const std::uint64_t TensorImp::get_data_size() const {
  UNI_LOG_CHECK(this->data_type_.valid(), XIR_UNKNOWNTYPE_TENSOR)
      << this->to_string() << "'s data type is invalid";
  std::uint64_t size =
      static_cast<std::uint64_t>(get_element_num()) * this->data_type_.bit_width;
  auto ret =
      static_cast<std::uint64_t>(std::ceil(static_cast<double>(size) / 8.0));
  if (size % 8 != 0)
    UNI_LOG_DEBUG_WARNING << "the size of tensor " << this->name_ << " is "
                          << size << " bits."
                          << " and " << ret << " bytes.";
  return ret;
}

std::unique_ptr<Attrs> TensorImp::get_attrs() const {
  if (nullptr == attrs_) {
    return Attrs::create();
  } else {
    return Attrs::clone(attrs_.get());
  }
}

Tensor* TensorImp::set_attrs(std::unique_ptr<Attrs> attrs) {
  //  UNI_LOG_CHECK(attrs != nullptr, XIR_OUT_OF_RANGE)
  //      << "You cannot set an empty Attribute pointer to this tensor.";
  attrs_ = std::move(attrs);
  return this;
}

const bool TensorImp::has_attr(const std::string& key) const {
  if (nullptr == attrs_) {
    return false;
  }
  return attrs_->has_attr(key);
}

const xir::any TensorImp::get_attr(const std::string& key) const {
  UNI_LOG_CHECK(attrs_->has_attr(key), XIR_UNREGISTERED_ATTR)
      << "Attrs doesn't contain attribute " << key;
  return attrs_->get_attr(key);
}

Tensor* TensorImp::set_attr(const std::string& key, const xir::any& value) {
  if (nullptr == attrs_) {
    attrs_ = Attrs::create();
  }
  attrs_->set_attr(key, value);
  return this;
}

Tensor* TensorImp::rename(const std::string& name) {
  if (this->name_ != name) {
    if (nullptr != this->producer_) {
      // already attached to the producer op, check the name identity
      auto all_ops = this->producer_->get_graph()->get_ops();
      auto op_same_name_it =
          std::find_if(all_ops.begin(), all_ops.end(), [name](const Op* op) {
            return !(static_cast<const OpImp*>(op)->to_be_removed_) &&
                   op->get_output_tensor()->get_name() == name;
          });
      UNI_LOG_CHECK(all_ops.end() == op_same_name_it, XIR_MULTI_DEFINED_TENSOR)
          << producer_->to_string() << "'s output tensor can't be renamed to "
          << name << ", because this name is alread used by "
          << (*op_same_name_it)->to_string() << "'s output tensor.";
    }
    this->name_ = name;
  }
  return this;
}

const std::string TensorImp::to_string(const std::string& delimiter,     //
                                       const std::string& left_bracket,  //
                                       const std::string& right_bracket) const {
  std::ostringstream out;
  out << "xir::Tensor" << left_bracket                                 //
      << "name = " << this->get_name() << delimiter                    //
      << " type = " << this->get_data_type().to_string() << delimiter  //
      << " shape = " << xir::to_string(this->get_shape())              //
      << right_bracket;
  return out.str();
}

};  // namespace xir
