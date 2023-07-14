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

#include "xir/util/tool_function.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <typeindex>

#include "../../../3rd-party/hash-library/md5.h"
#include "UniLog/UniLog.hpp"
#include "config.hpp"
#include "internal_util.hpp"
#include "xir/op/op_def.hpp"
#include "xir/op/op_def_factory_imp.hpp"
#include "xir/op/shape_inference.hpp"
#include "xir/util/data_type.hpp"

namespace xir {

const std::string get_md5_of_buffer(const void* buf, size_t size) {
  return MD5()(buf, size);
}

const std::string get_md5_of_file(const std::string& filepath) {
  std::ifstream file(filepath.c_str(), std::ifstream::in | std::ifstream::binary);
  UNI_LOG_CHECK(!file.fail(), XIR_FILE_NOT_EXIST)
      << filepath << " doesn't exist.";
  UNI_LOG_DEBUG_INFO << "Read all data from " << filepath
                     << " to calculate the md5sum.";

  const std::uint32_t buffer_size = 1024;
  char* buffer;
  buffer = new char[buffer_size];
  auto strm = MD5();
  while (file.read(buffer, buffer_size)) {
    strm.add(buffer, buffer_size);
  }
  auto count = file.gcount();
  strm.add(buffer, count);
  delete[] buffer;
  file.close();
  auto ret = strm.getHash();
  UNI_LOG_DEBUG_INFO << "md5sum(" << filepath << ") = " << ret << ".";
  return ret;
}

XIR_DLLESPEC const std::string get_lib_name() {
  const auto ret =
      std::string{PROJECT_NAME} + "." + std::string{PROJECT_VERSION};
  return ret;
}

XIR_DLLESPEC const std::string get_lib_id() {
  const auto ret = std::string{PROJECT_GIT_COMMIT_ID};
  return ret;
}

// name related
void add_prefix_helper(std::string& name, const std::string& prefix) {
  std::string prefix_inst = HEAD_DELIMITER + prefix + TAIL_DELIMITER;
  size_t insert_pos = 0;
  // name begin with  "__" will be hidden in serilization, so jump 2 position
  // before the prefix
  if (name.find_first_of("__") == 0) {
    insert_pos += 2;
  }
  name.insert(insert_pos, prefix_inst);
}

void add_suffix_helper(std::string& name, const std::string& suffix) {
  std::string suffix_inst = HEAD_DELIMITER + suffix + TAIL_DELIMITER;
  name += suffix_inst;
}

std::string remove_xfix(const std::string& name) {
  return *(extract_xfix(name).rbegin());
}

std::vector<std::string> extract_xfix(const std::string& name) {
  std::string head_delimiter = HEAD_DELIMITER;
  std::string tail_delimiter = TAIL_DELIMITER;
  std::string ret = name;
  std::vector<std::string> ret_vec;
  std::vector<std::size_t> head_pos_vec, tail_pos_vec;
  auto head_delimiter_pos = ret.find_first_of(head_delimiter);
  if (std::string::npos != head_delimiter_pos) {
    head_pos_vec.push_back(head_delimiter_pos);
  }
  while (head_pos_vec.size()) {
    auto current_head_delimiter_pos = *(head_pos_vec.rbegin());
    auto tail_delimiter_pos = ret.find(
        tail_delimiter, current_head_delimiter_pos + head_delimiter.size());
    if (std::string::npos != tail_delimiter_pos) {
      auto next_head_delimiter_pos = ret.find_first_of(
          head_delimiter, current_head_delimiter_pos + head_delimiter.size());
      // check if there a matching pair
      if ((std::string::npos == next_head_delimiter_pos) ||
          (next_head_delimiter_pos > tail_delimiter_pos)) {
        // if there's no next head delimiter or next head delimiter is after the
        // tail delimiter, remove one
        // collect the xfix
        ret_vec.push_back(
            ret.substr(current_head_delimiter_pos + head_delimiter.size(),
                       tail_delimiter_pos - current_head_delimiter_pos -
                           head_delimiter.size()));
        // remove the xfix and head_delimiter and tail_delimiter pair
        ret.erase(current_head_delimiter_pos, tail_delimiter_pos -
                                                  current_head_delimiter_pos +
                                                  tail_delimiter.size());
        // remove the current_head_delimiter_pos in head_pos_vec
        head_pos_vec.pop_back();
        // find the next head delimiter
        current_head_delimiter_pos =
            head_pos_vec.size() ? (*(head_pos_vec.rbegin())) : 0;
        next_head_delimiter_pos = ret.find_first_of(
            head_delimiter, current_head_delimiter_pos + head_delimiter.size());
        if (std::string::npos != next_head_delimiter_pos) {
          head_pos_vec.push_back(next_head_delimiter_pos);
        }
      } else {
        head_pos_vec.push_back(next_head_delimiter_pos);
        continue;
      }
    } else {
      // if there's no more tail_delimiter, break the loop
      break;
    }
  }
  ret_vec.push_back(ret);
  return ret_vec;
}

// math related
float xround(const float& data, const std::string& round_mode) {
  float ret;
  if ("STD_ROUND" == round_mode) {
    ret = std::round(data);
  } else if ("DPU_ROUND" == round_mode) {
    ret = internal::dpu_round_float(data);
  } else if ("PY3_ROUND" == round_mode) {
    ret = internal::py3_round_float(data);
  } else {
    UNI_LOG_FATAL(XIR_UNSUPPORTED_ROUND_MODE)
        << round_mode
        << " is not supported by xir now, if you require this mode, please "
           "contact us.";
  }
  return ret;
}

void register_customized_operator_definition(const std::string& name,
                                             const std::string& type) {
  UNI_LOG_WARNING
      << "The operator named " << name << ", type: " << type
      << ", is not defined in XIR. XIR creates the definition of this "
         "operator automatically. "
      << "You should specify the shape and "
         "the data_type of the output tensor of this operation by "
         "set_attr(\"shape\", std::vector<int>) and "
         "set_attr(\"data_type\", std::string)";
  auto new_operator =
      xir::OpDef(type)
          .add_input_arg(xir::OpArgDef{"input", OpArgDef::REPEATED,
                                       xir::DataType::FLOAT, ""})
          .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
              "shape", AttrDef::REQUIRED, 0,
              "`Datatype`: `vector<int>`\n\n"
              "The shape of the output tensor"))
          .add_attr(xir::AttrDefBuilder<std::string>::build(
              "data_type", AttrDef::REQUIRED,
              "`Datatype`: `string`\n\n"
              "The data type of the data of output feature maps, "
              "we use FLOAT32 as the default."))
          .set_annotation("This operator is not defined by XIR.")
          .set_shape_infer(xir::shape_infer_data);
  op_def_factory()->register_h(new_operator);
}

std::vector<float> get_float_vec_from_any(const xir::any& any) {
  auto type = std::type_index(any.type());
  auto f_vec = std::type_index(typeid(std::vector<float>));
  auto i_vec = std::type_index(typeid(std::vector<std::int32_t>));
  std::vector<float> fs;
  if (type == i_vec) {
    auto is = std::any_cast<std::vector<std::int32_t>>(any);
    for (auto i : is) fs.push_back(static_cast<float>(i));
  } else if (type == f_vec) {
    fs = std::any_cast<std::vector<float>>(any);
  } else {
    UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
        << "I cannot transform this xir::any to float.";
  }
  return fs;
}

bool TensorLexicographicalOrder(Tensor* a, Tensor* b) {
  return a->get_name() < b->get_name();
}

void validate_strided_slice(const xir::Op* cur,
                            std::vector<int32_t>& dense_begin,
                            std::vector<int32_t>& dense_end,
                            std::vector<int32_t>& dense_strides,
                            std::vector<int32_t>& out_shape) {
  std::int32_t in_shape_num =
      cur->get_input_tensor("input")->get_shape().size();
  auto in_dims = cur->get_input_tensor("input")->get_shape();
  auto begin = cur->get_attr<std::vector<std::int32_t>>("begin");
  auto end = cur->get_attr<std::vector<std::int32_t>>("end");
  auto strides = cur->get_attr<std::vector<std::int32_t>>("strides");

  const bool begin_is_wrong =
      !begin.empty() && !(begin.size() == strides.size() &&
                          begin.size() < 32 /* using 32 bit masks */);
  const bool end_is_wrong = !end.empty() && !(end.size() == strides.size());
  if (begin_is_wrong || end_is_wrong) {
    if (!begin.empty() && !end.empty()) {
      UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
          << "Expected begin, end, and strides to be 1D equal size tensors, "
          << "but got shapes " << xir::to_string(begin) << ", "
          << xir::to_string(end) << ", and " << xir::to_string(strides)
          << " instead.";
    } else {
      UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
          << "Expected begin, end, and strides to be 1D equal size tensors, "
          << "but got shape " << xir::to_string(strides) << " for strides.";
    }
  }
  auto begin_mask = 0;
  if (cur->has_attr("begin_mask"))
    begin_mask = cur->get_attr<std::int32_t>("begin_mask");
  auto end_mask = 0;
  if (cur->has_attr("end_mask"))
    end_mask = cur->get_attr<std::int32_t>("end_mask");
  auto ellipsis_mask = 0;
  if (cur->has_attr("ellipsis_mask"))
    ellipsis_mask = cur->get_attr<std::int32_t>("ellipsis_mask");
  auto new_axis_mask = 0;
  if (cur->has_attr("new_axis_mask"))
    new_axis_mask = cur->get_attr<std::int32_t>("new_axis_mask");
  auto shrink_axis_mask = 0;
  if (cur->has_attr("shrink_axis_mask"))
    shrink_axis_mask = cur->get_attr<std::int32_t>("shrink_axis_mask");

  // Use bit compares to ensure ellipsis_mask is 0 or a power of 2
  // i.e. there exists only no more than one ellipsis
  if (ellipsis_mask && ((ellipsis_mask & (ellipsis_mask - 1)) != 0)) {
    UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
        << "Multiple ellipses in slice spec not allowed";
  }
  constexpr int32_t kShrinkAxis = -1, kNewAxis = -2;
  // Step 1: Account for ellipsis and new axis
  bool ellipsis_seen = false;
  int32_t num_add_axis_after_ellipsis = 0;
  auto sparse_shape_num = in_shape_num;
  auto sparse_ellipsis_mask = ellipsis_mask;

  for (auto i = 0; i < sparse_shape_num; i++) {
    if (ellipsis_seen && ((1 << i) & new_axis_mask) != 0) {
      num_add_axis_after_ellipsis++;
    }
    if ((1 << i) & ellipsis_mask) {
      ellipsis_seen = true;
    }
  }
  if (!ellipsis_seen) {
    sparse_ellipsis_mask |= (1 << sparse_shape_num);
    sparse_shape_num++;  // this effects loop iteration below
  }
  // Step 2: Make a sparse spec into a full index spec
  int32_t dense_begin_mask = 0;
  int32_t dense_end_mask = 0;
  int32_t dense_shrink_axis_mask = 0;
  bool dense_begin_valid = false;
  bool dense_end_valid = false;
  // build dense spec
  std::vector<int32_t> final_shape_gather_indices;
  {
    auto full_index = 0;
    const int32_t* const strides_flat = strides.data();
    dense_begin_valid = !begin.empty();
    dense_end_valid = !end.empty();
    const int32_t* const begin_flat = begin.empty() ? nullptr : begin.data();
    const int32_t* const end_flat = end.empty() ? nullptr : end.data();

    for (auto i = 0; i < sparse_shape_num; i++) {
      if ((1 << i) & sparse_ellipsis_mask) {
        auto next_index = std::min(in_shape_num - (sparse_shape_num - i) + 1 +
                                       num_add_axis_after_ellipsis,
                                   in_shape_num);
        for (; full_index < next_index; full_index++) {
          dense_begin[full_index] = dense_end[full_index] = 0;
          dense_strides[full_index] = 1;
          dense_begin_mask |= (1 << full_index);
          dense_end_mask |= (1 << full_index);
          final_shape_gather_indices.push_back(full_index);
        }
      } else if ((1 << i) & new_axis_mask) {
        final_shape_gather_indices.push_back(kNewAxis);
      } else {
        if (full_index == (int)dense_begin.size()) {
          UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
              << "Index out of range using input dim " << full_index
              << "; input has only " << in_shape_num << " dims";
        }
        // Gather slicing spec into appropriate index
        if (begin_flat != nullptr) {
          dense_begin[full_index] = begin_flat[i];
        }
        if (end_flat != nullptr) {
          dense_end[full_index] = end_flat[i];
        }
        dense_strides[full_index] = strides_flat[i];
        if (begin_mask & (1 << i)) {
          dense_begin_mask |= (1 << full_index);
        }
        if (end_mask & (1 << i)) {
          dense_end_mask |= (1 << full_index);
        }
        // If shrink, record where to get the dimensionality from (i.e.
        // new_axis creates a fake 1 size dimension. Also remember shrink
        // axis (now in dense form) so we can ignore dense->end below.
        if (shrink_axis_mask & (1 << i)) {
          final_shape_gather_indices.push_back(kShrinkAxis);
          dense_shrink_axis_mask |= (1 << full_index);
        } else {
          final_shape_gather_indices.push_back(full_index);
        }
        full_index++;
      }
    }
  }
  // Step 3: Make implicit ranges (non-zero begin_masks and end_masks)
  // explicit and bounds check!
  std::vector<int32_t> processing_dims;
  for (auto i = 0; i < in_shape_num; ++i) {
    auto& begin_i = dense_begin[i];
    auto& end_i = dense_end[i];
    auto& stride_i = dense_strides[i];
    auto dim_i = in_dims[i];
    if (stride_i == 0) {
      UNI_LOG_FATAL(XIR_INVALID_ARG_OCCUR)
          << cur->to_string() << "'s strides[" << i << "] must be non-zero";
    }
    bool shrink_i = (dense_shrink_axis_mask & (1 << i));
    if (dim_i == -1) {
      processing_dims.push_back(static_cast<int32_t>(shrink_i ? 1 : -1));
      continue;
    }

    const std::array<int64_t, 2> masks = {
        {dense_begin_mask & (1 << i), dense_end_mask & (1 << i)}};
    const std::array<int64_t, 2> valid_range = {
        {stride_i > 0 ? 0 : -1, stride_i > 0 ? dim_i : dim_i - 1}};

    auto canonical = [stride_i, dim_i, masks, valid_range](int64_t x,
                                                           int32_t c) {
      if (masks[c]) {
        return stride_i > 0 ? valid_range[c] : valid_range[(c + 1) & 1];
      } else {
        int64_t x_fwd = x < 0 ? dim_i + x : x;
        return x_fwd < valid_range[0]   ? valid_range[0]
               : x_fwd > valid_range[1] ? valid_range[1]
                                        : x_fwd;
      }
    };
    if (shrink_i && stride_i <= 0) {
      UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
          << "only stride 1 allowed on non-range indexing.";
    }
    const bool begin_and_end_masked =
        (dense_begin_mask & (1 << i)) && (dense_end_mask & (1 << i));
    if (dense_begin_valid && dense_end_valid) {
      if (shrink_i) {
        // If we are shrinking, the end index is now possibly incorrect. In
        // particular foo[-1] produces sparse_begin = -1, sparse_end = 0.
        // and canonical puts these to n-1 and 0, which implies a degenerate
        // interval. Fortunately, it is now safe to re-create end as begin+1.
        int64_t x_fwd = begin_i < 0 ? dim_i + begin_i : begin_i;
        begin_i = x_fwd;
        end_i = begin_i + 1;
        if (x_fwd < 0 || x_fwd >= dim_i) {
          UNI_LOG_ERROR(XIR_INVALID_ARG_OCCUR)
              << "slice index " << begin_i << " of dimension " << i
              << " out of bounds.";
        }
      } else {
        begin_i = canonical(begin_i, 0);
        end_i = canonical(end_i, 1);
      }
    }
    int64_t interval_length;
    bool known_interval = false;
    if (dense_begin_valid && dense_end_valid) {
      interval_length = end_i - begin_i;
      known_interval = true;
    } else if (shrink_i) {
      interval_length = 1;
      known_interval = true;
    } else if (begin_and_end_masked) {
      if (dim_i >= 0) {
        if (stride_i < 0) {
          interval_length = -dim_i;
        } else {
          interval_length = dim_i;
        }
        known_interval = true;
      }
    }
    if (known_interval) {
      int64_t size_i;
      if (interval_length == 0 || ((interval_length < 0) != (stride_i < 0))) {
        size_i = 0;
      } else {
        size_i = interval_length / stride_i +
                 (interval_length % stride_i != 0 ? 1 : 0);
      }
      processing_dims.push_back(static_cast<int32_t>(size_i));
    } else {
      processing_dims.push_back(static_cast<int32_t>(-1));
    }
  }
  for (auto gather_index : final_shape_gather_indices) {
    if (gather_index >= 0) {
      out_shape.push_back(processing_dims[gather_index]);
    } else if (gather_index == kNewAxis) {
      out_shape.push_back(1);
    }
  }
}
}  // namespace xir
