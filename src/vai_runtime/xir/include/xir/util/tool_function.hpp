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
#include <algorithm>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>

#include "xir/op/op.hpp"

namespace xir {

/**
 * @brief Get the md5sum of a buff.
 *
 * @param buf The buffer base address.
 *
 * @param size The buffer's size, in bytes.
 *
 * @return A string contains the md5sum in hex format.
 */
XIR_DLLESPEC const std::string get_md5_of_buffer(const void* buf, size_t size);

/**
 * @brief Get the md5sum of a file.
 *
 * @param filepath The path of the input file.
 *
 * @return A string contains the md5sum in hex format.
 */
XIR_DLLESPEC const std::string get_md5_of_file(const std::string& filepath);

/**
 * @brief Get the xir-lib name.
 *
 * @return A string of the xir-lib name
 */
XIR_DLLESPEC const std::string get_lib_name();

/**
 * @brief Get the xir-lib version id.
 *
 * @return A string of the xir-lib id.
 */
XIR_DLLESPEC const std::string get_lib_id();

// template helper
namespace th {

template <typename T>
struct is_std_vector : public std::false_type {};
template <typename T>
struct is_std_vector<std::vector<T>> : public std::true_type {};

template <typename T>
struct is_std_set : public std::false_type {};
template <typename T>
struct is_std_set<std::set<T>> : std::true_type {};

template <bool head, bool... tail>
struct var_and : public std::false_type {};
template <bool... tail>
struct var_and<true, tail...> {
  static constexpr bool value = var_and<tail...>::value;
};
template <>
struct var_and<true> : public std::true_type {};

}  // namespace th

// vector related
// set related
/**
 * @brief Convert a vector/set to a string for visualization.
 *
 * @param content The input vector/set.
 *
 * @param delimiter The delimiter to seperate the elements of vector/set,
 * default value is ",".
 *
 * @param left_bracket The symbol as the left bracket, default value is "{".
 *
 * @param right_bracket The symbol as the right bracket, default value is "}".
 *
 * @return A string of the visualized vector/set.
 */
template <typename T, typename std::enable_if<th::is_std_set<T>::value ||
                                                  th::is_std_vector<T>::value,
                                              bool>::type = true>
std::string to_string(const T& content, const std::string& delimiter = ",",
                      const std::string& left_bracket = "{",
                      const std::string& right_bracket = "}") {
  std::ostringstream outstring;
  outstring << left_bracket;
  if (!content.empty()) {
    std::copy(content.begin(), std::prev(content.end()),
              std::ostream_iterator<typename T::value_type>(
                  outstring, (delimiter + " ").c_str()));
    outstring << *(content.rbegin());
  }
  outstring << right_bracket;
  auto ret = outstring.str();
  return ret;
}

// name related
/**
 * @brief A helper function to add a name prefix as the xir style.
 *
 * @param name The name.
 *
 * @param prefix The prefix.
 */
XIR_DLLESPEC void add_prefix_helper(std::string& name,
                                    const std::string& prefix);

/**
 * @brief A helper function to add a name suffix as the xir style.
 *
 * @param name The name.
 *
 * @param suffix The suffix.
 */
XIR_DLLESPEC void add_suffix_helper(std::string& name,
                                    const std::string& suffix);

/**
 * @brief Add a serial of prefixs for a name in the xir style.
 *
 * @param name The name.
 *
 * @param prefixs A serial of prefix.
 *
 * @return The name after adding all the prefixs.
 */
template <typename... Args>
typename std::enable_if<
    th::var_and<std::is_constructible<std::string, Args>::value...>::value,
    std::string>::type
add_prefix(const std::string& name, const Args&... prefixs) {
  std::vector<std::string> prefixs_vec{prefixs...};
  std::string ret = name;
  for (auto prefix = prefixs_vec.rbegin(); prefix != prefixs_vec.rend();
       prefix++) {
    add_prefix_helper(ret, *prefix);
  }
  return ret;
}

/**
 * @brief Add a serial of suffixs for a name in the xir style.
 *
 * @param name The name.
 *
 * @param prefixs A serial of suffix.
 *
 * @return The name after adding all the suffixs.
 */
template <typename... Args>
typename std::enable_if<
    th::var_and<std::is_constructible<std::string, Args>::value...>::value,
    std::string>::type
add_suffix(const std::string& name, const Args&... suffixs) {
  std::vector<std::string> suffixs_vec{suffixs...};
  std::string ret = name;
  for (auto suffix : suffixs_vec) {
    add_suffix_helper(ret, suffix);
  }
  return ret;
}

/**
 * @brief Remove all the xir style prefix and suffix, and get the original
 * name.
 *
 * @param name The name.
 *
 * @return The original name.
 */
XIR_DLLESPEC std::string remove_xfix(const std::string& name);

/**
 * @brief Extract all the prefix, suffix and the original name.
 *
 * @param name The name.
 *
 * @return A vector of prefix, suffix and original name, and the original name
 * is at the back of the vector.
 */
std::vector<std::string> extract_xfix(const std::string& name);

// math related
/**
 * @brief Round the input float data.
 *
 * @param data The input data.
 *
 * @param round_mode Then rounding mode.
 *
 * @return The result in float.
 */
float xround(const float& data, const std::string& round_mode = "STD_ROUND");

void register_customized_operator_definition(const std::string& name,
                                             const std::string& type);
XIR_DLLESPEC std::vector<float> get_float_vec_from_any(const xir::any& any);

/**
 * @brief Tensor lexicographical order sort function
 * name.
 *
 * @param name The name.
 *
 * @return The original name.
 */
XIR_DLLESPEC bool TensorLexicographicalOrder(Tensor* a, Tensor* b);

/**
 * @brief Strided Slice Op standardization function.
 * 
 * @ref tensorflow\core\util\strided_slice_op.cc
 * 
 * @param the strided_slice op, 
 *        begin, end, strides are 1-D vectors with the length of input dimension.
 *        out_shape is an empty out_shape.
 * 
 * @return begin, end, strides, and output shape after standardization.
 */
XIR_DLLESPEC void validate_strided_slice(const xir::Op* op_strided_slice,
                                         std::vector<int32_t>& begin,
                                         std::vector<int32_t>& end,
                                         std::vector<int32_t>& strides,
                                         std::vector<int32_t>& out_shape);
}  // namespace xir
