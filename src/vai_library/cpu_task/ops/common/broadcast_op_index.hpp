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
#include <glog/logging.h>

#include <array>
#include <cstdlib>
#include <tuple>
#include <vector>

namespace {  // namespace

struct BroadcastOpIndex {
  BroadcastOpIndex() {}
  BroadcastOpIndex(const std::vector<int32_t>& a,
                   const std::vector<int32_t>& b) {
    std::tie(a_, b_) = normalize({a, b});
    c_ = output_shape(a_, b_);
    num_of_c_ = num_of_elements(c_);
    a_strides_ = get_strides(a_);
    b_strides_ = get_strides(b_);
    index_c_ = 0;
    c_i_ = std::vector<int32_t>(c_.size(), 0);
    index_a_ = 0;
    index_b_ = 0;
    CHECK_GE(c_.size(), 1u);
    CHECK_GE(a_.size(), 1u);
    CHECK_GE(b_.size(), 1u);
  }

 public:
  void tick() {
    index_c_ = index_c_ + 1;
    index_a_ = index_a_ + 1;
    index_b_ = index_b_ + 1;
    if (!is_end()) {
      return;
    }
    //
    auto carry_c = true;
    auto dim = c_.size() - 1;
    do {
      carry_c = increase_carry(c_, c_i_, dim);
      if (!carry_c && a_[dim] == 1) {
        index_a_ = index_a_ - a_strides_[dim + 1];
      }
      if (!carry_c && b_[dim] == 1) {
        index_b_ = index_b_ - b_strides_[dim + 1];
      }
      dim--;
    } while (carry_c);
  }
  bool is_end() { return index_c_ < num_of_c_; }
  int get_num_of_c() { return num_of_elements(c_); }
  int get_a() { return index_a_; }
  int get_b() { return index_b_; }
  int get_c() { return index_c_; }
  void reset_to_zero() {
    index_a_ = 0;
    index_b_ = 0;
    index_c_ = 0;
    c_i_ = std::vector<int32_t>(c_.size(), 0);
  }

 public:
  std::vector<int32_t> a_;
  std::vector<int32_t> b_;
  std::vector<int32_t> c_;
  std::vector<int32_t> a_strides_;
  std::vector<int32_t> b_strides_;
  int index_c_=0;
  int num_of_c_=0;
  int index_a_=0;
  int index_b_=0;
  std::vector<int32_t> c_i_;

 private:
  static int32_t num_of_elements(const std::vector<int32_t>& a) {
    CHECK(!a.empty());
    auto ret = 1;
    for (auto i = 0u; i < a.size(); ++i) {
      ret = ret * a[i];
    }
    return ret;
  }
  static std::vector<int32_t> get_strides(const std::vector<int32_t>& a) {
    auto ret = std::vector<int32_t>(a.size() + 1);
    auto index = a.size();
    ret[index] = 1;
    for (auto i = 0u; i < a.size(); ++i) {
      ret[index - 1] = ret[index] * a[index - 1];
      index = index - 1;
    }
    return ret;
  }
  static std::tuple<std::vector<int32_t>, std::vector<int32_t>> normalize(
      std::array<std::vector<int32_t>, 2> shapes) {
    int longer = shapes[0].size() > shapes[1].size() ? 0 : 1;
    int shorter = shapes[0].size() > shapes[1].size() ? 1 : 0;
    // do not change the longer one.
    if (0) {
      shapes[longer] = shapes[longer];
    }
    auto new_shorter_shape = std::vector<int32_t>(shapes[longer].size(), 1);
    std::copy(shapes[shorter].rbegin(), shapes[shorter].rend(),
              new_shorter_shape.rbegin());
    shapes[shorter] = new_shorter_shape;
    return make_tuple(shapes[0], shapes[1]);
  }

  static std::vector<int32_t> output_shape(std::vector<int32_t> a,
                                           std::vector<int32_t> b) {
    CHECK_EQ(a.size(), b.size());
    auto ret = std::vector<int32_t>();
    ret.reserve(a.size());
    for (auto i = 0u; i < a.size(); ++i) {
      if (a[i] == 1 || b[i] == 1) {
        ret.push_back(a[i] * b[i]);
      } else if (a[i] == b[i]) {
        ret.push_back(a[i]);
      } else {
        return {};
      }
    }
    return ret;
  }
  // index[dim] = index[dim] + 1
  // return carry or not
  static bool increase_carry(const std::vector<int32_t>& shape,
                             std::vector<int32_t>& index, size_t dim) {
    index[dim] = index[dim] + 1;
    CHECK_LE(index[dim], shape[dim]) << " dim= " << dim;
    auto is_carry = index[dim] == shape[dim];
    if (is_carry) {
      index[dim] = 0;
    }
    return is_carry;
  }
};
}  // namespace
// https://numpy.org/doc/stable/user/basics.broadcasting.html
// General Broadcasting Rules
// When operating on two arrays, NumPy compares their shapes element-wise. It
// starts with the trailing (i.e. rightmost) dimensions and works its way left.
// Two dimensions are compatible when

// they are equal, or

// one of them is 1

// If these conditions are not met, a ValueError: operands could not be
// broadcast together exception is thrown, indicating that the arrays have
// incompatible shapes. The size of the resulting array is the size that is not
// 1 along each axis of the inputs.

// Arrays do not need to have the same number of dimensions. For example, if you
// have a 256x256x3 array of RGB values, and you want to scale each color in the
// image by a different value, you can multiply the image by a one-dimensional
// array with 3 values. Lining up the sizes of the trailing axes of these arrays
// according to the broadcast rules, shows that they are compatible:

// Image  (3d array): 256 x 256 x 3
// Scale  (1d array):             3
// Result (3d array): 256 x 256 x 3
// When either of the dimensions compared is one, the other is used. In other
// words, dimensions with size 1 are stretched or “copied” to match the other.

// In the following example, both the A and B arrays have axes with length one
// that are expanded to a larger size during the broadcast operation:

// A      (4d array):  8 x 1 x 6 x 1
// B      (3d array):      7 x 1 x 5
// Result (4d array):  8 x 7 x 6 x 5
// Here are some more examples:

// A      (2d array):  5 x 4
// B      (1d array):      1
// Result (2d array):  5 x 4

// A      (2d array):  5 x 4
// B      (1d array):      4
// Result (2d array):  5 x 4

// A      (3d array):  15 x 3 x 5
// B      (3d array):  15 x 1 x 5
// Result (3d array):  15 x 3 x 5

// A      (3d array):  15 x 3 x 5
// B      (2d array):       3 x 5
// Result (3d array):  15 x 3 x 5

// A      (3d array):  15 x 3 x 5
// B      (2d array):       3 x 1
// Result (3d array):  15 x 3 x 5
// Here are examples of shapes that do not broadcast:

// A      (1d array):  3
// B      (1d array):  4 # trailing dimensions do not match

// A      (2d array):      2 x 1
// B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched
// An example of broadcasting in practice:

// >>> x = np.arange(4)
// >>> xx = x.reshape(4,1)
// >>> y = np.ones(5)
// >>> z = np.ones((3,4))

// >>> x.shape
// (4,)

// >>> y.shape
// (5,)

// >>> x + y
// ValueError: operands could not be broadcast together with shapes (4,) (5,)

// >>> xx.shape
// (4, 1)

// >>> y.shape
// (5,)

// >>> (xx + y).shape
// (4, 5)

// >>> xx + y
// array([[ 1.,  1.,  1.,  1.,  1.],
//        [ 2.,  2.,  2.,  2.,  2.],
//        [ 3.,  3.,  3.,  3.,  3.],
//        [ 4.,  4.,  4.,  4.,  4.]])

// >>> x.shape
// (4,)

// >>> z.shape
// (3, 4)

// >>> (x + z).shape
// (3, 4)

// >>> x + z
// array([[ 1.,  2.,  3.,  4.],
//        [ 1.,  2.,  3.,  4.],
//        [ 1.,  2.,  3.,  4.]])
// Broadcasting provides a convenient way of taking the outer product (or any
// other outer operation) of two arrays. The following example shows an outer
// addition operation of two 1-d arrays:

// >>> a = np.array([0.0, 10.0, 20.0, 30.0])
// >>> b = np.array([1.0, 2.0, 3.0])
// >>> a[:, np.newaxis] + b
// array([[  1.,   2.,   3.],
//        [ 11.,  12.,  13.],
//        [ 21.,  22.,  23.],
//        [ 31.,  32.,  33.]])
// Here the newaxis index operator inserts a new axis into a, making it a
// two-dimensional 4x1 array. Combining the 4x1 array with b, which has shape
// (3,), yields a 4x3 array
