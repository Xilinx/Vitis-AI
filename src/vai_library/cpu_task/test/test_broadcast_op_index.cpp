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

#include <cstdlib>
#include <iostream>
#include <tuple>
#include <vector>
using namespace std;
#define UNI_LOG_FATAL(x) std::cout

#include "./broadcast_op_index.hpp"
std::tuple<bool, std::vector<std::int32_t>> size_broadcast(
    const std::vector<std::int32_t>& in_a,
    const std::vector<std::int32_t>& in_b) {
  // new imp
  bool if_success = false;
  std::vector<std::int32_t> ret;
  std::vector<std::int32_t> in_a_local, in_b_local;
  // here make the in_a_local is longer than in_b_local
  if (in_a.size() > in_b.size()) {
    in_a_local = in_a;
    in_b_local = in_b;
  } else {
    in_a_local = in_b;
    in_b_local = in_a;
  }
  auto size_a = in_a_local.size();
  auto size_b = in_b_local.size();
  ret.resize(size_a);
  if (in_a_local == in_b_local) {
    if_success = true;
    ret = in_a_local;
  } else if ((size_a == 0) || (size_b == 0)) {
    if_success = true;
    if (size_a == 0) {
      ret = in_b;
    } else if (size_b == 0) {
      ret = in_a;
    }
  } else {
    std::int32_t idx_a = size_a - 1;
    std::int32_t idx_b = size_b - 1;
    std::int32_t idx_ret = size_a - 1;
    for (; idx_ret >= 0; idx_ret--) {
      if ((idx_a >= 0) && (idx_b >= 0)) {
        auto dim_a = in_a_local[idx_a];
        auto dim_b = in_b_local[idx_b];
        if (dim_a == 1) {
          ret[idx_ret] = dim_b;
          idx_a--;
          idx_b--;
        } else if (dim_b == 1) {
          ret[idx_ret] = dim_a;
          idx_a--;
          idx_b--;
        } else if (dim_a == dim_b) {
          ret[idx_ret] = dim_a;
          idx_a--;
          idx_b--;
        } else {
          break;
        }
      } else if ((idx_a >= 0) && (idx_b < 0)) {
        auto dim_a = in_a_local[idx_a];
        ret[idx_ret] = dim_a;
        idx_a--;
      } else if ((idx_a < 0) && (idx_b >= 0)) {
        break;
      } else {
        abort();
      }
    }
    if_success = (idx_ret < 0);
  }
  return std::make_tuple(if_success, ret);
}

ostream& operator<<(ostream& s, const std::vector<int>& v) {
  s << "[";
  for (auto c = 0u; c < v.size(); ++c) {
    if (c != 0) {
      s << ",";
    }
    s << v[c];
  }
  s << "]";
  return s;
}

int test(std::vector<int32_t> a, std::vector<int32_t> b) {
  vector<int> c;
  bool ok;
  tie(ok, c) = size_broadcast(a, b);

  cout << "ok = " << ok << " \n"  //
       << " a=" << a << " \n"     //
       << " b=" << b << " \n"     //
       << " c=" << c << " \n"     //
       << endl;
  std::vector<int32_t> a1;
  std::vector<int32_t> b1;
  auto op_index = BroadcastOpIndex(a, b);
  cout << " a1=" << op_index.a_ << " \n"          //
       << " b1=" << op_index.b_ << " \n"          //
       << " c1=" << op_index.c_ << " \n"          //
       << " c1=" << op_index.a_strides_ << " \n"  //
       << endl;
  int i = 0;
  for (; op_index.is_end(); op_index.tick()) {
    auto index_a = op_index.get_a();
    auto index_b = op_index.get_b();
    auto index_c = op_index.get_c();
    cout << "i=" << i << ": c[" << index_c << "] = a[" << index_a << "] + b["
         << index_b << "]" << endl;
    ;
    cout << "                                                  debug="
         << "c=" << op_index.c_i_ << ","
         << "a=" << op_index.a_i_ << ","  //
         << "b=" << op_index.b_i_ << ","  //
         << endl;
    i = i + 1;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  test({2, 1, 3, 1}, {2, 1, 3});
  /*  test({5, 4}, {1});

    test({15, 3, 5}, {3, 5});
    test({15, 3, 2}, {3, 5});*/
  return 0;
}
