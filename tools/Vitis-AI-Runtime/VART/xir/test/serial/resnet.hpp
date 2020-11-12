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

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace std;

namespace xir {

class Op;
class Attrs;
class GraphImp;

class Resnet {
 public:
  using InputOpsMap = std::map<std::string, std::vector<Op*>>;

 private:
  vector<function<std::unique_ptr<Attrs>()>> op_attrs_;

 public:
  Resnet();
  ~Resnet() = default;

 public:
  unique_ptr<GraphImp> Build();

  template <typename T>
  T Random(T low, T high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(low, high);

    return dis(gen);
  }

  template <typename T>
  void Random(T* data, uint64_t size, T low, T high, uint64_t seed) {
    std::random_device rd;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<T> dis(low, high);

    for (auto i = 0U; i < size; i++) {
      if (seed == 0)
        data[i] = 0;
      else
        data[i] = dis(gen);
    }
  }

 private:
  InputOpsMap get_iom(const unique_ptr<GraphImp>& g, int layer_id);
  InputOpsMap get_const_iom(const unique_ptr<GraphImp>& g, int layer_id);
  InputOpsMap get_data_iom(const unique_ptr<GraphImp>& g, int layer_id);
  InputOpsMap get_conv_iom(const unique_ptr<GraphImp>& g, int layer_id);
  InputOpsMap get_relu_iom(const unique_ptr<GraphImp>& g, int layer_id);
  InputOpsMap get_elew_iom(const unique_ptr<GraphImp>& g, int layer_id);
  InputOpsMap get_maxpool_iom(const unique_ptr<GraphImp>& g, int layer_id);

 private:
  enum LayerEnum {
    RES5C_BRANCH2C_BIAS,
    RES5C_BRANCH2C_WEIGHTS,
    RES5C_BRANCH2B_BIAS,
    RES5C_BRANCH2B_WEIGHTS,
    RES5C_BRANCH2A_BIAS,
    RES5C_BRANCH2A_WEIGHTS,
    RES5B_BRANCH2C_BIAS,
    RES5B_BRANCH2C_WEIGHTS,
    RES5B_BRANCH2B_BIAS,
    RES5B_BRANCH2B_WEIGHTS,
    RES5B_BRANCH2A_BIAS,
    RES5B_BRANCH2A_WEIGHTS,
    RES5A_BRANCH2C_BIAS,
    RES5A_BRANCH2C_WEIGHTS,
    RES5A_BRANCH2B_BIAS,
    RES5A_BRANCH2B_WEIGHTS,
    RES5A_BRANCH2A_BIAS,
    RES5A_BRANCH2A_WEIGHTS,
    RES5A_BRANCH1_BIAS,
    RES5A_BRANCH1_WEIGHTS,
    RES4F_BRANCH2C_BIAS,
    RES4F_BRANCH2C_WEIGHTS,
    RES4F_BRANCH2B_BIAS,
    RES4F_BRANCH2B_WEIGHTS,
    RES4F_BRANCH2A_BIAS,
    RES4F_BRANCH2A_WEIGHTS,
    RES4E_BRANCH2C_BIAS,
    RES4E_BRANCH2C_WEIGHTS,
    RES4E_BRANCH2B_BIAS,
    RES4E_BRANCH2B_WEIGHTS,
    RES4E_BRANCH2A_BIAS,
    RES4E_BRANCH2A_WEIGHTS,
    RES4D_BRANCH2C_BIAS,
    RES4D_BRANCH2C_WEIGHTS,
    RES4D_BRANCH2B_BIAS,
    RES4D_BRANCH2B_WEIGHTS,
    RES4D_BRANCH2A_BIAS,
    RES4D_BRANCH2A_WEIGHTS,
    RES4C_BRANCH2C_BIAS,
    RES4C_BRANCH2C_WEIGHTS,
    RES4C_BRANCH2B_BIAS,
    RES4C_BRANCH2B_WEIGHTS,
    RES4C_BRANCH2A_BIAS,
    RES4C_BRANCH2A_WEIGHTS,
    RES4B_BRANCH2C_BIAS,
    RES4B_BRANCH2C_WEIGHTS,
    RES4B_BRANCH2B_BIAS,
    RES4B_BRANCH2B_WEIGHTS,
    RES4B_BRANCH2A_BIAS,
    RES4B_BRANCH2A_WEIGHTS,
    RES4A_BRANCH2C_BIAS,
    RES4A_BRANCH2C_WEIGHTS,
    RES4A_BRANCH2B_BIAS,
    RES4A_BRANCH2B_WEIGHTS,
    RES4A_BRANCH2A_BIAS,
    RES4A_BRANCH2A_WEIGHTS,
    RES4A_BRANCH1_BIAS,
    RES4A_BRANCH1_WEIGHTS,
    RES3D_BRANCH2C_BIAS,
    RES3D_BRANCH2C_WEIGHTS,
    RES3D_BRANCH2B_BIAS,
    RES3D_BRANCH2B_WEIGHTS,
    RES3D_BRANCH2A_BIAS,
    RES3D_BRANCH2A_WEIGHTS,
    RES3C_BRANCH2C_BIAS,
    RES3C_BRANCH2C_WEIGHTS,
    RES3C_BRANCH2B_BIAS,
    RES3C_BRANCH2B_WEIGHTS,
    RES3C_BRANCH2A_BIAS,
    RES3C_BRANCH2A_WEIGHTS,
    RES3B_BRANCH2C_BIAS,
    RES3B_BRANCH2C_WEIGHTS,
    RES3B_BRANCH2B_BIAS,
    RES3B_BRANCH2B_WEIGHTS,
    RES3B_BRANCH2A_BIAS,
    RES3B_BRANCH2A_WEIGHTS,
    RES3A_BRANCH2C_BIAS,
    RES3A_BRANCH2C_WEIGHTS,
    RES3A_BRANCH2B_BIAS,
    RES3A_BRANCH2B_WEIGHTS,
    RES3A_BRANCH2A_BIAS,
    RES3A_BRANCH2A_WEIGHTS,
    RES3A_BRANCH1_BIAS,
    RES3A_BRANCH1_WEIGHTS,
    RES2C_BRANCH2C_BIAS,
    RES2C_BRANCH2C_WEIGHTS,
    RES2C_BRANCH2B_BIAS,
    RES2C_BRANCH2B_WEIGHTS,
    RES2C_BRANCH2A_BIAS,
    RES2C_BRANCH2A_WEIGHTS,
    RES2B_BRANCH2C_BIAS,
    RES2B_BRANCH2C_WEIGHTS,
    RES2B_BRANCH2B_BIAS,
    RES2B_BRANCH2B_WEIGHTS,
    RES2B_BRANCH2A_BIAS,
    RES2B_BRANCH2A_WEIGHTS,
    RES2A_BRANCH2C_BIAS,
    RES2A_BRANCH2C_WEIGHTS,
    RES2A_BRANCH2B_BIAS,
    RES2A_BRANCH2B_WEIGHTS,
    RES2A_BRANCH2A_BIAS,
    RES2A_BRANCH2A_WEIGHTS,
    RES2A_BRANCH1_BIAS,
    RES2A_BRANCH1_WEIGHTS,
    CONV1_BIAS,
    CONV1_WEIGHTS,
    CONV1_INPUT,
    CONV1,
    CONV1_RELU,
    POOL1,
    RES2A_BRANCH2A,
    RES2A_BRANCH2A_RELU,
    RES2A_BRANCH2B,
    RES2A_BRANCH2B_RELU,
    RES2A_BRANCH2C,
    RES2A_BRANCH1,
    RES2A,
    RES2A_RELU,
    RES2B_BRANCH2A,
    RES2B_BRANCH2A_RELU,
    RES2B_BRANCH2B,
    RES2B_BRANCH2B_RELU,
    RES2B_BRANCH2C,
    RES2B,
    RES2B_RELU,
    RES2C_BRANCH2A,
    RES2C_BRANCH2A_RELU,
    RES2C_BRANCH2B,
    RES2C_BRANCH2B_RELU,
    RES2C_BRANCH2C,
    RES2C,
    RES2C_RELU,
    RES3A_BRANCH2A,
    RES3A_BRANCH2A_RELU,
    RES3A_BRANCH2B,
    RES3A_BRANCH2B_RELU,
    RES3A_BRANCH2C,
    RES3A_BRANCH1,
    RES3A,
    RES3A_RELU,
    RES3B_BRANCH2A,
    RES3B_BRANCH2A_RELU,
    RES3B_BRANCH2B,
    RES3B_BRANCH2B_RELU,
    RES3B_BRANCH2C,
    RES3B,
    RES3B_RELU,
    RES3C_BRANCH2A,
    RES3C_BRANCH2A_RELU,
    RES3C_BRANCH2B,
    RES3C_BRANCH2B_RELU,
    RES3C_BRANCH2C,
    RES3C,
    RES3C_RELU,
    RES3D_BRANCH2A,
    RES3D_BRANCH2A_RELU,
    RES3D_BRANCH2B,
    RES3D_BRANCH2B_RELU,
    RES3D_BRANCH2C,
    RES3D,
    RES3D_RELU,
    RES4A_BRANCH2A,
    RES4A_BRANCH2A_RELU,
    RES4A_BRANCH2B,
    RES4A_BRANCH2B_RELU,
    RES4A_BRANCH2C,
    RES4A_BRANCH1,
    RES4A,
    RES4A_RELU,
    RES4B_BRANCH2A,
    RES4B_BRANCH2A_RELU,
    RES4B_BRANCH2B,
    RES4B_BRANCH2B_RELU,
    RES4B_BRANCH2C,
    RES4B,
    RES4B_RELU,
    RES4C_BRANCH2A,
    RES4C_BRANCH2A_RELU,
    RES4C_BRANCH2B,
    RES4C_BRANCH2B_RELU,
    RES4C_BRANCH2C,
    RES4C,
    RES4C_RELU,
    RES4D_BRANCH2A,
    RES4D_BRANCH2A_RELU,
    RES4D_BRANCH2B,
    RES4D_BRANCH2B_RELU,
    RES4D_BRANCH2C,
    RES4D,
    RES4D_RELU,
    RES4E_BRANCH2A,
    RES4E_BRANCH2A_RELU,
    RES4E_BRANCH2B,
    RES4E_BRANCH2B_RELU,
    RES4E_BRANCH2C,
    RES4E,
    RES4E_RELU,
    RES4F_BRANCH2A,
    RES4F_BRANCH2A_RELU,
    RES4F_BRANCH2B,
    RES4F_BRANCH2B_RELU,
    RES4F_BRANCH2C,
    RES4F,
    RES4F_RELU,
    RES5A_BRANCH2A,
    RES5A_BRANCH2A_RELU,
    RES5A_BRANCH2B,
    RES5A_BRANCH2B_RELU,
    RES5A_BRANCH2C,
    RES5A_BRANCH1,
    RES5A,
    RES5A_RELU,
    RES5B_BRANCH2A,
    RES5B_BRANCH2A_RELU,
    RES5B_BRANCH2B,
    RES5B_BRANCH2B_RELU,
    RES5B_BRANCH2C,
    RES5B,
    RES5B_RELU,
    RES5C_BRANCH2A,
    RES5C_BRANCH2A_RELU,
    RES5C_BRANCH2B,
    RES5C_BRANCH2B_RELU,
    RES5C_BRANCH2C,
    RES5C,
    RES5C_RELU,
    LAYER_NUM
  };

  enum LayerTypeEnum {
    LAYER_TYPE_CONST,
    LAYER_TYPE_DATA,
    LAYER_TYPE_CONV,
    LAYER_TYPE_RELU,
    LAYER_TYPE_MAXPOOL,
    LAYER_TYPE_ELEW,
    LAYER_TYPE_NUM
  };
  const static vector<int> LayerType;
  const static vector<string> LayerName;
  const static vector<string> LayerTypeName;
  const static vector<vector<string>> LayerInputOpsName;
};

}  // namespace xir
