/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __AKS_NODE_PARAMS_H_
#define __AKS_NODE_PARAMS_H_

#include <map>
#include <string>
#include <vector>

using namespace std;

namespace AKS
{

  /// This is a simple container to store multiple args
  /// It can be used pass user args to each enqueueJob() operation (optional)
  /// All node parameters in graph.json also passed to Kernel through this
  // TODO : Once migrated to C++17, use std::any to improve the implementation
  struct NodeParams
  {
    std::map<std::string, int> _intParams;
    std::map<std::string, float> _floatParams;
    std::map<std::string, std::string> _stringParams;
    std::map<std::string, std::vector<int> > _intVectorParams;
    std::map<std::string, std::vector<float> > _floatVectorParams;
    std::map<std::string, std::vector<std::string> > _stringVectorParams;
    void dump(string prefix);

    // Get a value from NodeParams.
    // Throws exception if value is not found
    template<typename T> T& getValue(string key);

    // Set a value to NodeParams.
    template<typename T> void setValue(string key, T value);

    // Check if a key corresponding to a value of type T is present in NodeParams
    template<typename T> bool hasKey(string key);
  };

  struct DynamicParamValues : public NodeParams
  {
    DynamicParamValues(NodeParams userParams, std::vector<string> _imagePaths)
      :NodeParams(std::move(userParams)), imagePaths(std::move(_imagePaths)) {}

    std::vector<string> imagePaths;
    void dump(string prefix);
  };

}

#endif
