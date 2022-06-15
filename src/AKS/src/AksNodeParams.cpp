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
#include <iostream>

#include "aks/AksNodeParams.h"
#include "src/AksCommonUtils.h"
#include "aks/AksLogger.h"

using namespace AKS;

template <typename ValueType>
static ValueType& get_value(map<string, ValueType> &container, string key) {
  try {
    return container.at(key);
  } catch (const std::out_of_range &oor) {
    LOG_X(FATAL) << "Couldn't find Key : " << key << std::endl;
    throw;
  }
}

#define REGISTER_GET_SET_FIND(DTYPE, CONTAINER)               \
  template<>                                                  \
  DTYPE& NodeParams::getValue<DTYPE>(string key) {            \
    return get_value(CONTAINER, key);                         \
  }                                                           \
                                                              \
  template<>                                                  \
  void NodeParams::setValue<DTYPE>(string key, DTYPE value) { \
    CONTAINER[key] = value;                                   \
  }                                                           \
                                                              \
  template<>                                                  \
  bool NodeParams::hasKey<DTYPE>(string key) {                \
    return CONTAINER.find(key) != CONTAINER.end();            \
  }

REGISTER_GET_SET_FIND(int, _intParams)
REGISTER_GET_SET_FIND(float, _floatParams)
REGISTER_GET_SET_FIND(string, _stringParams)
REGISTER_GET_SET_FIND(vector<int>, _intVectorParams)
REGISTER_GET_SET_FIND(vector<float>, _floatVectorParams)
REGISTER_GET_SET_FIND(vector<string>, _stringVectorParams)

void NodeParams::dump(string prefix)
{
  for (const auto& ppair : _intParams) {
    cout<<prefix<<ppair.first<<":"<<ppair.second<<endl;
  }
  for (const auto& ppair : _floatParams) {
    cout<<prefix<<ppair.first<<":"<<ppair.second<<endl;
  }
  for (const auto& ppair : _stringParams) {
    cout<<prefix<<ppair.first<<":"<<ppair.second<<endl;
  }
  for (const auto& ppair : _intVectorParams) {
    cout<<prefix<<ppair.first<<":";
    for(const auto &val: ppair.second){
      cout<<val<<",";
    }
    cout<<endl;
  }
  for (const auto& ppair : _intVectorParams) {
    cout<<prefix<<ppair.first<<":";
    for(const auto &val: ppair.second){
      cout<<val<<",";
    }
    cout<<endl;
  }
  for (const auto& ppair : _floatVectorParams) {
    cout<<prefix<<ppair.first<<":";
    for(const auto &val: ppair.second){
      cout<<val<<",";
    }
    cout<<endl;
  }
  for (const auto& ppair : _stringVectorParams) {
    cout<<prefix<<ppair.first<<":";
    for(const auto &val: ppair.second){
      cout<<val<<",";
    }
    cout<<endl;
  }
}

void DynamicParamValues::dump(string prefix) {
	NodeParams::dump(prefix);
	for(const auto& imagePath: imagePaths) {
		cout << prefix << "Image : " << imagePath << endl;
	}
}
