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
#ifndef __AKS_PARAM_VALUES_H_
#define __AKS_PARAM_VALUES_H_

#include <map>
#include <string>
#include <vector>

using namespace std;

namespace AKS
{

  /// This is a simple container to store various user args
  /// and pass to each enqueueJob() operation (optional)
  struct UserParams
  {
    std::map<std::string, int> _intParams;
    std::map<std::string, float> _floatParams;
    std::map<std::string, std::string> _stringParams;
    std::map<std::string, std::vector<int> > _intVectorParams;
    std::map<std::string, std::vector<float> > _floatVectorParams;
    std::map<std::string, std::vector<std::string> > _stringVectorParams;
  };

  struct OpParamValues
  {
    OpParamValues() = default;

    map<string,int> _intParams;
    map<string,float> _floatParams;
    map<string,string> _stringParams;
    map<string,vector<int> > _intVectorParams;
    map<string,vector<float> > _floatVectorParams;
    map<string,vector<string> > _stringVectorParams;

    string xclBinPath;

    void dump(string prefix);
  };

  struct InitParams 
  {
    std::string _xclBinPath;
    std::string _kernelName;
    /// Add more if required
  };

  struct DynamicParamValues: OpParamValues
  {
    DynamicParamValues(UserParams* userParams, const string& imagePath)
      :_userParams(userParams), _imagePath(imagePath) {}

    UserParams * _userParams;
    string _imagePath;

    void dump(string prefix);
  };
}

#endif
