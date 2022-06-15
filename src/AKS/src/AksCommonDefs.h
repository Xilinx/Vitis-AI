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
#ifndef __AKS_COMMON_DEFS_H_
#define __AKS_COMMON_DEFS_H_

#include <string>
using namespace std;

#define _DEBUG_ 0
#define _TRACE_ 0
#define MAX_NUM_THREADS 1024
#define LOGSIZE 0
#if _TRACE_
  #define LOGSIZE 1024
#endif

namespace AKS
{
  enum class KernelType
  {
    HLS,
    DNN,
    CAFFE,
    TF,
    CPP,
    PYTHON,
    UNKNOWN
  };

  enum class DeviceType
  {
    CPU,
    FPGA,
    UNKNOWN
  };

  enum class KernelParamType
  {
    INT,
    INT_ARR,
    FLOAT,
    FLOAT_ARR,
    STRING,
    STRING_ARR,
    UNKNOWN
  }; 


}

#endif
