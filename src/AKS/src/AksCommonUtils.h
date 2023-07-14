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
#ifndef __AKS_COMMON_UTILS_H_
#define __AKS_COMMON_UTILS_H_

#include "AksCommonDefs.h"
#include <string>
using namespace std;

namespace AKS
{
  class CommonUtils
  {
    public:
       static KernelType getKernelTypeForStr(string str);
       static string getStrForKernelType(KernelType type);
       static DeviceType getDeviceTypeForStr(string str);
       static string getStrForDeviceType(DeviceType type);
       static KernelParamType getKernelParamTypeForStr(string str);
       static string getStrForKernelParamType(KernelParamType ptype);
       static bool getKernelParamOptionalForStr(string str);
  };
}

#endif
