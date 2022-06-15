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
#include "AksCommonUtils.h"
#include <iostream>
#include <cstdlib>

namespace AKS
{

KernelType CommonUtils::getKernelTypeForStr(string str)
{
  AKS::KernelType ret = AKS::KernelType::UNKNOWN;
  if(str == "hls"){
    ret = AKS::KernelType::HLS;
  }else if (str == "python"){
    ret = AKS::KernelType::PYTHON;
  }else if (str == "cpp"){
    ret = AKS::KernelType::CPP;
  }else if (str == "caffe"){
    ret = AKS::KernelType::CAFFE;
  }else if (str == "dnn"){
    ret = AKS::KernelType::DNN;
  }else if (str == "tf"){
    ret = AKS::KernelType::TF;
  }
  return ret;
}

DeviceType CommonUtils::getDeviceTypeForStr(string str)
{
  AKS::DeviceType ret = AKS::DeviceType::UNKNOWN;
  if(str == "cpu"){
    ret = AKS::DeviceType::CPU;
  }else if (str == "fpga"){
    ret = AKS::DeviceType::FPGA;
  }
  return ret;
}

string CommonUtils::getStrForKernelParamType(KernelParamType ptype)
{
  string ret = "";
  switch(ptype)
  {
    case KernelParamType::INT: ret = "int"; break;
    case KernelParamType::INT_ARR: ret = "int_array"; break;
    case KernelParamType::FLOAT: ret = "float"; break;
    case KernelParamType::FLOAT_ARR: ret = "float_array"; break;
    case KernelParamType::STRING: ret = "string"; break;
    case KernelParamType::STRING_ARR: ret = "string_array"; break;
    default: ret = "unknown"; break;
  }
  return ret;
}

string CommonUtils::getStrForKernelType(KernelType type)
{
  string ret = "";
  switch(type)
  {
    case KernelType::HLS: ret = "hls"; break;
    case KernelType::DNN: ret = "dnn"; break;
    case KernelType::CAFFE: ret = "caffe"; break;
    case KernelType::TF: ret = "tf"; break;
    case KernelType::CPP: ret = "cpp"; break;
    case KernelType::PYTHON: ret = "python"; break;
    default: ret = "unknown"; break;
  }
  return ret;
}

string CommonUtils::getStrForDeviceType(DeviceType type)
{
  string ret = "";
  switch(type)
  {
    case DeviceType::CPU: ret = "cpu"; break;
    case DeviceType::FPGA: ret = "fpga"; break;
    default: ret = "unknown"; break;
  }
  return ret;
}

KernelParamType CommonUtils::getKernelParamTypeForStr(string str)
{
  AKS::KernelParamType ret = AKS::KernelParamType::UNKNOWN;
  if(str == "int"){
    ret = AKS::KernelParamType::INT;
  }else if (str == "int_array"){
    ret = AKS::KernelParamType::INT_ARR;
  }else if (str == "float"){
    ret = AKS::KernelParamType::FLOAT;
  }else if (str == "float_array"){
    ret = AKS::KernelParamType::FLOAT_ARR;
  }else if (str == "string"){
    ret = AKS::KernelParamType::STRING;
  }else if (str == "string_array"){
    ret = AKS::KernelParamType::STRING_ARR;
  }
  return ret;
}

bool CommonUtils::getKernelParamOptionalForStr(string str)
{
  bool ret = false;
  if(str == "true"){
    ret = true;
  }else if(str == "false"){
    ret = false;
  }else{
    cout<<"Unknown value for optional:"<<str<<". Defaulting to false"<<endl;
  }
  return ret;
}

} // AKS
