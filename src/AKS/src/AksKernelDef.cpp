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
#include "aks/AksKernelBase.h"
#include "AksKernelDef.h"
#include "AksCommonUtils.h"
#include "AksParamProps.h"

using namespace AKS;

#include <iostream>
using namespace std;

KernelDef::KernelDef(string name,boost::filesystem::path jsonFilePath):_name(name),_jsonFilePath(jsonFilePath)
{
  _queueSize = 1;
  _initFunc = nullptr;
}

KernelDef::~KernelDef()
{
  for (const auto& param : _params) delete param.second;
  delete _kernelHandle;
}  

void KernelDef::addParam(string pname, ParamProps *props)
{
  _params[pname] = props;
}

bool KernelDef::isKernelOK(string &err)
{
  //TODO - Check all the fields
  
  //TODO - function implemented but returning true always for now
  return true;
  
  bool ret = false;
  if(_name.empty()){
    //no name
    err = "Kernel doesn't have a name";
  }else if(_libPath.empty()){
    //empty lib path
    err = "No lib path specified";
  }else if(_jsonFilePath.empty()){
    //empty json file path
    err = "No json file path specified";
  }else if(_kType == KernelType::UNKNOWN){
    //unknown type
    err = "Unknown kernel type";
  }else{
    ret = true;
  }
  return ret;
}

void KernelDef::dump(string prefix)
{
  //TODO - check if everything being printed
  cout<<prefix<<"Kernel Name:"<<_name<<endl;
  cout<<prefix<<"  Queue Size:"<<_queueSize<<endl;
  cout<<prefix<<"  Async Queue Size:"<<_asyncQueueSize<<endl;
  cout<<prefix<<"  Kernel Type:"<<CommonUtils::getStrForKernelType(_kType)<<endl;
  cout<<prefix<<"  Device Type:"<<CommonUtils::getStrForDeviceType(_dType)<<endl;
  cout<<prefix<<"  Kernel Lib Path:"<<_libPath.string()<<endl;
  cout<<prefix<<"  Kernel JSON Path:"<<_jsonFilePath.string()<<endl;
  cout<<prefix<<"  is Async :"<<to_string(_isKernelAsyncFlag)<<endl;
  cout<<prefix<<"  # wait threads :"<<_numThreads.load()<<endl;
  if(_initFunc){
    cout<<prefix<<"  Init Func Ptr:"<<_initFunc<<endl;
  }
  /*
   * for (const auto& kv : _supportedOps) {
    cout <<prefix<< "  Op Name:"<<kv.first<<endl;
    if(kv.second){
      kv.second->dump(prefix+"    ");
    }
  }*/
  for (const auto& kv : _params) {
    cout <<prefix<< "  Param Name:"<<kv.first<<endl;
    if(kv.second){
      kv.second->dump(prefix+"    ");
    }
  }
}
ParamProps* KernelDef::getParamProps(string pName)
{
  auto itr = _params.find(pName);
  if(itr != _params.end()){
    return itr->second;
  }
  return nullptr;
}
