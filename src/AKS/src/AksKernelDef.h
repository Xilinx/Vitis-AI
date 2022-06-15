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
#ifndef __AKS_KERNEL_DEF_H_
#define __AKS_KERNEL_DEF_H_

#include <string>
#include <map>
#include <mutex>
#include <chrono>
#include <iostream>
#include <atomic>

using namespace std;

#include "AksCommonDefs.h"
#include <boost/filesystem.hpp>

namespace AKS
{
  struct ParamProps;
  class KernelBase;

  class KernelDef
  {
    public:
      KernelDef(string name,boost::filesystem::path jsonFilePath);
      ~KernelDef(); 
    
      string  getName() const { return _name;}
      void    setKernelType(KernelType ktype) { _kType = ktype;}
      void    setDeviceType(DeviceType dtype) { _dType = dtype;}
      void    setLibPath(boost::filesystem::path p) { _libPath = p; }
      void    setQueueSize(int s) { _queueSize = s;}
      bool    isKernelOK(string &err);
      void    dump(string prefix);
      DeviceType getDeviceType() const { return _dType;}
      int     getQueueSize() const { return _queueSize;}
      void    addParam(string pname, ParamProps *props);
      boost::filesystem::path getLibPath() const { return _libPath;}
      string getLibPathStr() const { return _libPath.string();}
      void setInitFunc(void *funcPtr){ _initFunc = funcPtr;}
      void* getInitFunc() const { return _initFunc;}
      void setKernelHandle(KernelBase *handle){ _kernelHandle = handle;}
      KernelBase* getKernelHandle() const { return _kernelHandle;}
      int         getJsonNumCUs() const { return _numCUs;}
      void        setJsonNumCUs(int num) { _numCUs = num;}
      //Should return a const reference
      ParamProps* getParamProps(string pName);

      void decNumWaitThreads() { --_numThreads; }
      void incNumWaitThreads() { ++_numThreads; }
      void setNumWaitThreads(int16_t n) { _numThreads = n; }
      uint16_t getNumWaitThreads() { return _numThreads.load(); }

      void setKernelAsyncFlag(bool flag) { _isKernelAsyncFlag = flag; }
      bool isKernelAsync() { return _isKernelAsyncFlag; }

      void decAsyncQueueSize() { --_asyncQueueSize; }
      void incAsyncQueueSize() { ++_asyncQueueSize; }
      void setAsyncQueueSize(uint16_t n) { _asyncQueueSize = n; }
      uint16_t getAsyncQueueSize() { return _asyncQueueSize.load(); }

      // Increment & decrement numExisting Jobs
      // Inc : If number of current #jobs = 0, start the timer and increment jobs
      // Dec : decrement #jobs, if it reaches, stop the timer.
      void incNumExistingJobs() {
        std::lock_guard<std::mutex> lk(_mtx_njobs);
        if(_numExistingJobs == 0) {
          _timerStart = std::chrono::steady_clock::now();
        }
        ++_numExistingJobs;
      }

      void decNumExistingJobs() {
        std::lock_guard<std::mutex> lk(_mtx_njobs);
        --_numExistingJobs;
        if(_numExistingJobs == 0) {
          _activeTime += (std::chrono::steady_clock::now() - _timerStart);
        }
      }

      // This to be called after all jobs in the sysMan are done.
      float getKernelActiveTime() {
        std::lock_guard<std::mutex> lk(_mtx_njobs);
        return _activeTime.count();
      }

    private:
      string _name;
      int    _numCUs = 1;//Read from Kernel JSON
      KernelType _kType;
      DeviceType _dType;
      boost::filesystem::path _libPath;
      boost::filesystem::path _jsonFilePath;
      map<string, ParamProps*> _params;
      int _queueSize;

      void         *_initFunc = nullptr;
      KernelBase   *_kernelHandle = nullptr;
      std::atomic<int16_t> _numThreads{MAX_NUM_THREADS};
      bool _isKernelAsyncFlag = false;
      std::atomic<uint16_t> _asyncQueueSize{1};

      // Setup to track total time a kernel was busy. Mainly used for async kernels
      // For blocking kernels, it is directly available from performJob()
      std::mutex _mtx_njobs;
      uint16_t _numExistingJobs = 0;
      std::chrono::time_point<std::chrono::steady_clock> _timerStart;
      std::chrono::duration<float> _activeTime{0};
  };
}

#endif
