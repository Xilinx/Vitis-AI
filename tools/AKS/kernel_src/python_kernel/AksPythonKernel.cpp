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
// Python Kernel Implementation
// TODO : Make it thread-safe

#include <iostream>
#include <vector>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <aks/AksKernelBase.h>
#include <aks/AksNodeParams.h>
#include <aks/AksDataDescriptor.h>
#include <aks/AksLogger.h>
#include <AksEmbedPy.h>

using namespace AKS;

#pragma GCC visibility push(hidden)
struct PythonParams {
  py::module module;
  py::object kernel;
  py::object exec_async, wait, report;
  bool isExecAsync = false;
};
#pragma GCC visibility pop

// Convert vector<string> to Python Dict-style string
// Eg : vector['a':2, 'b':4] --> "{'a':2, 'b':4}"
std::string formatStrListToStrDict(std::vector<std::string>& strList) {
  std::string strDict = "";
  strDict += "{";
  for(auto&& item: strList) {
    strDict += item;
    strDict += ", ";
  }
  strDict += "}";
  return strDict;
}

#pragma GCC visibility push(hidden)
class PythonKernelBase: public AKS::KernelBase {
  public:
    PythonKernelBase();
    int exec_async (
        std::vector<AKS::DataDescriptor *> &in,
        std::vector<AKS::DataDescriptor *> &out,
        AKS::NodeParams* params,
        AKS::DynamicParamValues* dynParams);
    void nodeInit(AKS::NodeParams*);
    bool isExecAsync() { return false; }
    int getNumCUs() { return 1; }
    void report(AKS::NodeParams* nodeParams);
    ~PythonKernelBase();
  private:
    std::map<AKS::NodeParams*, PythonParams*> _opParamDict;
    void setData(AKS::NodeParams* nodeParams, PythonParams* params);
    PythonParams* getData(AKS::NodeParams *nodeParams);
    bool isInterpreterInitialized = false;
    py::gil_scoped_release* release = nullptr;
};
#pragma GCC visibility pop

extern "C" {
  AKS::KernelBase* getKernel (AKS::NodeParams* params) {
    return new PythonKernelBase();
  }
} // extern C

PythonKernelBase::PythonKernelBase() {
  // If app is running from C++, we invoke our own Python interpreter
  if(!Py_IsInitialized()) {
    LOG(DEBUG) << "Python Kernel : Initializing Python Interpreter " << std::endl;
    py::initialize_interpreter();
    isInterpreterInitialized = true;
  }

  // At this stage, Python should be up & running, either
  // via app is running inside Python, OR
  // Python is started by above block
  if(Py_IsInitialized()) {
    {
      // For Python Ext, this is a double acquisition
      py::gil_scoped_acquire acq;
      // TODO : importing threading module to avoid a crash.
      auto np = py::module::import("threading");
    }
    if(isInterpreterInitialized) release = new py::gil_scoped_release();
  } else {
    LOG(ERROR) << "Python Kernel : Couldn't Initialize Python Interpreter" << std::endl;
  }
}

PythonKernelBase::~PythonKernelBase() {
  // Acquire the GIL
  if(isInterpreterInitialized && release) delete release;

  // Clear all Python objects before closing interpreter
  {
    // For Python Ext, this is a double acquisition
    py::gil_scoped_acquire acq;
    for(auto& item: _opParamDict) {
      delete item.second;
    }
  }

  // If we invoked our own Python, it is our duty to clean it up too.
  if(isInterpreterInitialized) {
    LOG(DEBUG) << "Python Kernel : Closing Python Interpreter" << std::endl;
    py::finalize_interpreter();
    isInterpreterInitialized = false;
  }
}

void PythonKernelBase::setData(AKS::NodeParams* nodeParams, PythonParams* params)
{
  _opParamDict[nodeParams] = params;
}

PythonParams* PythonKernelBase::getData(AKS::NodeParams *nodeParams)
{
  auto itr = _opParamDict.find(nodeParams);
  if(itr != _opParamDict.end()) return itr->second;
  return nullptr;
}

// Read the op params and load pymodule
// TODO : Dynamically add path to sys.path
// py::module::import("sys").attr("path").cast<py::list>().append(".");
void PythonKernelBase::nodeInit(AKS::NodeParams* params) {
  PythonParams* pyparams = getData(params);
  if(!pyparams) {
    auto module_name = params->getValue<string>("module");
    auto kernel_name = params->getValue<string>("kernel");
    auto isExecAsync = 0; 
    // TODO : Currently, async python kernels are not supported
    // auto isExecAsync = params->_intParams.find("isExecAsync") != params->_intParams.end() ?
    //                    params->getValue<int>("isExecAsync") : 0;

    py::gil_scoped_acquire gil;
    LOG(DEBUG) << "Loading Python Kernel : " << module_name << " " << kernel_name << std::endl;
    pyparams = new PythonParams;

    // convert pyargs string_array to pyargs dict
    // TODO : Delete pyargs from _vectorString (optional)
    auto pyargs = params->getValue<std::vector<std::string>>("pyargs");
    auto strDict = formatStrListToStrDict(pyargs);
    params->setValue<std::string>("pyargs", strDict);

    pyparams->module = py::module::import(module_name.c_str());
    pyparams->kernel = pyparams->module.attr(kernel_name.c_str())(params);
    pyparams->exec_async = pyparams->kernel.attr("exec_async");
    pyparams->wait = pyparams->kernel.attr("wait");
    pyparams->report = pyparams->kernel.attr("report");
    pyparams->isExecAsync = isExecAsync;

    setData(params, pyparams);
  }
}

int PythonKernelBase::exec_async (
  vector<AKS::DataDescriptor *>& in, vector<AKS::DataDescriptor *>& out,
  AKS::NodeParams* params, AKS::DynamicParamValues* dynParams)
{
  PythonParams* pyparams = getData(params);
  py::gil_scoped_acquire gil;
  auto res = pyparams->exec_async(in, params, dynParams);
  out = res.cast<std::vector<AKS::DataDescriptor*>>();
  return 0;
}

void PythonKernelBase::report(AKS::NodeParams* nodeParams) {
  PythonParams* pyparams = getData(nodeParams);
  py::gil_scoped_acquire gil;
  pyparams->report(nodeParams);
}
