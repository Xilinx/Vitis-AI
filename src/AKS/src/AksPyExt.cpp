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
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "aks/AksNodeParams.h"
#include "aks/AksSysManagerExt.h"
#include "aks/AksDataDescriptor.h"
#include "src/AksAIGraph.h"

namespace py = pybind11;
using namespace AKS;

// type caster: DD <-> NumPy-array
namespace pybind11 { namespace detail {
  template <> struct type_caster<AKS::DataDescriptor>
  {
    public:
      PYBIND11_TYPE_CASTER(AKS::DataDescriptor, _("DataDescriptor"));

      // Conversion part 1 (Python -> C++)
      bool load(py::handle src, bool convert)
      {
        if (!convert && !py::array_t<float>::check_(src))
          return false;

        auto buf    = py::array_t<float>::ensure(src);
        if (!buf)
          return false;

        auto dims   = buf.ndim();
        if (dims < 1)
          return false;

        std::vector<int> shape(buf.ndim());

        for ( int i = 0 ; i<buf.ndim() ; i++ )
          shape[i]  = buf.shape()[i];

        value       = std::move(AKS::DataDescriptor(shape,DataType::FLOAT32));
        memcpy(value.data(), buf.data(), value.getNumberOfElements() * sizeof(float));
        return true;
      }

      //Conversion part 2 (C++ -> Python)
      static py::handle cast(const AKS::DataDescriptor& src,
        py::return_value_policy policy, py::handle parent)
      {
        py::array a(src.getShape(), src.getStride(), (float*)src.const_data<float>() );
        return a.release();
      }
  };
}}


PYBIND11_MODULE(libaks, m) {
  m.doc() = "Python extension for AKS";

  py::class_<AIGraph>(m, "AIGraph");

  py::class_<SysManagerExt, std::unique_ptr<SysManagerExt, py::nodelete>>(m, "SysManager")
    .def(py::init([](){ return SysManagerExt::getGlobal(); }),
        py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>())
    .def("loadKernels", &SysManagerExt::loadKernels, "Loads all the kernels in the system",
        py::call_guard<py::gil_scoped_release>())
    .def("loadGraphs", &SysManagerExt::loadGraphs, "Loads the graph for execution",
        py::call_guard<py::gil_scoped_release>())
    .def("getGraph", &SysManagerExt::getGraph, "Returns a handle to the graph with given name",
        py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>())
    .def("enqueueJob",
        (void(SysManagerExt::*)(AIGraph*, const std::string&))&SysManagerExt::pyEnqueueJob,
        "Enqueue a job to SysManager",
        py::arg("graph"), py::arg("filePath"),
        py::call_guard<py::gil_scoped_release>())
    .def("enqueueJob",
        (void(SysManagerExt::*)(AIGraph*, const std::vector<std::string>&))&SysManagerExt::pyEnqueueJob,
        "Enqueue a batch of jobs to SysManager",
        py::arg("graph"), py::arg("filePaths"),
        py::call_guard<py::gil_scoped_release>())
    .def("waitForAllResults", (void(SysManagerExt::*)())&SysManagerExt::waitForAllResults,
        "Wait for all the enqueued Jobs to SysManager to complete",
        py::call_guard<py::gil_scoped_release>())
    .def("waitForAllResults", (void(SysManagerExt::*)(AIGraph*))&SysManagerExt::waitForAllResults,
        "Wait for all the enqueued Jobs to a particular graph to complete",
        py::call_guard<py::gil_scoped_release>())
    .def("report", &SysManagerExt::report, "Calls the report() method of every kernel")
    .def("resetTimer", &SysManagerExt::resetTimer, "Resets the internal timer in SysManager")
    .def_static("clear", &SysManagerExt::deleteGlobal, "Destroy the System Manager",
        py::call_guard<py::gil_scoped_release>());

  py::class_<NodeParams>(m, "NodeParams")
    .def("getInt", &NodeParams::getValue<int>)
    .def("setInt", &NodeParams::setValue<int>)
    .def("getFloat", &NodeParams::getValue<float>)
    .def("setFloat", &NodeParams::setValue<float>)
    .def("getString", &NodeParams::getValue<std::string>)
    .def("setString", &NodeParams::setValue<std::string>)
    .def("getIntList", &NodeParams::getValue<vector<int>>)
    .def("setIntList", &NodeParams::setValue<vector<int>>)
    .def("getFloatList", &NodeParams::getValue<vector<float>>)
    .def("setFloatList", &NodeParams::setValue<vector<float>>)
    .def("getStringList", &NodeParams::getValue<vector<std::string>>)
    .def("setStringList", &NodeParams::setValue<vector<std::string>>)
    .def("dump", &NodeParams::dump);

  py::class_<DynamicParamValues, NodeParams>(m, "DynamicParamValues")
    .def("dump", &DynamicParamValues::dump)
    .def_readwrite("imagePaths", &DynamicParamValues::imagePaths);
}
