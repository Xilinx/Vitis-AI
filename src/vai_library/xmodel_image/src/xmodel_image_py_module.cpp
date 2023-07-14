/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * */

#include <glog/logging.h>
#include <pybind11/detail/common.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vitis/ai/Mat.hpp"
#include "vitis/ai/xmodel_image.hpp"

namespace py = pybind11;

static vitis::ai::Mat convert_py_buffer_info_to_my_mat(
    const py::buffer_info& info) {
  auto mat_ = vitis::ai::Mat();
  CHECK_EQ(info.shape.size(), 3u) << "only support cv::mat";
  CHECK(info.format == py::format_descriptor<uint8_t>::format())
      << "only support 8UC3";
  auto& shape = info.shape;
  mat_.rows = shape[0];
  mat_.cols = shape[1];
  mat_.data = info.ptr;
  mat_.step = info.strides[0];
  mat_.type = 16;                                                    // CV_8UC3
  LOG(INFO) << "info.strides.size() " << info.strides.size() << " "  //
            << "mat_.step " << mat_.step << " "                      //
            << "mat_.rows " << mat_.rows << " "                      //
            << "mat_.cols " << mat_.cols << " "                      //
      ;
  return mat_;
}

static std::vector<py::buffer_info> request_py_infos(
    const std::vector<py::buffer>& inputs) {
  auto ret = std::vector<py::buffer_info>();
  ret.reserve(inputs.size());
  for (auto& buffer : inputs) {
    ret.emplace_back(buffer.request());
  }
  return ret;
}

static std::vector<vitis::ai::Mat> convert_to_mat(
    std::vector<py::buffer_info>& infos) {
  auto ret = std::vector<vitis::ai::Mat>();
  ret.reserve(infos.size());
  for (auto& info : infos) {
    ret.emplace_back(convert_py_buffer_info_to_my_mat(info));
  }
  return ret;
}

PYBIND11_MODULE(MODULE_NAME, m) {
  m.doc() = "xmodel_image python bindings";
  m.add_object("pb2", py::module::import("vitis.ai.proto.dpu_model_param_pb2"));
  m.def("init_glog", [](const std::string& dir) {
    FLAGS_logtostderr = true;
    FLAGS_alsologtostderr = true;
    FLAGS_log_dir = dir;
    google::InitGoogleLogging("xmodel_image");
  });
  py::class_<vitis::ai::XmodelImage>(m, "XmodelImage")
      .def(py::init<>(&vitis::ai::XmodelImage::create))
      .def("get_batch", &vitis::ai::XmodelImage::get_batch)
      .def("get_width", &vitis::ai::XmodelImage::get_width)
      .def("get_height", &vitis::ai::XmodelImage::get_height)
      .def("get_depth", &vitis::ai::XmodelImage::get_depth)
      .def("run",
           [m](vitis::ai::XmodelImage* self,
               const std::vector<py::buffer>& inputs) {
             auto infos = request_py_infos(inputs);
             auto mats = convert_to_mat(infos);
             auto results = self->run(mats);
             auto ret = py::list();
             for (auto& result : results) {
               std::string buf;
               CHECK(result.SerializeToString(&buf)) << "serialization error";
               auto py_xmodel_result = m.attr("pb2").attr("DpuModelResult")();
               LOG(INFO) << "size of buf=" << buf.size();
               py_xmodel_result.attr("ParseFromString")(
                   py::bytes(&buf[0], buf.size()));
               ret.append(py_xmodel_result);
             }
             return ret;
           })
      //
      ;
}
