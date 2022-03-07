/*
 * Copyright 2019 Xilinx Inc.
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
 */
#include "./my_xrt_bo.hpp"

#include <glog/logging.h>

#include <vitis/ai/env_config.hpp>

DEF_ENV_PARAM(DEBUG_MY_XRT_BO, "0")
namespace vitis {
namespace ai {
XrtBo::XrtBo(xrtDeviceHandle h, xclDeviceHandle xcl_handle,
             xclBufferHandle xcl_bo) {
  this->xcl_handle = xcl_handle;
  this->xcl_bo = xcl_bo;
  auto xcl_bo_exp = xclExportBO(xcl_handle, xcl_bo);
  this->handle = xcl_bo_exp;
  this->bo = xrtBOImport(h, xcl_bo_exp);
  LOG_IF(INFO, false) << "import "
                      << "bo_handle " << xcl_bo << " "  //
                      << "xcl_handle " << xcl_handle << " @" << (void*)this;

  CHECK(xcl_bo_exp > 0) << "xcl_bo_exp " << xcl_bo_exp << " "  //
                        << "bo_handle " << handle << " "       //
                        << "xcl_handle " << xcl_handle << " "  //
      ;
  CHECK(bo != nullptr) << "h " << h << " "                     //
                       << "xcl_bo_exp " << xcl_bo_exp << " ";  //
  LOG_IF(INFO, ENV_PARAM(DEBUG_MY_XRT_BO)) << "this=" << *this;
}

XrtBo::~XrtBo() {
  LOG_IF(INFO, false) << "free bo: " << *this;
  xrtBOFree(bo);
}

std::shared_ptr<XrtBo> XrtBo::import(xrtDeviceHandle h,
                                     xclDeviceHandle xcl_handle,
                                     xclBufferHandle xcl_bo) {
  auto key = std::string("x") + std::to_string((intptr_t)xcl_handle) + ":" +
             std::to_string(xcl_bo);
  return vitis::ai::WeakStore<std::string, XrtBo>::create(key, h, xcl_handle,
                                                          xcl_bo);
}

ImportedXrtBo::~ImportedXrtBo() { real = nullptr; }

std::ostream& operator<<(std::ostream& out, const XrtBo& info) {
  out << "XrtBO{";
  out << "@" << (void*)&info << ",";
  out << "handle=" << info.handle << ",";
  out << "xcl_bo=" << info.xcl_bo << ",";
  out << "xrt_bo=" << info.bo << ",";
  out << std::hex << "0x" << xrtBOAddress(info.bo) << std::dec;
  out << "}";
  return out;
}
std::ostream& operator<<(std::ostream& out, const ImportedXrtBo& info) {
  out << "ImportXrBo{";
  out << "real=" << *info.real << ",";
  out << "offset=" << info.offset << ",";
  out << "}";
  return out;
}

}  // namespace ai
}  // namespace vitis
