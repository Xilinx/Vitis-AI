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
/*
 * Filename: fmy_xrt_bo.hpp
 *
 * Description:
 * This network is used to getting position and score of faces in the input
 * image Please refer to document "XILINX_AI_SDK_Programming_Guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <xrt/experimental/xrt_aie.h>
#include <xrt/experimental/xrt_bo.h>
#include <xrt/experimental/xrt_device.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/xrt.h>

#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>

namespace vitis {
namespace ai {
struct XrtBo {
  XrtBo(xrtDeviceHandle h, xclDeviceHandle xcl_handle, xclBufferHandle xcl_bo);
  ~XrtBo();

 public:
  static std::shared_ptr<XrtBo> import(xrtDeviceHandle h,
                                       xclDeviceHandle xcl_handle,
                                       xclBufferHandle xcl_bo);

 public:
  xclDeviceHandle xcl_handle;
  xclBufferHandle xcl_bo;
  xrtBufferHandle bo;
  xclBufferExportHandle handle;
};

struct ImportedXrtBo {
  std::shared_ptr<XrtBo> real;
  int offset;
  void* ptr_;
  template <typename Tensor>
  static ImportedXrtBo create(xrtDeviceHandle h, Tensor& tensor, int b) {
    auto ret = ImportedXrtBo{};
    ret.real = XrtBo::import(h, tensor.xcl_bo[b].xcl_handle,
                             tensor.xcl_bo[b].bo_handle);
    ret.offset = tensor.xcl_bo[b].offset;
    ret.ptr_ = tensor.get_data(b);
    return ret;
  }
  template <typename Tensor>
  static std::vector<ImportedXrtBo> create(xrtDeviceHandle h, Tensor& tensor) {
    auto batch = tensor.batch;
    auto ret = std::vector<ImportedXrtBo>{};
    for (auto b = 0u; b < batch; b++) {
      ret.emplace_back(create(h, tensor, b));
    }
    return ret;
  }
  ~ImportedXrtBo();
};

std::ostream& operator<<(std::ostream& out, const XrtBo& bo);
std::ostream& operator<<(std::ostream& out, const ImportedXrtBo& bo);
}  // namespace ai
}  // namespace vitis
