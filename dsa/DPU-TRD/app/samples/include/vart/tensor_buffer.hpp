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
#pragma once

#include <cstdint>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace xir {
class Tensor;
}  // namespace xir

namespace vart {

class TensorBuffer {
 protected:
  explicit TensorBuffer(const xir::Tensor* tensor);

 public:
  virtual ~TensorBuffer() = default;

  enum class location_t {
    /** Host only
     *
     * data() should return a valid pair (0,Nonzerou);
     *
     * data_phy() should return an invalid pair (0,0u);
     * */
    HOST_VIRT = 0,
    /** continuous physicial memory, shared among host and device.
     * both data () and data_phy() should return a valid pair.
     */
    HOST_PHY = 1,
    /** only accessiable by device.
     *  data () should return an invalid pair (0,0u);
     *  data_phy() should return a valid pair.
     * */
    DEVICE_0 = 2,
    DEVICE_1 = 3,
    DEVICE_2 = 4,
    DEVICE_3 = 5,
    DEVICE_4 = 6,
    DEVICE_5 = 7,
    DEVICE_6 = 8,
    DEVICE_7 = 9
  };
  static std::string to_string(location_t value);
  // copy tensor
  static void copy_tensor_buffer(vart::TensorBuffer* tb_from,
                                 vart::TensorBuffer* tb_to);

 public:
  virtual std::pair<std::uint64_t, std::size_t> data(
      const std::vector<std::int32_t> idx = {}) = 0;
  /** @brief return where the tensor buffer resistant. */
  virtual location_t get_location() const { return location_t::HOST_VIRT; }

  /** @brief return the physical addresses for zero copy. */
  virtual std::pair<uint64_t, size_t> data_phy(
      const std::vector<std::int32_t> idx) {
    return std::make_pair<uint64_t, size_t>(0u, 0u);
  }

  /** @brief invalid cache for reading, it is no-op in case get_location()
   * returns DEVICE_ONLY or HOST_VIRT */
  virtual void sync_for_read(uint64_t offset, size_t size) {}
  /** @brief flush cache for writing, it is no-op in case get_location()
   * returns DEVICE_ONLY or HOST_VIRT */
  virtual void sync_for_write(uint64_t offset, size_t size){};

  virtual void copy_from_host(size_t batch_idx, const void* buf, size_t size,
                              size_t offset);
  virtual void copy_to_host(size_t batch_idx, void* buf, size_t size,
                            size_t offset);

 public:
  const xir::Tensor* get_tensor() const;

  /** @brief for fancy log messages */
  virtual std::string to_string() const;

 protected:
  const xir::Tensor* tensor_;
};

}  // namespace vart
