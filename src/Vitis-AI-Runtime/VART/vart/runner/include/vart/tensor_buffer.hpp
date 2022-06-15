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
 public:
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
  /** @brief for TensorBuffer location message */
  static std::string to_string(location_t value);
  /**
   * @brief copy TensorBuffer from one to another.
   * @param the source TensorBuffer.
   * @param the destination TensorBuffer.
   */
  static void copy_tensor_buffer(vart::TensorBuffer* tb_from,
                                 vart::TensorBuffer* tb_to);

 public:
  /**
   * @brief Get the data address of the index and
   * the left accessible data size.
   * @param The index of the data to be accessed,
   * its dimension same to the tensor shape.
   * @return A pair of the data address of the index and
   * the left accessible data size in byte unit.
   */
  virtual std::pair<std::uint64_t, std::size_t> data(
      const std::vector<std::int32_t> idx = {}) = 0;
  /**
   *@brief Get where the tensor buffer located.
   *@return the tensor buffer location : HOST_VIRT/HOST_PHY/DEVICE_*.
   */
  virtual location_t get_location() const { return location_t::HOST_VIRT; }
  /**
   * @brief Get the data physical address of the index and
   * the left accessible data size.
   * @param The index of the data to be accessed,
   * its dimension same to the tensor shape.
   * @return A pair of the data physical address of the index and
   * the left accessible data size in byte unit.
   */
  virtual std::pair<uint64_t, size_t> data_phy(
      const std::vector<std::int32_t> idx) {
    return std::make_pair<uint64_t, size_t>(0u, 0u);
  }

  /**
   * @brief Invalid cache for reading Before read, it is no-op
   * in case get_location() returns DEVICE_ONLY or HOST_VIRT.
   * @param The start offset address.
   * @param The data size.
   */
  virtual void sync_for_read(uint64_t offset, size_t size) {}
  /**
   * @brief Flush cache for writing after write, it is no-op
   * in case get_location() returns DEVICE_ONLY or HOST_VIRT.
   * @param The start offset address.
   * @param The data size.
   */
  virtual void sync_for_write(uint64_t offset, size_t size){};
  /**
   * @brief copy data from source buffer.
   * @param the batch index.
   * @param source buffer start address.
   * @param data size to be copied.
   * @param the start offset to be copied.
   */
  virtual void copy_from_host(size_t batch_idx, const void* buf, size_t size,
                              size_t offset);
  /**
   * @brief copy data to destination buffer.
   * @param the batch index.
   * @param destination buffer start address.
   * @param data size to be copied.
   * @param the start offset to be copied.
   */
  virtual void copy_to_host(size_t batch_idx, void* buf, size_t size,
                            size_t offset);

 public:
  /**
   *@brief Get tensor of TensorBuffer.
   *@return A pointer to the tensor.
   */
  const xir::Tensor* get_tensor() const;

  /** @brief for fancy log messages */
  virtual std::string to_string() const;

 protected:
  const xir::Tensor* tensor_;
};

struct XclBo {
  void* xcl_handle;
  unsigned int bo_handle;
};

class TensorBufferExt : public TensorBuffer {
 protected:
  explicit TensorBufferExt(const xir::Tensor* tensor) : TensorBuffer{tensor} {}

 public:
  virtual XclBo get_xcl_bo(int batch_index) const = 0;
};

}  // namespace vart
