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
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace xir {
class Tensor;
}  // namespace xir

namespace vart {

/**
 * @class TensorBuffer
 * @brief Class of TensorBuffer
 * */
class TensorBuffer {
 public:
  explicit TensorBuffer(const xir::Tensor* tensor);

 public:
  virtual ~TensorBuffer() = default;

  enum class location_t {
    /** Host only
     * data() should return a valid pair (0,Nonzero_u);
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
    /** only accessiable by device. */
    DEVICE_1 = 3,
    /** only accessiable by device. */
    DEVICE_2 = 4,
    /** only accessiable by device. */
    DEVICE_3 = 5,
    /** only accessiable by device. */
    DEVICE_4 = 6,
    /** only accessiable by device. */
    DEVICE_5 = 7,
    /** only accessiable by device. */
    DEVICE_6 = 8,
    /** only accessiable by device. */
    DEVICE_7 = 9
  };
  /** @brief for TensorBuffer location message */
  static std::string to_string(location_t value);
  /**
   * @brief copy TensorBuffer from one to another.
   * @param tb_from the source TensorBuffer.
   * @param tb_to the destination TensorBuffer.
   * @return void
   
   Sample code:

   @code
   vart::TensorBuffer* tb_from;
   vart::TensorBuffer* tb_to;
   vart::TensorBuffer::copy_tensor_buffer(tb_from.get(), tb_to.get());
   @endcode
   */
  static void copy_tensor_buffer(vart::TensorBuffer* tb_from,
                                 vart::TensorBuffer* tb_to);

  /**
   * @brief create unowned device tensor buffer with device physical addresses for a tensor.

    There are some limitations on the arguments:
     1. The addr_arrsize must NOT be greater than the tensor batch.
     2. The tensor must have attribute ddr_addr whose value must be 0.

   * @param tensor XIR tensor pointer
   * @param batch_addr Array which contains device physical address for each batch
   * @param addr_arrsize The array size of batch_addr
   * @return Unique pointer of created tensor buffer.
  
   Sample code:
  
   @code
    auto runner = vart::RunnerExt::create_runner(subgraph, attrs);
    auto input_tensors = runner->get_input_tensors();
    auto output_tensors = runner->get_output_tensors();
    std::vector<vart::TensorBuffer*> input_tensor_buffers;
    std::vector<vart::TensorBuffer*> output_tensor_buffers;
    uint64_t in_batch_addr[1];
    uint64_t out_batch_addr[1];
    in_batch_addr[0] = DEVICE_PHY_ADDRESS_IN;
    out_batch_addr[0] = DEVICE_PHY_ADDRESS_OUT;
    auto input_tb = vart::TensorBuffer::create_unowned_device_tensor_buffer(
          input_tensors[0], in_batch_addr, 1);
    auto output_tb = vart::TensorBuffer::create_unowned_device_tensor_buffer(
          output_tensors[0], out_batch_addr, 1);
    input_tensor_buffers.emplace_back(input_tb.get());
    output_tensor_buffers.emplace_back(output_tb.get());
    auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
   @endcode
   */
  static std::unique_ptr<TensorBuffer> create_unowned_device_tensor_buffer(
      const xir::Tensor* tensor, uint64_t batch_addr[], size_t addr_arrsize);

 public:
  /**
   * @brief Get the data address of the index and the 
   *     size of the data available for use.
   * @param idx The index of the data to be accessed,
   * its dimension same as the tensor shape.
   * @return A pair of the data address of the index 
   *  and the size of the data available for use in byte unit.
  
   Sample code:

   @code
    vart::TensorBuffer* tb;
    std::tie(data_addr, tensor_size) = tb->data({0,0,0,0});
   @endcode
   */
  virtual std::pair<std::uint64_t, std::size_t> data(
      const std::vector<std::int32_t> idx = {}) = 0;
  /**
   *@brief Get where the tensor buffer located.
   *@return the tensor buffer location, a location_t enum type value: HOST_VIRT/HOST_PHY/DEVICE_*.

   Sample code:  

   @code
     vart::TensorBuffer* tb;
     switch (tb->get_location()) {
                 case vart::TensorBuffer::location_t::HOST_VIRT:
                       // do nothing
                       break;
                 case vart::TensorBuffer::location_t::HOST_PHY:
                       // do nothing
                       break;
                default:
                       // do nothing
                       break;
           }
   @endcode
   */
  virtual location_t get_location() const { return location_t::HOST_VIRT; }
  /**
   * @brief Get the data physical address of the index 
   * and the size of the data available for use.
   * @param idx The index of the data to be accessed,
   * its dimension same to the tensor shape.
   * @return A pair of the data physical address of the index 
   * and the size of the data available for use in byte unit.

   Sample code:  

   @code
    vart::TensorBuffer* tb;
    std::tie(phy_data, phy_size) = tb->data_phy({0, 0});
   @endcode
   */
  virtual std::pair<uint64_t, size_t> data_phy(
      const std::vector<std::int32_t> idx) {
    return std::make_pair<uint64_t, size_t>(0u, 0u);
  }

  /**
   * @brief Invalid cache for reading Before read, it is no-op
   * in case get_location() returns DEVICE_ONLY or HOST_VIRT.
   * @param offset The start offset address.
   * @param size The data size.
   * @return void
   
   Sample code:  

   @code
    for (auto& output : output_tensor_buffers) {
        output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                    output->get_tensor()->get_shape()[0]);
    }
   @endcode
   */
  virtual void sync_for_read(uint64_t offset, size_t size) {}
  /**
   * @brief Flush cache for writing after write, it is no-op
   * in case get_location() returns DEVICE_ONLY or HOST_VIRT.
   * @param offset The start offset address.
   * @param size The data size.
   * @return void
   
   Sample code:  

   @code
   for (auto& input : input_tensor_buffers) {
       input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                 input->get_tensor()->get_shape()[0]);
   }
   @endcode
   */
  virtual void sync_for_write(uint64_t offset, size_t size){};
  /**
   * @brief copy data from source buffer.
   * @param batch_idx the batch index.
   * @param buf source buffer start address.
   * @param size data size to be copied.
   * @param offset the start offset to be copied.
   * @return void
   */
  virtual void copy_from_host(size_t batch_idx, const void* buf, size_t size,
                              size_t offset);
  /**
   * @brief copy data to destination buffer.
   * @param batch_idx the batch index.
   * @param buf destination buffer start address.
   * @param size data size to be copied.
   * @param offset the start offset to be copied.
   * @return void
   
   Sample code:  

   @code
   vart::TensorBuffer* tb_from;
   vart::TensorBuffer* tb_to;
   for (auto batch = 0u; batch < batch_size; ++batch) {
          std::tie(data, tensor_size) = tb_to->data({(int)batch, 0, 0, 0});
       tb_from->copy_to_host(batch, reinterpret_cast<void*>(data),
                           tensor_size, 0u);
   }
   @endcode 
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
#ifdef _WIN32
  void* bo_handle;
#else
  unsigned int bo_handle;
#endif
};

class TensorBufferExt : public TensorBuffer {
 protected:
  explicit TensorBufferExt(const xir::Tensor* tensor) : TensorBuffer{tensor} {}

 public:
  virtual XclBo get_xcl_bo(int batch_index) const = 0;
};

}  // namespace vart
