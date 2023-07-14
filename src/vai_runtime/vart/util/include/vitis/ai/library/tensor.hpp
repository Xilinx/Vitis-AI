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
#include <array>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <string>
namespace vitis {
namespace ai {
namespace library {

/**
 * @brief The DataType of the tensor
 */
enum DataType {
  DT_INVALID = 0,
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_UINT8 = 4,
  DT_INT16 = 5,
  DT_INT8 = 6,
  DT_STRING = 7,
  DT_COMPLEX64 = 8,
  DT_INT64 = 9,
  DT_BOOL = 10,
  DT_QINT8 = 11,
  DT_QUINT8 = 12,
  DT_QINT32 = 13,
  DT_BFLOAT16 = 14,
  DT_QINT16 = 15,
  DT_QUINT16 = 16,
  DT_UINT16 = 17,
  DT_COMPLEX128 = 18,
  DT_HALF = 19,
  DT_RESOURCE = 20,
  DT_VARIANT = 21,
  DT_UINT32 = 22,
  DT_UINT64 = 23,
  DT_FLOAT_REF = 101,
  DT_DOUBLE_REF = 102,
  DT_INT32_REF = 103,
  DT_UINT8_REF = 104,
  DT_INT16_REF = 105,
  DT_INT8_REF = 106,
  DT_STRING_REF = 107,
  DT_COMPLEX64_REF = 108,
  DT_INT64_REF = 109,
  DT_BOOL_REF = 110,
  DT_QINT8_REF = 111,
  DT_QUINT8_REF = 112,
  DT_QINT32_REF = 113,
  DT_BFLOAT16_REF = 114,
  DT_QINT16_REF = 115,
  DT_QUINT16_REF = 116,
  DT_UINT16_REF = 117,
  DT_COMPLEX128_REF = 118,
  DT_HALF_REF = 119,
  DT_RESOURCE_REF = 120,
  DT_VARIANT_REF = 121,
  DT_UINT32_REF = 122,
  DT_UINT64_REF = 123
};

struct XclBoInfo {
  unsigned int offset;
  void* xcl_handle;
  unsigned int bo_handle;
};

/**
 *@struct Tensor
 *@brief The basic abstract structure of neural network layer.
 */
struct Tensor {
  /// The total size of this tensor's data.
  size_t size;
  /// The batch required by DPU core.
  size_t batch;
  /// The height of this tensor.
  size_t height;
  /// The width of this tensor.
  size_t width;
  /// The channel of this tensor.
  size_t channel;
  /// The fixed position of this tensor, the value range from 0 to 7.
  int fixpos;
  /// This tensor's data type.
  DataType dtype;
  /// name for debug purpose
  std::string name;
  std::array<XclBoInfo, 16> xcl_bo;
};

/**
 * @struct InputTensor
 * @brief The actual data of input tensor.
 */
struct InputTensor : public Tensor {
  /// The start pointer of this Tensor.
  void*& get_data(size_t batch) { return data[batch]; }
  const void* get_data(size_t batch) const { return data[batch]; }

 public:
  std::array<void*, 16> data;
};

/**
 * @struct OutputTensor
 * @brief The actual data of output tensor.
 */
struct OutputTensor : public Tensor {
  /// The start pointer of this tensor.
  void*& get_data(size_t batch) { return data[batch]; }
  const void* get_data(size_t batch) const { return data[batch]; }

 private:
  std::array<void*, 16> data;
};

inline std::ostream& operator<<(std::ostream& out,
                                const vitis::ai::library::InputTensor& v) {
  out << "Input [ " << v.name << " ] {"  //
      << ", size=" << v.size             //
      << ", batch=" << v.batch           //
      << ", h=" << v.height              //
      << ", w=" << v.width               //
      << ", c=" << v.channel             //
      << ", fixpos=" << v.fixpos         //
      << ", virt= (";
  for (size_t b = 0; b < v.batch; ++b) {
    out << "{"                              //
        << v.get_data(b) << " "             //
        << v.xcl_bo[b].offset << " "        //
        << v.xcl_bo[b].xcl_handle << " "    //
        << v.xcl_bo[b].bo_handle << " " <<  //
        "}";                                //
  }
  out << ")}";  //

  return out;
}

inline std::ostream& operator<<(std::ostream& out,
                                const vitis::ai::library::OutputTensor& v) {
  out << "Output [ " << v.name << " ] {"  //、·
      << ", size=" << v.size              //
      << ", h=" << v.height               //
      << ", w=" << v.width                //
      << ", c=" << v.channel              //
      << ", fixpos=" << v.fixpos          //
      << ", virt= (";
  for (size_t b = 0; b < v.batch; ++b) {
    out << "{"                              //
        << v.get_data(b) << " "             //
        << v.xcl_bo[b].offset << " "        //
        << v.xcl_bo[b].xcl_handle << " "    //
        << v.xcl_bo[b].bo_handle << " " <<  //
        "}";                                //
  }
  out << ")}";  //
  return out;
}

inline std::ostream& operator<<(std::ostream& out,
                                const vitis::ai::library::Tensor& v) {
  out << "Tensor [ " << v.name << " ] {"  //
      << ", size=" << v.size              //
      << ", batch=" << v.batch            //
      << ", h=" << v.height               //
      << ", w=" << v.width                //
      << ", c=" << v.channel              //
      << ", fixpos=" << v.fixpos          //
      << "}";                             //
  return out;
}

/**
 *@brief Calculate fix-scale of input tensor
 * In dpu task there should be all the input values are fixed-point
 * number, we could use this for convert within fixed or float.
 *@param tensor InputTensor
 *@return scale
 *
 */
inline float tensor_scale(const vitis::ai::library::InputTensor& tensor) {
  return std::exp2f(1.0f * (float)tensor.fixpos);
}

/**
 *@brief Calculate fix-scale for output tensor
 * For dpu task, in order to convert a float-point tensor into a fix
 * point tensor, we need to multiply values by the scale
 *@param tensor OutputTensor
 *@return scale
 *
 */
inline float tensor_scale(const vitis::ai::library::OutputTensor& tensor) {
  return std::exp2f(-1.0f * (float)tensor.fixpos);
}
}  // namespace library
}  // namespace ai
}  // namespace vitis
