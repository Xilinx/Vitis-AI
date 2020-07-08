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

#ifndef __AKS_DATA_DESCRIPTOR_H_
#define __AKS_DATA_DESCRIPTOR_H_

#include <string>
#include <initializer_list>
#include <vector>


namespace AKS
{
  enum class DataOrg
  {
    UNKNOWN,
    NCHW,
    NHWC
  };

  /// Data type of the underlying data in a data descriptor
  enum class DataType
  {
    UINT8   = 1,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 2,
    UINT32  = 4,
    INT32   = 4,
    FLOAT32 = 4
  };

  /// Multi-dimensional array structure that stores the input/output of each node.
  /// It encapsulates underlying buffer, datatype, shape of buffer etc.
  class DataDescriptor
  {
    public:
      /// Creates a Data Descriptor from the given specifications.
      /// It allocates a contiguous buffer as per the given shape and data type
      /// @param shape It could be a vector or initializer list
      /// @param dtype Type of data (from enum class DataType)
      /// @param ownData Decides who owns the buffer. If true, destructor will free the data.
      /// @param buf Use existing buffer instead of creating new one.
      DataDescriptor();
      DataDescriptor(std::initializer_list<int> shape, AKS::DataType dtype);
      DataDescriptor(std::vector<int> shape, AKS::DataType dtype);

      DataDescriptor(const DataDescriptor& src);
      DataDescriptor(DataDescriptor&& src);
      DataDescriptor& operator=(const DataDescriptor& src);
      DataDescriptor& operator=(DataDescriptor&& src);
      ~DataDescriptor();

      /// Get the shape of data
      const std::vector<int>& getShape() const { return _shape; }
      std::vector<int> getStride() const {
        auto dsize  = static_cast<int>(_dtype);
        std::vector<int> strides(_shape.begin()+1, _shape.end());
        strides.push_back(dsize);
        for(int i=_shape.size()-2; i>=0; --i) strides[i] *= strides[i+1];
        return strides;
      }

      const DataType dtype() { return _dtype; }

      /// Get the shape as a string
      const std::string getStringShape() {
        std::string s = "Shape : ";
        for(auto val: _shape) {
          s += std::to_string(val);
          s += "x";
        }
        return s;
      }

      /// Get a mutable handle to underlying data
      /// Template parameter specifies the return type of buffer.
      template<typename T=void>
      T* data() { return static_cast<T*>(_data); }

      /// Get an immutable handle to underlying data
      template<typename T=void>
      const T* const_data() const { return static_cast<T*>(_data); }

      /// Get nelems from shape
      size_t getNumberOfElements();

      /// Get CRC checksum of all the underlying data. Useful for debugging
      int getCRC();

    private:
      // Data and Shape
      void *_data = nullptr;
      std::vector<int> _shape;

      // Data Order/Format and Type
      DataOrg _org = DataOrg::NCHW;
      DataType _dtype;
  };
}
#endif // __AKS_DATA_DESCRIPTOR_H_
