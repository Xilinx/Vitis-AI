/*
 * Copyright 2019 Xilinx, Inc.
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
#ifndef XF_RTM_TYPES_HPP
#define XF_RTM_TYPES_HPP

/**
 * @file types.hpp
 * @brief class WideData provides wide datatype for host code
 * @tparam t_DataType the basic datatype
 * @tparam t_DataSize is the number of data in the object
 */

template <typename t_DataType, unsigned int t_DataSize>
class WideData {
   private:
    t_DataType m_data[t_DataSize];

   public:
    WideData(const t_DataType l_value = 0) {
        for (unsigned int i = 0; i < t_DataSize; i++) m_data[i] = l_value;
    }
    t_DataType& operator[](int index) { return m_data[index]; }
    const t_DataType& operator[](int index) const { return m_data[index]; }
};

#endif
