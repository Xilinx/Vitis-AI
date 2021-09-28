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

template <typename t_DataType, int t_DataSize>
class WideData {
   private:
    t_DataType m_data[t_DataSize];

   public:
    WideData() {
        for (int i = 0; i < t_DataSize; i++) m_data[i] = 0;
    }

    WideData(const WideData& wd) {
        for (int i = 0; i < t_DataSize; i++) m_data[i] = wd[i];
    }

    WideData(const t_DataType l_value) {
        for (int i = 0; i < t_DataSize; i++) m_data[i] = l_value;
    }

    t_DataType& operator[](int index) { return m_data[index]; }

    const t_DataType& operator[](int index) const { return m_data[index]; }
};

template <int t_PEX, int t_PEZ, typename t_DataType>
void converter(
    int l_x, int l_y, int l_z, const WideData<WideData<t_DataType, t_PEX>, t_PEZ>* mem, vector<t_DataType>& vec) {
    int index = 0;
    for (int i = 0; i < l_x / t_PEX; i++) {
        for (int j = 0; j < l_y; j++) {
            for (int k = 0; k < l_z / t_PEZ; k++) {
                for (int pez = 0; pez < t_PEZ; pez++) {
                    for (int pex = 0; pex < t_PEX; pex++) {
                        vec[(i * t_PEX + pex) * l_y * l_z + j * l_z + k * t_PEZ + pez] = mem[index][pez][pex];
                    }
                }
                index++;
            }
        }
    }
}

template <int t_PEX, int t_PEZ, typename t_DataType>
void converter(
    int l_x, int l_y, int l_z, const vector<t_DataType>& vec, WideData<WideData<t_DataType, t_PEX>, t_PEZ>* mem) {
    int index = 0;
    for (int i = 0; i < l_x / t_PEX; i++) {
        for (int j = 0; j < l_y; j++) {
            for (int k = 0; k < l_z / t_PEZ; k++) {
                for (int pez = 0; pez < t_PEZ; pez++) {
                    for (int pex = 0; pex < t_PEX; pex++) {
                        mem[index][pez][pex] = vec[(i * t_PEX + pex) * l_y * l_z + j * l_z + k * t_PEZ + pez];
                    }
                }
                index++;
            }
        }
    }
}

#endif
