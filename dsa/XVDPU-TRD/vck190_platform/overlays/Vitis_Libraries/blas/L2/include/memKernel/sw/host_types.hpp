/**********
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
 * **********/
/**
 *  @brief GEMX common datatypes for HLS kernel code.
 *
 */

#ifndef XF_BLAS_HOST_TYPES_HPP
#define XF_BLAS_HOST_TYPES_HPP

#include <stdint.h>
#include <ostream>
#include <iomanip>
#include <iostream>
#include "ap_int.h"
#include "ap_shift_reg.h"

namespace xf {

namespace blas {

// For C++11
// template<class T>
// auto operator<<(std::ostream& os, const T& t) -> decltype(t.print(os), os)
//{
//  t.print(os);
//  return os;
//}

#define BLAS_CMP_WIDTH 11

template <typename T>
bool cmpVal(
    float p_TolRel, float p_TolAbs, T vRef, T v, std::string p_Prefix, bool& p_exactMatch, unsigned int p_Verbose) {
    float l_diffAbs = std::abs(v - vRef);
    float l_diffRel = l_diffAbs;
    if (vRef != 0) {
        l_diffRel /= std::abs(vRef);
    }
    p_exactMatch = (vRef == v);
    bool l_status = p_exactMatch || (l_diffRel <= p_TolRel) || (l_diffAbs <= p_TolAbs);
    if ((p_Verbose >= 3) || ((p_Verbose >= 2) && !p_exactMatch) || ((p_Verbose >= 1) && !l_status)) {
        std::cout << p_Prefix << "  ValRef " << std::left << std::setw(BLAS_CMP_WIDTH) << vRef << " Val " << std::left
                  << std::setw(BLAS_CMP_WIDTH) << v << "  DifRel " << std::left << std::setw(BLAS_CMP_WIDTH)
                  << l_diffRel << " DifAbs " << std::left << std::setw(BLAS_CMP_WIDTH) << l_diffAbs << "  Status "
                  << l_status << "\n";
    }
    return (l_status);
}

// Memory allocation descriptor for FPGA, file, and program data exchange
// It never allocates nor freed memory
class MemDesc {
   private:
    size_t m_Num4kPages;
    void* m_PageSpace;
    static const unsigned int t_4k = 4096;

   public:
    MemDesc() {}
    MemDesc(size_t p_Num4kPages, void* p_PageSpace) : m_Num4kPages(p_Num4kPages), m_PageSpace(p_PageSpace) {}
    void init(size_t p_Num4kPages, void* p_PageSpace) {
        m_Num4kPages = p_Num4kPages;
        m_PageSpace = p_PageSpace;
    }
    size_t sizeBytes() { return m_Num4kPages * t_4k; }
    void* data() { return m_PageSpace; }
    size_t sizePages() { return m_Num4kPages; }
};

} // namespace
}
#endif
