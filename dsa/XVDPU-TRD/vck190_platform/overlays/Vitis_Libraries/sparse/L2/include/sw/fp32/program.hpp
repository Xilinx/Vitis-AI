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

/**
 * @file program.hpp
 * @brief header file for managing host memories.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_PROGRAM_HPP
#define XF_SPARSE_PROGRAM_HPP
#include <stdlib.h>
#include <unordered_map>
#include "L2_utils.hpp"
using namespace std;
namespace xf {
namespace sparse {

template <unsigned int t_PageSize = 4096>
class Program {
   public:
    Program(){};
    ~Program() {
        unordered_map<void*, unsigned long long>::iterator l_it = m_hostBufSz.begin();
        while (l_it != m_hostBufSz.end()) {
            free(l_it->first);
            ++l_it;
        }
    }
    void* allocMem(unsigned long long p_bytes) {
        if (p_bytes == 0) {
            return nullptr;
        }
        void* l_memPtr = aligned_alloc(t_PageSize, alignedNum(p_bytes, t_PageSize));
        if (l_memPtr == nullptr) {
            throw bad_alloc();
        } else {
            m_hostBufSz[l_memPtr] = p_bytes;
        }
        return l_memPtr;
    }
    unsigned long long getBufSz(void* p_bufPtr) {
        unsigned long long l_sz = 0;
        if (m_hostBufSz.find(p_bufPtr) != m_hostBufSz.end()) {
            l_sz = m_hostBufSz[p_bufPtr];
        }
        return l_sz;
    }

   protected:
    unordered_map<void*, unsigned long long> m_hostBufSz;
}; // end class Program

} // end namespace sparse
} // end namespace xf
#endif
