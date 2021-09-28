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
/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XF_BLAS_GEMV_HOST_HPP
#define XF_BLAS_GEMV_HOST_HPP

#include "handle.hpp"
#include "host.hpp"

namespace xf {

namespace blas {

class GemvArgs : public BLASArgs {
   public:
    virtual ~GemvArgs() {}
    GemvArgs() = delete;
    GemvArgs(unsigned int p_aOffset,
             unsigned int p_bOffset,
             unsigned int p_cOffset,
             unsigned int p_m,
             unsigned int p_n,
             unsigned int p_lda)
        : m_GemvArgs({int(OpGemv), p_aOffset, p_bOffset, p_cOffset, p_m, p_n, p_lda, 0, 0, 0, 0, 0, 0, 0, 0, 0}) {}
    size_t sizeInBytes() { return sizeof(m_GemvArgs); }
    char* asByteArray() { return reinterpret_cast<char*>(&m_GemvArgs); }

   protected:
    struct {
        int m_optype;
        unsigned int m_aOffset, m_bOffset, m_cOffset, m_m, m_n, m_lda;
        int m_empty[9];
    } m_GemvArgs;
};

class GEMVHost : public BLASHost {
   public:
    GEMVHost() = delete;
    virtual ~GEMVHost() {}
    GEMVHost(const GEMVHost&) = delete;
    GEMVHost(const char* p_xclbin, xfblasStatus_t* p_status, unsigned int p_kernelIndex, unsigned int p_deviceIndex)
        : BLASHost(p_xclbin, p_status, p_kernelIndex, p_deviceIndex) {}

    virtual xfblasStatus_t addGEMVOp(
        void* p_a, void* p_b, void* p_c, unsigned int p_m, unsigned int p_n, unsigned int p_lda) {
        if (this->m_bufHandle.find(p_a) == this->m_bufHandle.end() ||
            this->m_bufHandle.find(p_b) == this->m_bufHandle.end() ||
            this->m_bufHandle.find(p_c) == this->m_bufHandle.end()) {
            return XFBLAS_STATUS_ALLOC_FAILED;
        }

        auto& l_devPtr = this->m_bufHandle;

        uint64_t address_A = l_devPtr[p_a].address();
        uint64_t address_B = l_devPtr[p_b].address();
        uint64_t address_C = l_devPtr[p_c].address();
        unsigned long long l_aOff, l_bOff, l_cOff;
        l_aOff = (unsigned long long)address_A;
        l_bOff = (unsigned long long)address_B;
        l_cOff = (unsigned long long)address_C;

        l_aOff -= this->m_baseAddress;
        l_bOff -= this->m_baseAddress;
        l_cOff -= this->m_baseAddress;

        l_aOff /= this->PAGE_SIZE;
        l_bOff /= this->PAGE_SIZE;
        l_cOff /= this->PAGE_SIZE;

        GemvArgs l_gargs(l_aOff, l_bOff, l_cOff, p_m, p_n, p_lda);
        this->addInstr(&l_gargs);
        this->enableRun();

        return XFBLAS_STATUS_SUCCESS;
    }
};

} // namespace blas

} // namespace xf

#endif
