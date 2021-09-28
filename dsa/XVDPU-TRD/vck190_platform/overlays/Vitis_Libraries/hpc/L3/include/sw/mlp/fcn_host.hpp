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

#ifndef XF_HPC_MLP_FCN_HOST_HPP
#define XF_HPC_MLP_FCN_HOST_HPP

//#include "../../handle.hpp"
#include "host.hpp"

namespace xf {
namespace hpc {
namespace mlp {

class FcnArgs : public xf::blas::BLASArgs {
   public:
    virtual ~FcnArgs() {}
    FcnArgs() = delete;
    FcnArgs(unsigned int p_aOffset,
            unsigned int p_bOffset,
            unsigned int p_cOffset,
            unsigned int p_xOffset,
            unsigned int p_m,
            unsigned int p_k,
            unsigned int p_n,
            unsigned int p_lda,
            unsigned int p_ldb,
            unsigned int p_ldc,
            unsigned int p_ldx,
            int p_postScale,
            int p_postShift,
            short p_preluScale,
            short p_preluAlpha)
        : m_fcn_args({xf::blas::OpFcn, p_aOffset, p_bOffset, p_cOffset, p_xOffset, p_m, p_k, p_n, p_lda, p_ldb, p_ldc,
                      p_ldx, 0, 0, 0, 0}) {
        m_fcn_args.m_postScaleVal = (p_postScale << 8) | (p_postShift & 0x000000ff);
        m_fcn_args.m_PReLUVal = (p_preluScale << 6) | (p_preluAlpha & 0x003f);
    }
    size_t sizeInBytes() { return sizeof(m_fcn_args); }
    char* asByteArray() { return reinterpret_cast<char*>(&m_fcn_args); }

   protected:
    struct {
        int m_optype;
        unsigned int m_aOffset, m_bOffset, m_cOffset, m_xOffset, m_m, m_k, m_n, m_lda, m_ldb, m_ldc, m_ldx;
        int m_postScaleVal;
        short m_PReLUVal;
        short m_shortEmpty;
        int m_empty[2];
    } m_fcn_args;
};

class FCNHost : public xf::blas::BLASHost {
   public:
    FCNHost() = delete;
    virtual ~FCNHost() {}
    FCNHost(const FCNHost&) = delete;
    FCNHost(const char* p_xclbin,
            xf::blas::xfblasStatus_t* p_status,
            unsigned int p_kernelIndex,
            unsigned int p_deviceIndex)
        : xf::blas::BLASHost(p_xclbin, p_status, p_kernelIndex, p_deviceIndex) {}

    virtual bool addFCNOp(void* p_a,
                          void* p_b,
                          void* p_c,
                          void* p_bias,
                          unsigned int p_m,
                          unsigned int p_n,
                          unsigned int p_k,
                          unsigned int p_lda,
                          unsigned int p_ldb,
                          unsigned int p_ldc,
                          unsigned int p_ldx,
                          int p_postScale,
                          int p_postShift,
                          short p_preluScale,
                          short p_preluAlpha) {
        if (this->m_bufHandle.find(p_a) == this->m_bufHandle.end() ||
            this->m_bufHandle.find(p_b) == this->m_bufHandle.end() ||
            this->m_bufHandle.find(p_c) == this->m_bufHandle.end() ||
            this->m_bufHandle.find(p_bias) == this->m_bufHandle.end()) {
            return false;
        }

        auto& l_devPtr = this->m_bufHandle;

        uint64_t address_A = l_devPtr[p_a].address();
        uint64_t address_B = l_devPtr[p_b].address();
        uint64_t address_C = l_devPtr[p_c].address();
        uint64_t address_bias = l_devPtr[p_bias].address();

        unsigned long long l_aOff, l_bOff, l_cOff, l_xOff;
        l_aOff = (unsigned long long)address_A;
        l_bOff = (unsigned long long)address_B;
        l_cOff = (unsigned long long)address_C;
        l_xOff = (unsigned long long)address_bias;

        l_aOff -= this->m_baseAddress;
        l_bOff -= this->m_baseAddress;
        l_cOff -= this->m_baseAddress;
        l_xOff -= this->m_baseAddress;

        l_aOff /= this->PAGE_SIZE;
        l_bOff /= this->PAGE_SIZE;
        l_cOff /= this->PAGE_SIZE;
        l_xOff /= this->PAGE_SIZE;
        FcnArgs args(l_aOff, l_bOff, l_cOff, l_xOff, p_m, p_k, p_n, p_lda, p_ldb, p_ldc, p_ldx, p_postScale,
                     p_postShift, p_preluScale, p_preluAlpha);
        this->addInstr(&args);
        this->enableRun();
        return true;
    }

    virtual void addFCNOpByAddress(unsigned int l_aOff,
                                   unsigned int l_bOff,
                                   unsigned int l_cOff,
                                   unsigned int l_xOff,
                                   unsigned int p_m,
                                   unsigned int p_n,
                                   unsigned int p_k,
                                   unsigned int p_lda,
                                   unsigned int p_ldb,
                                   unsigned int p_ldc,
                                   unsigned int p_ldx,
                                   int p_postScale,
                                   int p_postShift,
                                   short p_preluScale,
                                   short p_preluAlpha) {
        FcnArgs args(l_aOff, l_bOff, l_cOff, l_xOff, p_m, p_k, p_n, p_lda, p_ldb, p_ldc, p_ldx, p_postScale,
                     p_postShift, p_preluScale, p_preluAlpha);
        this->addInstr(&args);
        this->enableRun();
    }
};

} // namespace mlp
} // namespace hpc
} // namespace xf

#endif
