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
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef XF_BLAS_HOST_HPP
#define XF_BLAS_HOST_HPP

#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

#include <unistd.h>

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_bo.h"

#include "../utility/utility.hpp"
#include "helper.hpp"

#if BLAS_streamingKernel
#include "ISA.hpp"
#endif

#define IDX2R(i, j, ld) (((i) * (ld)) + (j))

using namespace std;

namespace xf {

namespace blas {

#if BLAS_streamingKernel
typedef MemInstr<64> MemInstrType;
typedef GemmInstr<64> GemmInstrType;
#endif

class XFpga {
   public:
    xrt::device m_device;
    xrt::uuid m_uuid;

    XFpga() = delete;
    XFpga(const char* p_xclbin, int* p_err, unsigned int deviceIndex = 0) {
        m_device = xrt::device(deviceIndex);
        m_uuid = m_device.load_xclbin(p_xclbin);
    }

    ~XFpga() {}

    xrt::bo createBuf(void* p_ptr, size_t p_szBytes, unsigned int p_mem) {
        return xrt::bo(m_device, p_ptr, p_szBytes, XCL_BO_FLAGS_NONE, p_mem);
    }

    bool copyToFpga(xrt::bo& p_bufHandle, size_t p_szBytes) {
        p_bufHandle.sync(XCL_BO_SYNC_BO_TO_DEVICE, p_szBytes, 0);
        return true;
    }

    bool copyFromFpga(xrt::bo& p_bufHandle, size_t p_szBytes) {
        p_bufHandle.sync(XCL_BO_SYNC_BO_FROM_DEVICE, p_szBytes, 0);
        return true;
    }

    void execKernel(xrt::bo& m_instrBufHandle, xrt::kernel& p_kernel) {
        xrt::run l_run;
        l_run = p_kernel(m_instrBufHandle, m_instrBufHandle);
        l_run.wait();
    }
};

class XFpgaHold {
   public:
    unordered_map<unsigned int, shared_ptr<XFpga> > m_xFpgaPtr;
    static XFpgaHold& instance() {
        static XFpgaHold theInstance;
        return theInstance;
    }

   protected:
    XFpgaHold() {}
};

class XHost {
   protected:
    static const unsigned int PAGE_SIZE = 4096;
    unordered_map<void*, void*> m_hostMat;
    unordered_map<void*, xrt::bo> m_bufHandle;
    unordered_map<void*, unsigned long long> m_hostMatSz;
    shared_ptr<XFpga> m_fpga;
    vector<unsigned int> m_execHandles;
#if BLAS_streamingKernel
    unsigned char* m_instrBuf;
#else
    char* m_instrBuf;
#endif
    unsigned int m_instrOffset;
    unsigned int m_cuIndex;

    vector<xrt::kernel> m_kernel;
    int m_memId;
    uint64_t m_baseAddress;
    xrt::bo m_instrBufHandle;

   public:
    XHost() = delete;
    XHost(const char* p_xclbin, xfblasStatus_t* p_status, unsigned int p_kernelIndex, unsigned int p_deviceIndex) {
        m_fpga = XFpgaHold::instance().m_xFpgaPtr[p_deviceIndex];
#if BLAS_streamingKernel
        xrt::kernel l_kernel = xrt::kernel(m_fpga->m_device, m_fpga->m_uuid.get(), "gemmLoadStoreKernel");
        m_kernel.push_back(l_kernel);
        m_memId = 0;
#elif BLAS_runFcn
        xrt::kernel l_kernel = xrt::kernel(m_fpga->m_device, m_fpga->m_uuid.get(), "fcnKernel");
        m_kernel.push_back(l_kernel);
        m_memId = m_kernel[0].group_id(0);
#else
        xrt::kernel l_kernel = xrt::kernel(m_fpga->m_device, m_fpga->m_uuid.get(), "blasKernel");
        m_kernel.push_back(l_kernel);
        m_memId = m_kernel[0].group_id(0);
#endif
        int l_memAllocStatus = posix_memalign((void**)&m_instrBuf, PAGE_SIZE, PAGE_SIZE * 2);
        if (l_memAllocStatus) {
            *p_status = XFBLAS_STATUS_ALLOC_FAILED;
        }
        memset(m_instrBuf, 0, PAGE_SIZE * 2);
        m_instrOffset = 0;
        m_instrBufHandle = m_fpga->createBuf(m_instrBuf, PAGE_SIZE * 2, m_memId);
        m_baseAddress = m_instrBufHandle.address();
    }

    bool addMatRestricted(void* p_hostHandle, void* p_matPtr, unsigned long long p_bufSize) {
        auto& l_hostPtr = m_hostMat;
        auto& l_hostSzPtr = m_hostMatSz;
        if (((unsigned long)p_matPtr & (PAGE_SIZE - 1)) != 0) {
            void* l_matPtr;
            posix_memalign((void**)&l_matPtr, 4096, p_bufSize);
            memcpy(l_matPtr, p_matPtr, p_bufSize);
            if (l_hostPtr.find(p_hostHandle) == l_hostPtr.end()) {
                l_hostPtr[p_hostHandle] = l_matPtr;
                l_hostSzPtr[p_hostHandle] = p_bufSize;
                return true;
            } else if (m_hostMatSz[p_hostHandle] != p_bufSize) {
                l_hostPtr[p_hostHandle] = l_matPtr;
                l_hostSzPtr[p_hostHandle] = p_bufSize;
                this->m_bufHandle.erase(p_hostHandle);
                return true;
            }
        } else {
            if (l_hostPtr.find(p_hostHandle) == l_hostPtr.end()) {
                l_hostPtr[p_hostHandle] = p_matPtr;
                l_hostSzPtr[p_hostHandle] = p_bufSize;
                return true;
            } else if (m_hostMatSz[p_hostHandle] != p_bufSize) {
                l_hostPtr[p_hostHandle] = p_matPtr;
                l_hostSzPtr[p_hostHandle] = p_bufSize;
                this->m_bufHandle.erase(p_hostHandle);
                return true;
            }
        }
        return false;
    }

    xfblasStatus_t allocMatRestricted(void* p_hostHandle, void* p_matPtr, unsigned long long p_bufSize) {
        addMatRestricted(p_hostHandle, p_matPtr, p_bufSize);
        auto& l_hostPtr = m_hostMat;
        auto& l_devPtr = m_bufHandle;
        auto& l_hostSzPtr = m_hostMatSz;
        if (l_devPtr.find(p_hostHandle) != l_devPtr.end()) {
            // xclFreeBO(m_fpga->m_handle, l_devPtr[p_hostHandle]);
            if (((unsigned long)p_matPtr & (PAGE_SIZE - 1)) != 0) {
                void* l_matPtr;
                posix_memalign((void**)&l_matPtr, 4096, p_bufSize);
                memcpy(l_matPtr, p_matPtr, p_bufSize);
                l_hostPtr[p_hostHandle] = l_matPtr;
                l_devPtr[p_hostHandle] = m_fpga->createBuf(l_hostPtr[p_hostHandle], l_hostSzPtr[p_hostHandle], m_memId);
            } else {
                l_hostPtr[p_hostHandle] = p_matPtr;
                l_devPtr[p_hostHandle] = m_fpga->createBuf(l_hostPtr[p_hostHandle], l_hostSzPtr[p_hostHandle], m_memId);
            }
        } else {
            l_devPtr[p_hostHandle] = m_fpga->createBuf(l_hostPtr[p_hostHandle], l_hostSzPtr[p_hostHandle], m_memId);
        }
        return XFBLAS_STATUS_SUCCESS;
    }

    template <typename t_dataType>
    xfblasStatus_t allocMat(t_dataType* p_devPtr, size_t p_bufSize) {
        auto& l_devPtr = m_bufHandle;
        auto& l_hostSzPtr = m_hostMatSz;
        if (l_devPtr.find(*p_devPtr) != l_devPtr.end()) {
            return XFBLAS_STATUS_ALLOC_FAILED;
        } else {
            xrt::bo l_deviceHandle = xrt::bo(m_fpga->m_device, p_bufSize, XCL_BO_FLAGS_NONE, m_memId);
            p_devPtr = l_deviceHandle.map<t_dataType*>();
            memset(*p_devPtr, 0, p_bufSize);
            l_hostSzPtr[*p_devPtr] = p_bufSize;
            l_devPtr[*p_devPtr] = xrt::bo(l_deviceHandle);
            return XFBLAS_STATUS_SUCCESS;
        }

        return XFBLAS_STATUS_SUCCESS;
    }

    template <typename t_dataType>
    xfblasStatus_t setMatToFPGA(
        void* p_hostHandle, int p_rows, int p_lda, int p_paddedLda, t_dataType& p_hostPtr, t_dataType& p_devPtr) {
        auto& l_devPtr = m_bufHandle;
        auto& l_hostSzPtr = m_hostMatSz;
        if (l_devPtr.find(p_hostHandle) != l_devPtr.end()) {
            for (int i = 0; i < p_rows; i++) {
                for (int j = 0; j < p_lda; j++) {
                    p_devPtr[IDX2R(i, j, p_paddedLda)] = p_hostPtr[IDX2R(i, j, p_lda)];
                }
            }
            if (!m_fpga->copyToFpga(l_devPtr[p_hostHandle], l_hostSzPtr[p_hostHandle])) {
                return XFBLAS_STATUS_ALLOC_FAILED;
            }
        } else {
            return XFBLAS_STATUS_ALLOC_FAILED;
        }
        return XFBLAS_STATUS_SUCCESS;
    }

    xfblasStatus_t setMatToFPGARestricted(void* p_hostHandle) {
        auto& l_devPtr = m_bufHandle;
        auto& l_hostSzPtr = m_hostMatSz;
        if (!m_fpga->copyToFpga(l_devPtr[p_hostHandle], l_hostSzPtr[p_hostHandle])) {
            return XFBLAS_STATUS_ALLOC_FAILED;
        }

        return XFBLAS_STATUS_SUCCESS;
    }

#if BLAS_streamingKernel
    void addInstr(MemInstrType* l_memInstr, size_t l_instrSize) {
        unsigned char* l_currPos = &m_instrBuf[m_instrOffset];
        l_memInstr->storeMem(l_currPos);
        m_instrOffset += l_instrSize;
    }
#else
    void addInstr(BLASArgs* p_args) {
        char* l_instr = p_args->asByteArray();
        char* l_currPos = &m_instrBuf[m_instrOffset];
        memcpy(l_currPos, l_instr, p_args->sizeInBytes());
        m_instrOffset += p_args->sizeInBytes();
    }
#endif

    template <typename t_dataType>
    xfblasStatus_t getMat(
        void* p_hostHandle, int p_rows, int p_lda, int p_paddedLda, t_dataType& p_hostPtr, t_dataType& p_devPtr) {
        auto& l_hostSzPtr = m_hostMatSz;
        auto& l_devPtr = m_bufHandle;
        if (l_devPtr.find(p_hostHandle) != l_devPtr.end()) {
            if (!m_fpga->copyFromFpga(l_devPtr[p_hostHandle], l_hostSzPtr[p_hostHandle])) {
                return XFBLAS_STATUS_ALLOC_FAILED;
            }
            for (int i = 0; i < p_rows; i++) {
                for (int j = 0; j < p_lda; j++) {
                    p_hostPtr[IDX2R(i, j, p_lda)] = p_devPtr[IDX2R(i, j, p_paddedLda)];
                }
            }
        } else {
            return XFBLAS_STATUS_ALLOC_FAILED;
        }
        return XFBLAS_STATUS_SUCCESS;
    }

    xfblasStatus_t deviceSync() {
        for (auto& l_devPtr : m_bufHandle) {
            if (!m_fpga->copyToFpga(l_devPtr.second, m_hostMatSz[l_devPtr.first])) {
                return XFBLAS_STATUS_ALLOC_FAILED;
            }
        }
        return XFBLAS_STATUS_SUCCESS;
    }

    xfblasStatus_t getMatManaged() {
        for (auto& l_devPtr : m_bufHandle) {
            if (!m_fpga->copyFromFpga(l_devPtr.second, m_hostMatSz[l_devPtr.first])) {
                return XFBLAS_STATUS_ALLOC_FAILED;
            }
        }
        return XFBLAS_STATUS_SUCCESS;
    }

    xfblasStatus_t getMatRestricted(void* p_hostHandle, void* p_matPtr) {
        auto& l_hostPtr = m_hostMat;
        auto& l_hostSzPtr = m_hostMatSz;
        auto& l_devPtr = m_bufHandle;
        if (l_hostPtr.find(p_hostHandle) != l_hostPtr.end()) {
            if (!m_fpga->copyFromFpga(l_devPtr[p_hostHandle], l_hostSzPtr[p_hostHandle])) {
                return XFBLAS_STATUS_ALLOC_FAILED;
            }
            if (((unsigned long)p_matPtr & (PAGE_SIZE - 1)) != 0) {
                memcpy(p_matPtr, l_hostPtr[p_hostHandle], l_hostSzPtr[p_hostHandle]);
            }
        } else {
            return XFBLAS_STATUS_ALLOC_FAILED;
        }
        return XFBLAS_STATUS_SUCCESS;
    }

    xfblasStatus_t getMatByAddress(void* p_matPtr, unsigned long long p_bufSize, unsigned int offset) {
        uint64_t l_address = offset * PAGE_SIZE + m_baseAddress;
        if (xclUnmgdPread(m_fpga->m_device, 0, p_matPtr, p_bufSize, l_address) < 0) {
            return XFBLAS_STATUS_ALLOC_FAILED;
        }
        return XFBLAS_STATUS_SUCCESS;
    }

    void clearInstrBuf() {
        memset(this->m_instrBuf, 0, PAGE_SIZE);
        this->m_instrOffset = 0;
    }

    unsigned long long getAddress(void* p_hostHandle) {
        uint64_t address_A = m_bufHandle[p_hostHandle].address();
        unsigned long long l_aOff = (unsigned long long)address_A;
        l_aOff -= m_baseAddress;
        l_aOff /= PAGE_SIZE;
        return l_aOff;
    }

    xfblasStatus_t freeMat(void* p_hostHandle) {
        auto& l_devPtr = m_bufHandle;
        if (l_devPtr.find(p_hostHandle) == l_devPtr.end()) {
            return XFBLAS_STATUS_ALLOC_FAILED;
        } else {
            this->m_bufHandle.erase(p_hostHandle);
            this->m_hostMatSz.erase(p_hostHandle);
            if (!m_hostMat.empty()) {
                this->m_hostMat.erase(p_hostHandle);
            }
            return XFBLAS_STATUS_SUCCESS;
        }
    }

    xfblasStatus_t closeContext(unsigned int p_kernelIndex) {
        free(m_instrBuf);
        return XFBLAS_STATUS_SUCCESS;
    }
};

class BLASHost : public XHost {
   private:
    bool m_execControl = true;

   public:
    BLASHost() = delete;
    virtual ~BLASHost() {}
    BLASHost(const BLASHost&) = delete;

    BLASHost(const char* p_xclbin, xfblasStatus_t* p_status, unsigned int p_kernelIndex, unsigned int p_deviceIndex)
        : XHost(p_xclbin, p_status, p_kernelIndex, p_deviceIndex) {}

    xfblasStatus_t execute() {
        xfblasStatus_t l_status = XFBLAS_STATUS_SUCCESS;
        if (m_execControl) {
            if (!this->m_fpga->copyToFpga(this->m_instrBufHandle, this->PAGE_SIZE * 2)) {
                l_status = XFBLAS_STATUS_ALLOC_FAILED;
            }
            this->m_fpga->execKernel(this->m_instrBufHandle, this->m_kernel[0]);
            m_execControl = false;
        }
        return l_status;
    }

    void enableRun() { m_execControl = true; }
};

} // namespace blas

} // namespace xf

#endif
