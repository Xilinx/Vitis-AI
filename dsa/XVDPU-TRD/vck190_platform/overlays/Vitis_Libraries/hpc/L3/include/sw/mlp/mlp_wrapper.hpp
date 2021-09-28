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

#ifndef XF_HPC_MLP_WRAPPER_HPP
#define XF_HPC_MLP_WRAPPER_HPP

#include "host.hpp" //from xf::blas

#include "fcn_host.hpp"
#include "handle.hpp"
#include <future>

namespace xf {
namespace hpc {
namespace mlp {

typedef xf::blas::XFpga XFpga;
typedef xf::blas::XFpgaHold XFpgaHold;
typedef xf::blas::xfblasStatus_t xfblasStatus_t;

/**
 * @brief This function initializes the XFHPC library and creates a handle for the specific engine. It must be called
 * prior to any other XFHPC library calls.
 * @param xclbin file path to FPGA bitstream
 * @param kernelNumber number of kernels that is being used, default is 1
 * @param deviceIndex index of device that is being used, default is 0
 * @retval bool
 */
bool xfhpcCreate(const char* xclbin, unsigned int kernelNumber = 1, unsigned int deviceIndex = 0) {
    int l_err = 0;

    shared_ptr<XFpga> l_xFpga(new XFpga(xclbin, &l_err, deviceIndex));
    XFpgaHold::instance().m_xFpgaPtr[deviceIndex] = l_xFpga;

    if (l_err != 0) {
        return false;
    }

    for (unsigned int i = 0; i < kernelNumber; i++) {
        HPCHostHandle::instance().m_handlePtr[deviceIndex].push_back(
            shared_ptr<xf::blas::BLASHost>(new FCNHost(xclbin, nullptr, i, deviceIndex)));
    }
    return true;
}

/**
 * @brief This function allocates memory for host row-major format matrix on the FPGA device.
 * @param rows number of rows in the matrix
 * @param cols number of cols in the matrix that is being used
 * @param elemSize number of bytes required to store each element in the matrix
 * @param A pointer to the matrix array in the host memory
 * @param lda leading dimension of the matrix that indicates the total number of cols in the matrix
 * @param kernelIndex index of kernel that is being used, default is 0
 * @param deviceIndex index of device that is being used, default is 0
 * @retval bool
 */
bool xfhpcMalloc(
    int rows, int cols, int elemSize, void* A, int lda, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0) {
    unsigned long long l_bufSize = rows * lda * elemSize;
    xfblasStatus_t l_status =
        HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->allocMatRestricted(A, A, l_bufSize);

    if (l_status == 0) {
        return true;
    } else {
        return false;
    }
}

/**
 * @brief This function copies a matrix in host memory to FPGA device memory. xfhpcMallocRestricted() need to be called
 * prior to this function.
 * @param A pointer to the matrix array in the host memory
 * @param kernelIndex index of kernel that is being used, default is 0
 * @param deviceIndex index of device that is being used, default is 0
 * @retval bool
 */
bool xfhpcSetMatrix(void* A, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0) {
    xfblasStatus_t l_status =
        HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->setMatToFPGARestricted(A);
    if (l_status == 0) {
        return true;
    } else {
        return false;
    }
}

/**
 * @brief This function copies a matrix in FPGA device memory to host memory
 * @param A pointer to matrix A in the host memory
 * @param kernelIndex index of kernel that is being used, default is 0
 * @param deviceIndex index of device that is being used, default is 0
 * @retval bool
 */
bool xfhpcGetMatrix(void* A, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0) {
    xfblasStatus_t l_status = HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->execute();
    l_status = HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->getMatRestricted(A, A);
    if (l_status == 0) {
        return true;
    } else {
        return false;
    }
}

/**
 * @brief This function frees memory in FPGA device.
 * @param A pointer to matrix A in the host memory
 * @param kernelIndex index of kernel that is being used, default is 0
 * @param deviceIndex index of device that is being used, default is 0
 * @retval bool
 */
bool xfhpcFree(void* A, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0) {
    xfblasStatus_t l_status = HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->freeMat(A);
    if (l_status == 0) {
        return true;
    } else {
        return false;
    }
}

/**
 * @brief This function frees instrution
 * @param kernelIndex index of kernel that is being used, default is 0
 * @param deviceIndex index of device that is being used, default is 0
 */
void xfhpcFreeInstr(unsigned int kernelIndex = 0, unsigned int deviceIndex = 0) {
    HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->clearInstrBuf();
}

/**
 * @brief This function releases handle used by the XFHPC library.
 * @param kernelNumber number of kernels that is being used, default is 1
 * @param deviceIndex index of device that is being used, default is 0
 */
void xfhpcDestroy(unsigned int kernelNumber = 1, unsigned int deviceIndex = 0) {
    for (unsigned int i = 0; i < kernelNumber; i++) {
        HPCHostHandle::instance().m_handlePtr[deviceIndex][i]->clearInstrBuf();
        HPCHostHandle::instance().m_handlePtr[deviceIndex][i]->closeContext(i);
    }
    // HPCHostHandle::instance().m_handlePtr[deviceIndex][0]->closeDevice();
    XFpgaHold::instance().m_xFpgaPtr.clear();
    HPCHostHandle::instance().m_handlePtr.clear();
}

bool xfhpcFcn(int m,
              int n,
              int k,
              int alpha,
              void* A,
              int lda,
              void* B,
              int ldb,
              int beta,
              void* C,
              int ldc,
              void* X,
              int ldx,
              int p_postScale,
              int p_postShift,
              short p_preluScale,
              short p_preluAlpha,
              unsigned int kernelIndex = 0,
              unsigned int deviceIndex = 0) {
    FCNHost* l_fcnPtr = static_cast<FCNHost*>(HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex].get());
    bool l_check = l_fcnPtr->addFCNOp(A, B, C, X, m, n, k, lda, ldb, ldc, ldx, p_postScale, p_postShift, p_preluScale,
                                      p_preluAlpha);
    return l_check;
}

/**
 * @brief This function starts the kernel and wait until it finishes
 * @param kernelIndex index of kernel that is being used, default is 0
 * @param deviceIndex index of device that is being used, default is 0
 * @retval bool
 */
bool xfhpcExecute(unsigned int kernelIndex = 0, unsigned int deviceIndex = 0) {
    xfblasStatus_t l_status = HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->execute();
    if (l_status == 0) {
        return true;
    } else {
        return false;
    }
}

/**
 * @brief This asynchronous function starts all kernels and wait until them finish
 * @param numKernels number of kernels that is being used, default is 1
 * @param deviceIndex index of device that is being used, default is 0
 */
void xfhpcExecuteAsync(unsigned int numKernels = 1, unsigned int deviceIndex = 0) {
    vector<future<bool> > fuStatus;
    for (unsigned int i = 0; i < numKernels; i++) {
        fuStatus.push_back(async(launch::async, xfhpcExecute, i, deviceIndex));
    }
    for (auto& fu : fuStatus) {
        fu.wait();
    }
}

/**
 * @brief This function copies a matrix in FPGA device memory to host memory by its address in device memory
 * @param A pointer to matrix A in the host memory
 * @param p_bufSize size of matrix A
 * @param offset A's address in device memory
 * @param kernelIndex index of kernel that is being used, default is 0
 * @param deviceIndex index of device that is being used, default is 0
 * @retval bool
 */
bool xfhpcGetByAddress(void* A,
                       unsigned long long p_bufSize,
                       unsigned int offset,
                       unsigned int kernelIndex = 0,
                       unsigned int deviceIndex = 0) {
    xfblasStatus_t l_status =
        HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->getMatByAddress(A, p_bufSize, offset);
    if (l_status == 0) {
        return true;
    } else {
        return false;
    }
}

bool xfhpcGetByPointer(void* A, unsigned int kernelIndex = 0, unsigned int deviceIndex = 0) {
    xfblasStatus_t l_status = HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->getMatRestricted(A, A);
    if (l_status == 0) {
        return true;
    } else {
        return false;
    }
}

void xfhpcFcnByAddress(unsigned int l_aOff,
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
                       short p_preluAlpha,
                       unsigned int kernelIndex = 0,
                       unsigned int deviceIndex = 0) {
    FCNHost* l_fcnPtr = static_cast<FCNHost*>(HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex].get());
    l_fcnPtr->addFCNOpByAddress(l_aOff, l_bOff, l_cOff, l_xOff, p_m, p_n, p_k, p_lda, p_ldb, p_ldc, p_ldx, p_postScale,
                                p_postShift, p_preluScale, p_preluAlpha);
}

} // namespace mlp
} // namespace hpc
} // namespace xf

#endif