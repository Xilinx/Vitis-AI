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

#include <omp.h>
#include "host.hpp" //from xf::blas

#include "fcn_host.hpp"
#include "handle.hpp"

#include "api.hpp"

using namespace xf::hpc::mlp;
typedef xf::blas::XFpga XFpga;
typedef xf::blas::XFpgaHold XFpgaHold;
typedef xf::blas::xfblasStatus_t xfblasStatus_t;

bool xfhpcCreate(char* xclbin, unsigned int kernelNumber, unsigned int deviceIndex) {
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

bool xfhpcSend(void* A, unsigned long long numElem, int elemSize, unsigned int kernelIndex, unsigned int deviceIndex) {
    unsigned long long l_bufSize = numElem * elemSize;
    xfblasStatus_t l_status =
        HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->allocMatRestricted(A, A, l_bufSize);
    if (l_status != 0) {
        return false;
    }
    l_status = HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->setMatToFPGARestricted(A);
    if (l_status != 0) {
        return false;
    }
    return true;
}

bool xfhpcGet(void* A, unsigned int kernelIndex, unsigned int deviceIndex) {
    xfblasStatus_t l_status = HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->execute();
    if (l_status != 0) {
        return false;
    }
    l_status = HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->getMatRestricted(A, A);
    if (l_status != 0) {
        return false;
    }
    return true;
}

bool xfhpcGetByAddress(
    void* A, unsigned long long p_bufSize, unsigned int offset, unsigned int kernelIndex, unsigned int deviceIndex) {
    xfblasStatus_t l_status =
        HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->getMatByAddress(A, p_bufSize, offset);
    if (l_status != 0) {
        return false;
    }
    return true;
}

bool xfhpcExecute(unsigned int kernelIndex, unsigned int deviceIndex) {
    xfblasStatus_t l_status = HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->execute();
    if (l_status != 0) {
        return false;
    }
    return true;
}

void xfhpcExecuteAsync(unsigned int numkernels, unsigned int deviceIndex) {
#pragma omp parallel
    {
        omp_set_dynamic(0);
        omp_set_num_threads(numkernels);
#pragma omp for
        for (unsigned int i = 0; i < numkernels; i++) {
            HPCHostHandle::instance().m_handlePtr[deviceIndex][i]->execute();
        }
    }
}

void xfhpcFreeInstr(unsigned int kernelIndex, unsigned int deviceIndex) {
    HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->clearInstrBuf();
}

void xfhpcFree(void* A, unsigned int kernelIndex, unsigned int deviceIndex) {
    HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex]->freeMat(A);
}

void xfhpcDestroy(unsigned int kernelNumber, unsigned int deviceIndex) {
    for (unsigned int i = 0; i < kernelNumber; i++) {
        HPCHostHandle::instance().m_handlePtr[deviceIndex][i]->clearInstrBuf();
        HPCHostHandle::instance().m_handlePtr[deviceIndex][i]->closeContext(i);
    }
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
              unsigned int kernelIndex,
              unsigned int deviceIndex) {
    FCNHost* l_fcnPtr = static_cast<FCNHost*>(HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex].get());
    bool l_check = l_fcnPtr->addFCNOp(A, B, C, X, m, n, k, lda, ldb, ldc, ldx, p_postScale, p_postShift, p_preluScale,
                                      p_preluAlpha);
    return l_check;
}

bool xfhpcFcnByAddress(unsigned int l_aOff,
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
                       unsigned int kernelIndex,
                       unsigned int deviceIndex) {
    FCNHost* l_fcnPtr = static_cast<FCNHost*>(HPCHostHandle::instance().m_handlePtr[deviceIndex][kernelIndex].get());
    l_fcnPtr->addFCNOpByAddress(l_aOff, l_bOff, l_cOff, l_xOff, p_m, p_n, p_k, p_lda, p_ldb, p_ldc, p_ldx, p_postScale,
                                p_postShift, p_preluScale, p_preluAlpha);
    return true;
}
