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

#ifndef XF_BLAS_API_HPP
#define XF_BLAS_API_HPP

extern "C" {

bool xfblasCreate(char* xclbin, char* engineName, unsigned int kernelNumber, unsigned int deviceIndex);
bool xfblasSend(void* A, unsigned long long numElem, int elemSize, unsigned int kernelIndex, unsigned int deviceIndex);
bool xfblasGet(void* A, unsigned int kernelIndex, unsigned int deviceIndex);
void xfblasFreeInstr(unsigned int kernelIndex, unsigned int deviceIndex);
void xfblasDestroy(unsigned int kernelNumber, unsigned int deviceIndex);
void xfblasFree(void* A, unsigned int kernelIndex, unsigned int deviceIndex);
bool xfblasGemm(int m,
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
                unsigned int kernelIndex,
                unsigned int deviceIndex);
bool xfblasGemv(int m,
                int n,
                int alpha,
                void* A,
                int lda,
                void* x,
                int incx,
                int beta,
                void* y,
                int incy,
                unsigned int kernelIndex,
                unsigned int deviceIndex);
bool xfblasGetByAddress(
    void* A, unsigned long long p_bufSize, unsigned int offset, unsigned int kernelIndex, unsigned int deviceIndex);
void xfblasExecuteAsync(unsigned int numkernels, unsigned int deviceIndex);
bool xfblasExecute(unsigned int kernelIndex, unsigned int deviceIndex);
}

#endif