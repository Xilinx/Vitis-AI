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

#ifndef _XF_FINTECH_HESTON_KERNEL_INERFACE_H_
#define _XF_FINTECH_HESTON_KERNEL_INERFACE_H_

namespace xf {
namespace fintech {
namespace hestonfd {

void kernel_call(std::map<std::pair<int, int>, double>& sparse_map_A,
                 std::vector<std::vector<double> >& A1_vec,
                 std::vector<std::vector<double> >& A2_vec,
                 std::vector<std::vector<double> >& X1,
                 std::vector<std::vector<double> >& X2,
                 std::vector<double>& b,
                 std::vector<double>& u0,
                 int M1,
                 int M2,
                 int N,
                 double* price);

void kernel_call(cl::Context* pContext,
                 cl::CommandQueue* pCommandQueue,
                 cl::Kernel* pKernel,
                 std::map<std::pair<int, int>, double>& sparse_map_A,
                 std::vector<std::vector<double> >& A1_vec,
                 std::vector<std::vector<double> >& A2_vec,
                 std::vector<std::vector<double> >& X1_vec,
                 std::vector<std::vector<double> >& X2_vec,
                 std::vector<double>& b_vec,
                 std::vector<double>& u0_vec,
                 int M1,
                 int M2,
                 int N,
                 double* price_grid);

} // namespace hestonfd
} // namespace fintech
} // namespace xf

#endif
