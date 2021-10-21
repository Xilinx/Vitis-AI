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

#include <stdio.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "xcl2.hpp"

#include "xf_fintech_heston_kernel_interface.hpp"

#include "xf_fintech_heston_kernel_constants.hpp"

using namespace std;
// using namespace xf::fintech::hestonfd;

// Temporary copy of this macro definition until new xcl2.hpp is used
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

namespace xf {
namespace fintech {
namespace hestonfd {

//#define KERNEL_DEBUG

void kernel_call(std::map<std::pair<int, int>, double>& sparse_map_A,
                 std::vector<std::vector<double> >& A1_vec,
                 std::vector<std::vector<double> >& A2_vec,
                 std::vector<std::vector<double> >& X1_vec,
                 std::vector<std::vector<double> >& X2_vec,
                 std::vector<double>& b_vec,
                 std::vector<double>& u0_vec,
                 int M1,
                 int M2,
                 int N,
                 double* price_grid) {
    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl_int err;
    string xclbin_file = "fd_heston_kernel_u200_hw_m8192_double.xclbin";

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    // Load the binary file (using function from xcl2.cpp)
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_file);

    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel krnl_fd_heston(program, "fd_kernel", &err));

    kernel_call(&context, &q, &krnl_fd_heston, sparse_map_A, A1_vec, A2_vec, X1_vec, X2_vec, b_vec, u0_vec, M1, M2, N,
                price_grid);
}

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
                 double* price_grid) {
    cl_int err;

    // Reference vector/array sizes based on grid size
    const unsigned int M = FD_mSize;
    const unsigned int a_size = M * 10;
    const unsigned int a1_size = M * 3; // Guaranteed to fit in integer number of DDR words
    const unsigned int a2_size = M * 5; // regardless of data type for any sensible M

    // Vectors for parameter storage.  These use an aligned allocator in order
    // to avoid an additional copy of the host memory into the device
    vector<FD_dataType, aligned_allocator<FD_dataType> > A(a_size);
    vector<unsigned int, aligned_allocator<unsigned int> > Ar(a_size);
    vector<unsigned int, aligned_allocator<unsigned int> > Ac(a_size);
    vector<FD_dataType, aligned_allocator<FD_dataType> > A1(a1_size);
    vector<FD_dataType, aligned_allocator<FD_dataType> > X1(a1_size);
    vector<FD_dataType, aligned_allocator<FD_dataType> > A2(a2_size);
    vector<FD_dataType, aligned_allocator<FD_dataType> > X2(a2_size);
    vector<FD_dataType, aligned_allocator<FD_dataType> > b(M);
    vector<FD_dataType, aligned_allocator<FD_dataType> > u0(M);
    vector<FD_dataType, aligned_allocator<FD_dataType> > price(M);
    vector<FD_dataType, aligned_allocator<FD_dataType> > ref(M);

    // Sparse array non-zero count
    unsigned int A_nnz;

    // Size of data and index vectors padded to fill whole 512-bit DDR word
    unsigned int A_pad;
    unsigned int Arc_pad;

    unsigned int i = 0;
    for (auto elem : sparse_map_A) {
        Ar[i] = (unsigned int)elem.first.first;
        Ac[i] = (unsigned int)elem.first.second;
        A[i] = (FD_dataType)elem.second;
        i++;
    }
    A_nnz = sparse_map_A.size();

    // Need to pad the A array and row/column arrays so they fit into DDR word
    // Different amounts of padding needed depending on width of data
    A_pad = A_nnz;
    Arc_pad = A_nnz;
    while (A_pad % (64 / sizeof(FD_dataType)) != 0) {
        A[A_pad++] = 0;
    }
    while (Arc_pad % (64 / sizeof(unsigned int)) != 0) {
        Ar[Arc_pad] = 0;
        Ac[Arc_pad] = 0;
        Arc_pad++;
    }

    unsigned int row, col;
    i = 0;
    for (row = 0; row < 3; row++) {
        for (col = 0; col < M; col++) {
            A1[i] = (FD_dataType)A1_vec[col][row];
            X1[i] = (FD_dataType)X1_vec[col][row];
            i++;
        }
    }

    i = 0;
    for (row = 0; row < 5; row++) {
        for (col = 0; col < M; col++) {
            A2[i] = (FD_dataType)A2_vec[col][row];
            X2[i] = (FD_dataType)X2_vec[col][row];
            i++;
        }
    }

    for (i = 0; i < M; i++) {
        u0[i] = (FD_dataType)u0_vec.at(i);
        b[i] = (FD_dataType)b_vec.at(i);
    }

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_A(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, A_pad * sizeof(FD_dataType),
                                       A.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_A_row(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           Arc_pad * sizeof(unsigned int), Ar.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_A_col(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           Arc_pad * sizeof(unsigned int), Ac.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_A1(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        a1_size * sizeof(FD_dataType), A1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_A2(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        a2_size * sizeof(FD_dataType), A2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_X1(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        a1_size * sizeof(FD_dataType), X1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_X2(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        a2_size * sizeof(FD_dataType), X2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_b(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, M * sizeof(FD_dataType),
                                       b.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_u0(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, M * sizeof(FD_dataType),
                                        u0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_price(*pContext, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, M * sizeof(FD_dataType),
                                           price.data(), &err));

    // Set the arguments
    OCL_CHECK(err, err = pKernel->setArg(0, buffer_A));
    OCL_CHECK(err, err = pKernel->setArg(1, buffer_A_row));
    OCL_CHECK(err, err = pKernel->setArg(2, buffer_A_col));
    OCL_CHECK(err, err = pKernel->setArg(3, A_nnz));
    OCL_CHECK(err, err = pKernel->setArg(4, buffer_A1));
    OCL_CHECK(err, err = pKernel->setArg(5, buffer_A2));
    OCL_CHECK(err, err = pKernel->setArg(6, buffer_X1));
    OCL_CHECK(err, err = pKernel->setArg(7, buffer_X2));
    OCL_CHECK(err, err = pKernel->setArg(8, buffer_b));
    OCL_CHECK(err, err = pKernel->setArg(9, buffer_u0));
    OCL_CHECK(err, err = pKernel->setArg(10, M1));
    OCL_CHECK(err, err = pKernel->setArg(11, M2));
    OCL_CHECK(err, err = pKernel->setArg(12, N));
    OCL_CHECK(err, err = pKernel->setArg(13, buffer_price));

    // Copy input data to device global memory
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_A}, 0));
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_A_row}, 0));
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_A_col}, 0));
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_A1}, 0));
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_X1}, 0));
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_A2}, 0));
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_X2}, 0));
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_b}, 0));
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_u0}, 0));

    // Launch the Kernel
    OCL_CHECK(err, err = pCommandQueue->enqueueTask(*pKernel));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = pCommandQueue->enqueueMigrateMemObjects({buffer_price}, CL_MIGRATE_MEM_OBJECT_HOST));
    pCommandQueue->finish();
    // OPENCL HOST CODE AREA END

    // Return the price grid
    for (i = 0; i < M; ++i) price_grid[i] = price[i];
}

} // namespace hestonfd
} // namespace fintech
} // namespace xf
