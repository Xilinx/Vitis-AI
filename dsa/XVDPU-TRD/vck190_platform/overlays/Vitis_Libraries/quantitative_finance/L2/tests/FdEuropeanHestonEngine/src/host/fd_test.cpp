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
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "fd_util.hpp"
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

using namespace fd;

// Temporary copy of this macro definition until new xcl2.hpp is used
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

int main(int argc, char* argv[]) {
    // Sanity check the input parameters
    if (argc != 6) {
        std::cout << "ERROR: passed " << argc << " arguments instead of 8, exiting" << std::endl;
        std::cout << "  Usage:" << std::endl;
        std::cout << "    fd_test.exe <path/fd.xclbin> path_to_param_files M1 M2 N" << std::endl;
        std::cout << "  Example:" << std::endl;
        std::cout << "    fd_test.exe path-to-xclbin/fd.xclbin ./data 100 50 200" << std::endl;
        return EXIT_FAILURE;
    }
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // Parse arguments
    unsigned int argIdx = 1;
    string xclbin_file(argv[argIdx++]);
    string path_raw(argv[argIdx++]);
    unsigned int M1 = atoi(argv[argIdx++]);
    unsigned int M2 = atoi(argv[argIdx++]);
    unsigned int N = atoi(argv[argIdx++]);

    // Assemble directory name of reference data based on parameters
    std::stringstream sstm;
    sstm << path_raw << "/ref_" << M1 << "x" << M2 << "_N" << N;
    string path = sstm.str();

    // Reference vector/array sizes based on grid size
    const unsigned int m_size = FD_M_SIZE;
    const unsigned int a_size = m_size * 10;
    const unsigned int a1_size = m_size * 3; // Guaranteed to fit in integer
                                             // number of DDR words regardless of
                                             // data type
    const unsigned int a2_size = m_size * 5; // Guaranteed to fit in integer
                                             // number of DDR words regardless of
                                             // data type

    // Vectors for parameter storage.  These use an aligned allocator in order
    // to avoid an additional copy of the host memory into the device
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > A(a_size);
    std::vector<unsigned int, aligned_allocator<unsigned int> > A_row(a_size);
    std::vector<unsigned int, aligned_allocator<unsigned int> > A_col(a_size);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > A1(a1_size);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > X1(a1_size);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > A2(a2_size);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > X2(a2_size);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > b(m_size);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > u0(m_size);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > price(m_size);
    std::vector<FD_DATA_TYPE, aligned_allocator<FD_DATA_TYPE> > ref(m_size);

    // Sparse array non-zero count
    unsigned int A_nnz;

    // Size of data and index vectors padded to fill whole 512-bit DDR word
    unsigned int A_pad;
    unsigned int Arc_pad;

    // Read precomputed array data and golden reference from .csv files
    std::cout << "Loading precomputed data from Python reference model..." << std::endl;
    FdUtil<FD_DATA_TYPE, FD_DATA_WORDS_IN_DDR> util;
    if (util.ReadSparse(path + "/A.csv", A, A_row, A_col, A_nnz, A_pad, Arc_pad, a_size)) return EXIT_FAILURE;
    if (util.ReadDiag3(path + "/A1.csv", A1, m_size)) return EXIT_FAILURE;
    if (util.ReadDiag5(path + "/A2.csv", A2, m_size)) return EXIT_FAILURE;
    if (util.ReadDiag3(path + "/X1.csv", X1, m_size)) return EXIT_FAILURE;
    if (util.ReadDiag5(path + "/X2.csv", X2, m_size)) return EXIT_FAILURE;
    if (util.ReadVector(path + "/b.csv", b, m_size)) return EXIT_FAILURE;
    if (util.ReadVector(path + "/u0.csv", u0, m_size)) return EXIT_FAILURE;
    if (util.ReadVector(path + "/ref.csv", ref, m_size)) return EXIT_FAILURE;

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl_int err;

    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // Load the binary file (using function from xcl2.cpp)
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_file);

    devices.resize(1);
    cl::Program program(context, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel krnl_fd_heston(program, "fd_kernel", &err);
    logger.logCreateKernel(err);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, A_pad * sizeof(FD_DATA_TYPE),
                                       A.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_A_row(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           Arc_pad * sizeof(unsigned int), A_row.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_A_col(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           Arc_pad * sizeof(unsigned int), A_col.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_A1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, a1_size * sizeof(FD_DATA_TYPE),
                                        A1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_A2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, a2_size * sizeof(FD_DATA_TYPE),
                                        A2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_X1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, a1_size * sizeof(FD_DATA_TYPE),
                                        X1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_X2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, a2_size * sizeof(FD_DATA_TYPE),
                                        X2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_b(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, m_size * sizeof(FD_DATA_TYPE),
                                       b.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_u0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, m_size * sizeof(FD_DATA_TYPE),
                                        u0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_price(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                           m_size * sizeof(FD_DATA_TYPE), price.data(), &err));

    // Set the arguments
    OCL_CHECK(err, err = krnl_fd_heston.setArg(0, buffer_A));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(1, buffer_A_row));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(2, buffer_A_col));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(3, A_nnz));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(4, buffer_A1));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(5, buffer_A2));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(6, buffer_X1));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(7, buffer_X2));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(8, buffer_b));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(9, buffer_u0));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(10, M1));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(11, M2));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(12, N));
    OCL_CHECK(err, err = krnl_fd_heston.setArg(13, buffer_price));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A_row}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A_col}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A1}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_X1}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A2}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_X2}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_u0}, 0));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_fd_heston));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_price}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    // OPENCL HOST CODE AREA END

    // Calculate absolute worst difference across whole grid compared to reference
    FD_DATA_TYPE MaxDiff = util.CompareReference(price, ref, m_size);

    int ret = 0;
    if (std::abs(MaxDiff) > 8e34) {
        std::cout << "FAIL: MaxDiff = " << MaxDiff << std::endl;
        ret = 1;
    }

    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return ret;
}
