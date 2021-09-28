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
 * @file quanto_test.cpp
 * @brief Testbench to generate randomized input data and launch on kernel.
 * Results are compared to a full precision model.
 */

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include "quanto_host.hpp"
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

/// @def Controls the data type used in the kernel
#define KERNEL_DT float

// Temporary copy of this macro definition until new xcl2.hpp is used
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

static int check_exp = 0;

static void usage(std::string exe) {
    std::cout << "Usage: " << exe << " ./xclbin/<kernel_name> <test data file> [exp]" << std::endl;
    std::cout << "if 3rd parameter = exp then the calculated value is compared with the expected value" << std::endl;
    std::cout << "Test data file line format:" << std::endl;
    std::cout << "s=<>, k=<>, r_domestic=<>, r_foreign=<>, v=<>, t=<>, q=<>, E=<>, fxv=<>, corr=<> [exp=<>]"
              << std::endl;
    std::cout << "# comments out the line" << std::endl;
}

static int validate_parameters(int argc, char* argv[]) {
    /* check 2 arguments specified */
    if (argc != 3 && argc != 4) {
        usage(argv[0]);
        return 0;
    }

    /* check xclbin file exists */
    std::ifstream ifs1(argv[1]);
    if (!ifs1.is_open()) {
        std::cout << "ERROR: cannot open " << argv[1] << std::endl;
        return 0;
    }

    /* check test data file exists */
    std::ifstream ifs2(argv[2]);
    if (!ifs2.is_open()) {
        std::cout << "ERROR: cannot open " << argv[2] << std::endl;
        return 0;
    }

    if (argc == 4 && std::string(argv[3]).compare("exp") == 0) {
        check_exp = 1;
    }

    return 1;
}

/// @brief Main entry point to test
///
/// This is a command-line application to test the kernel.  It supports software
/// and hardware emulation as well as
/// running on an Alveo target.
///
/// Usage: ./quanto_test ./xclbin/<kernel_name> <test data file>
///
/// @param[in] argc Standard C++ argument count
/// @param[in] argv Standard C++ input arguments
int main(int argc, char* argv[]) {
    std::cout << std::endl << std::endl;
    std::cout << "*********************" << std::endl;
    std::cout << "Quanto call Demo v1.0" << std::endl;
    std::cout << "*********************" << std::endl;
    std::cout << std::endl;

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    if (!validate_parameters(argc, argv)) {
        exit(1);
    }

    // parse the input test data file
    std::vector<struct parsed_params*>* vect = parse_file(argv[2]);
    if (vect == nullptr) {
        return 1;
    }
    unsigned int num = vect->size();
    // kernel expects multiple of 16 input data
    unsigned int remainder = num % 16;
    unsigned int extra = 0;
    if (remainder != 0) {
        extra = 16 - remainder;
    }
    unsigned int modified_num = num + extra;

    // Test parameters
    static const unsigned int call = 1;
    std::string xclbin_file(argv[1]);

    // Vectors for parameter storage.  These use an aligned allocator in order
    // to avoid an additional copy of the host memory into the device
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > s(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > v(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > rd(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > rf(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > t(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > k(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > q(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > E(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > fxv(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > corr(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > price(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > delta(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > gamma(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > vega(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > theta(modified_num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > rho(modified_num);

    // Host results (always double precision)
    double* host_price = new double[modified_num];
    double* host_delta = new double[modified_num];
    double* host_gamma = new double[modified_num];
    double* host_vega = new double[modified_num];
    double* host_theta = new double[modified_num];
    double* host_rho = new double[modified_num];

    // write the test data to the input vectors and calculate the model results
    std::cout << "Generating reference results..." << std::endl;
    for (unsigned int i = 0; i < num; i++) {
        s[i] = vect->at(i)->s;
        v[i] = vect->at(i)->v;
        t[i] = vect->at(i)->t;
        k[i] = vect->at(i)->k;
        q[i] = vect->at(i)->q;
        E[i] = vect->at(i)->E;
        fxv[i] = vect->at(i)->fxv;
        corr[i] = vect->at(i)->corr;
        rd[i] = vect->at(i)->rd;
        rf[i] = vect->at(i)->rf;
    }
    for (unsigned int i = num; i < modified_num; i++) {
        s[i] = 0;
        v[i] = 0;
        t[i] = 0;
        k[i] = 0;
        q[i] = 0;
        E[i] = 0;
        fxv[i] = 0;
        corr[i] = 0;
        rd[i] = 0;
        rf[i] = 0;
    }

    std::cout << "Running CPU model..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < num; i++) {
        quanto_model(s[i], v[i], rd[i], t[i], k[i], rf[i], q[i], E[i], fxv[i], corr[i], call, host_price[i],
                     host_delta[i], host_gamma[i], host_vega[i], host_theta[i], host_rho[i]);
    }
    auto cpu_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::cout << "Connecting to device and loading kernel..." << std::endl << std::flush;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl_int err;

    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue cq(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // Load the binary file (using function from xcl2.cpp)
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_file);

    devices.resize(1);
    cl::Program program(context, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel krnl_cfGKEngine(program, "quanto_kernel", &err);
    logger.logCreateKernel(err);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    std::cout << "Allocating buffers..." << std::endl;
    OCL_CHECK(err, cl::Buffer buffer_s(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                       modified_num * sizeof(KERNEL_DT), s.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_v(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                       modified_num * sizeof(KERNEL_DT), v.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_rd(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        modified_num * sizeof(KERNEL_DT), rd.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_t(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                       modified_num * sizeof(KERNEL_DT), t.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_k(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                       modified_num * sizeof(KERNEL_DT), k.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_rf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        modified_num * sizeof(KERNEL_DT), rf.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_q(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                       modified_num * sizeof(KERNEL_DT), q.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_E(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                       modified_num * sizeof(KERNEL_DT), E.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_fxv(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                         modified_num * sizeof(KERNEL_DT), fxv.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_corr(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          modified_num * sizeof(KERNEL_DT), corr.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_price(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                           modified_num * sizeof(KERNEL_DT), price.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_delta(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                           modified_num * sizeof(KERNEL_DT), delta.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_gamma(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                           modified_num * sizeof(KERNEL_DT), gamma.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_vega(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                          modified_num * sizeof(KERNEL_DT), vega.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_theta(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                           modified_num * sizeof(KERNEL_DT), theta.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_rho(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                         modified_num * sizeof(KERNEL_DT), rho.data(), &err));

    // Set the arguments
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(0, buffer_s));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(1, buffer_v));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(2, buffer_rd));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(3, buffer_t));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(4, buffer_k));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(5, buffer_rf));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(6, buffer_q));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(7, buffer_E));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(8, buffer_fxv));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(9, buffer_corr));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(10, call));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(11, modified_num));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(12, buffer_price));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(13, buffer_delta));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(14, buffer_gamma));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(15, buffer_vega));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(16, buffer_theta));
    OCL_CHECK(err, err = krnl_cfGKEngine.setArg(17, buffer_rho));

    // Copy input data to device global memory
    t_start = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_s}, 0));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_v}, 0));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_rd}, 0));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_t}, 0));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_k}, 0));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_rf}, 0));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_q}, 0));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_E}, 0));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_fxv}, 0));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_corr}, 0));

    // Launch the Kernel
    std::cout << "Launching kernel..." << std::endl;
    uint64_t nstimestart, nstimeend;
    cl::Event event;
    OCL_CHECK(err, err = cq.enqueueTask(krnl_cfGKEngine, NULL, &event));
    OCL_CHECK(err, err = cq.finish());
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto duration_nanosec = nstimeend - nstimestart;

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_price}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_delta}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_gamma}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_vega}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_theta}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_rho}, CL_MIGRATE_MEM_OBJECT_HOST));
    cq.finish();
    auto fpga_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();
    // OPENCL HOST CODE AREA END

    std::cout << "Kernel done!" << std::endl;
    std::cout << "Comparing results..." << std::endl;
    // Check results
    double max_price_diff = 0.0f;
    double max_delta_diff = 0.0f;
    double max_gamma_diff = 0.0f;
    double max_vega_diff = 0.0f;
    double max_theta_diff = 0.0f;
    double max_rho_diff = 0.0f;

    int total_tests = 0;
    int total_fails = 0;
    double tolerance = 0.00001;
    for (unsigned int i = 0; i < num; i++) {
        double temp = 0.0f;
        if (std::abs(temp = (price[i] - host_price[i])) > std::abs(max_price_diff)) max_price_diff = temp;
        if (std::abs(temp = (delta[i] - host_delta[i])) > std::abs(max_delta_diff)) max_delta_diff = temp;
        if (std::abs(temp = (gamma[i] - host_gamma[i])) > std::abs(max_gamma_diff)) max_gamma_diff = temp;
        if (std::abs(temp = (vega[i] - host_vega[i])) > std::abs(max_vega_diff)) max_vega_diff = temp;
        if (std::abs(temp = (theta[i] - host_theta[i])) > std::abs(max_theta_diff)) max_theta_diff = temp;
        if (std::abs(temp = (rho[i] - host_rho[i])) > std::abs(max_rho_diff)) max_rho_diff = temp;

        if (check_exp) {
            total_tests++;
            if (std::abs(price[i] - vect->at(i)->exp) > tolerance) {
                total_fails++;
                std::cout << "FAIL[test " << i << "]: expected(" << vect->at(i)->exp << "), got(" << price[i] << ")";
                std::cout << " cpu(" << host_price[i] << ")";
                std::cout << std::endl;
                std::cout << "s=" << vect->at(i)->s << " k=" << vect->at(i)->k;
                std::cout << " v=" << vect->at(i)->v << " q=" << vect->at(i)->q;
                std::cout << " t=" << vect->at(i)->t << " rd=" << vect->at(i)->rd << " rf=" << vect->at(i)->rf;
                std::cout << " E=" << vect->at(i)->E << " fxv=" << vect->at(i)->fxv << " corr=" << vect->at(i)->corr;
                std::cout << std::endl << std::endl;
            }
        }
    }

    std::cout << "Processed " << num;
    if (call) {
        std::cout << " call options:" << std::endl;
    } else {
        std::cout << " put options:" << std::endl;
    }
    std::cout << "Throughput = " << (1.0 * num) / (duration_nanosec * 1.0e-9) / 1.0e6 << " Mega options/sec"
              << std::endl;

    std::cout << std::endl;
    std::cout << "  Largest host-kernel price difference = " << max_price_diff << std::endl;
    std::cout << "  Largest host-kernel delta difference = " << max_delta_diff << std::endl;
    std::cout << "  Largest host-kernel gamma difference = " << max_gamma_diff << std::endl;
    std::cout << "  Largest host-kernel vega difference  = " << max_vega_diff << std::endl;
    std::cout << "  Largest host-kernel theta difference = " << max_theta_diff << std::endl;
    std::cout << "  Largest host-kernel rho difference   = " << max_rho_diff << std::endl;

    std::cout << "CPU execution time                          = " << cpu_duration << "us" << std::endl;
    std::cout << "FPGA time returned by profile API           = " << (duration_nanosec * (1.0e-6)) << " ms"
              << std::endl;
    std::cout << "FPGA execution time (including mem transfer)= " << fpga_duration << "us" << std::endl;

    if (check_exp) {
        std::cout << "Total tests = " << total_tests << std::endl;
        std::cout << "Total fails = " << total_fails << std::endl;
        std::cout << "Tolerance   = " << tolerance << std::endl;
    }

    int ret = 0;
    if (std::abs(max_price_diff) > 5.0e-7) {
        std::cout << "FAIL: max_price_diff = " << max_price_diff << std::endl;
        ret = 1;
    }
    if (std::abs(max_delta_diff) > 6.0e-8) {
        std::cout << "FAIL: max_delta_diff = " << max_delta_diff << std::endl;
        ret = 1;
    }
    if (std::abs(max_gamma_diff) > 10.0e-11) {
        std::cout << "FAIL: max_gamma_diff = " << max_gamma_diff << std::endl;
        ret = 1;
    }
    if (std::abs(max_vega_diff) > 5.0e-8) {
        std::cout << "FAIL: max_vega_diff = " << max_vega_diff << std::endl;
        ret = 1;
    }
    if (std::abs(max_theta_diff) > 4.0e-9) {
        std::cout << "FAIL: max_theta_diff = " << max_theta_diff << std::endl;
        ret = 1;
    }
    if (std::abs(max_rho_diff) > 3.0e-7) {
        std::cout << "FAIL: max_rho_diff = " << max_rho_diff << std::endl;
        ret = 1;
    }

    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return ret;
}
