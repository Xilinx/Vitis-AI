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
 * @file quad_test.cpp
 * @brief Testbench to launch kernel with test vectors.
 * Results are compared to a closed form integration model.
 */

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
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

struct kernel_params_type {
    KERNEL_DT a;
    KERNEL_DT b;
};

KERNEL_DT integrate_poly(KERNEL_DT a, KERNEL_DT b) {
    KERNEL_DT bb = (b * b * b * b * 0.1 / 4) + (b * b * b * 4 / 3) + (b * b * 15.5 / 2) + (33.3 * b);
    KERNEL_DT aa = (a * a * a * a * 0.1 / 4) + (a * a * a * 4 / 3) + (a * a * 15.5 / 2) + (33.3 * a);
    return bb - aa;
}

int main(int argc, char* argv[]) {
    std::cout << std::endl << std::endl;
    std::cout << "************" << std::endl;
    std::cout << "Quadrature Demo v1.0" << std::endl;
    std::cout << "************" << std::endl;
    std::cout << std::endl;

    // Test parameters
    std::string xclbin_file(argv[1]);
    KERNEL_DT integration_tolerance = 0.0001;

    struct kernel_params_type params[] = {
        {-0.1, 0.1},   {-0.2, 0.1},   {-0.3, 0.1},   {-0.4, 0.1},   {-0.5, 0.1},

        {-0.1, 0.1},   {-0.1, 0.2},   {-0.1, 0.3},   {-0.1, 0.4},   {-0.1, 0.5},

        {-10.1, -8.1}, {-10.1, -8.2}, {-10.1, -8.3}, {-10.1, -8.4}, {-10.1, -8.5},

        {0.1, 2},      {0.2, 2},      {0.3, 2},      {0.4, 2},      {0.5, 2},

        {10.1, 11},    {10.2, 12},    {10.3, 13},    {10.4, 13.5},  {10.5, 13.6},

        {-5, 0},       {-4, 0},       {-3, 0},       {-2, 0},       {-1, 0},

        {0, 5},        {0, 4},        {0, 3},        {0, 2},        {0, 1},

        {-2, 2},       {-1, 1},       {-.15, 1.5},   {-0.5, 0.5},   {-0.1, 0.1},
    };

    int num = sizeof(params) / sizeof(struct kernel_params_type);

    // Vectors for parameter storage.  These use an aligned allocator in order
    // to avoid an additional copy of the host memory into the device
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > a(num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > b(num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > method(num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > tol(num);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > res(num);

    // Host results (always double precision)
    KERNEL_DT* host_res = new KERNEL_DT[num];

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // get device
    std::cout << "Acquiring device ... " << std::endl;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // get context
    cl_int err;
    cl::Context ctx(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);

    // create command queue
    std::cout << "Creating command queue" << std::endl;
    cl::CommandQueue q(ctx, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // import and program the xclbin
    std::cout << "Programming device" << std::endl;
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_file);
    devices.resize(1);
    cl::Program program(ctx, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel krnl(program, "quad_kernel", &err);
    logger.logCreateKernel(err);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    std::cout << "Allocating buffers..." << std::endl;
    OCL_CHECK(
        err, cl::Buffer buffer_a(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, num * sizeof(KERNEL_DT), a.data(), &err));
    OCL_CHECK(
        err, cl::Buffer buffer_b(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, num * sizeof(KERNEL_DT), b.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_method(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, num * sizeof(KERNEL_DT),
                                            method.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_tol(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, num * sizeof(KERNEL_DT),
                                         tol.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_res(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, num * sizeof(KERNEL_DT),
                                         res.data(), &err));

    // generate the test vectors
    std::cout << "Generating test vectors..." << std::endl;
    for (int i = 0; i < num; i++) {
        a[i] = params[i].a;
        b[i] = params[i].b;
        tol[i] = integration_tolerance;
        method[i] = 0;
    }

    // run the host model
    for (int i = 0; i < num; i++) {
        host_res[i] = integrate_poly(params[i].a, params[i].b);
    }

    double max_diff[3];
    double mean_diff[3];

    for (int j = 0; j < 3; j++) {
        // change the integration method
        for (int i = 0; i < num; i++) {
            method[i] = j;
        }

        // Set the arguments
        OCL_CHECK(err, err = krnl.setArg(0, buffer_a));
        OCL_CHECK(err, err = krnl.setArg(1, buffer_b));
        OCL_CHECK(err, err = krnl.setArg(2, buffer_method));
        OCL_CHECK(err, err = krnl.setArg(3, buffer_tol));
        OCL_CHECK(err, err = krnl.setArg(4, num));
        OCL_CHECK(err, err = krnl.setArg(5, buffer_res));

        // Copy input data to device global memory
        std::cout << "Migrate memory to device..." << std::endl;
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_a}, 0));
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_b}, 0));
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_method}, 0));
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_tol}, 0));

        // Launch the Kernel
        std::cout << "Launching kernel..." << std::endl;
        cl::Event event;
        OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event));
        OCL_CHECK(err, err = q.finish());

        // Copy Result from Device Global Memory to Host Local Memory
        std::cout << "Migrate memory from device..." << std::endl;
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_res}, CL_MIGRATE_MEM_OBJECT_HOST));
        q.finish();
        // OPENCL HOST CODE AREA END

        // Check results
        max_diff[j] = 0.0f;
        mean_diff[j] = 0.0f;

        std::cout << "Comparing results..." << std::endl;
        std::cout << "Integration tolerance = " << integration_tolerance << std::endl;
        for (int i = 0; i < num; i++) {
            double diff = std::abs(res[i] - host_res[i]);
            std::cout << "FPGA (" << res[i] << ") CPU(" << host_res[i] << ") diff (" << diff << ")" << std::endl;
            if (diff > max_diff[j]) {
                max_diff[j] = diff;
            }
            mean_diff[j] += diff;
        }
        mean_diff[j] /= num;
        std::cout << "Largest host-kernel difference = " << max_diff[j] << std::endl;
        std::cout << "Mean host-kernel difference = " << mean_diff[j] << std::endl;
    }

    // check pass/fail
    int fail = 0;
    for (int i = 0; i < 3; i++) {
        std::string s;
        if (i == 0) {
            s = "Trapezoidal";
        } else if (i == 1) {
            s = "Simpson";
        } else {
            s = "Romberg";
        }
        if (max_diff[i] > integration_tolerance * 10) {
            std::cout << "FAIL: " << s << ": max_diff(" << max_diff[i] << ") > " << integration_tolerance * 10
                      << std::endl;
            fail = 1;
        }
        if (mean_diff[i] > integration_tolerance * 3) {
            std::cout << "FAIL: " << s << ": mean_diff(" << mean_diff[i] << ") > " << integration_tolerance * 3
                      << std::endl;
            fail = 1;
        }
    }

    std::cout << "All tests PASS" << std::endl;
    fail ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return fail;
}
