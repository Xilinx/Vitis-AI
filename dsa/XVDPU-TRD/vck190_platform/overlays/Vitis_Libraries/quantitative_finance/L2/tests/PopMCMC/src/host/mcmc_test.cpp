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
* @file mcmc_test.cpp
* @brief Testbench to generate randomized input data and launch on kernel.
*/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

//#include "mcmc_kernel.hpp"
#define NUM_CHAINS 10
#define KERNEL_DT double

// Temporary copy of this macro definition until new xcl2.hpp is used
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

/// @brief Main entry point to test
///
/// This is a command-line application to test the kernel.  It supports software and hardware emulation as well as
/// running on an Alveo target.
///
/// Usage: ./mcmc_test ./xclbin/<kernel_name> <number of prices>
///
/// @param[in] argc Standard C++ argument count
/// @param[in] argv Standard C++ input arguments
int main(int argc, char* argv[]) {
    std::cout << std::endl << std::endl;
    std::cout << "*************" << std::endl;
    std::cout << "MCMC Demo v1.0" << std::endl;
    std::cout << "*************" << std::endl;
    std::cout << std::endl;

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    FILE* fp;
    unsigned int argIdx = 1;
    std::string xclbin_file(argv[argIdx++]);
    unsigned int num_samples = std::atoi(argv[argIdx++]);
    unsigned int num_burn = std::atoi(argv[argIdx++]);

    // Vectors for parameter storage.  These use an aligned allocator in order
    // to avoid an additional copy of the host memory into the device
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > temp_inv(NUM_CHAINS);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > sigma(NUM_CHAINS);
    std::vector<KERNEL_DT, aligned_allocator<KERNEL_DT> > sample(num_samples);

    for (unsigned int n = 0; n < NUM_CHAINS; n++) {
        sigma[n] = 0.4;
        double temp = pow(NUM_CHAINS / (NUM_CHAINS - n), 2);
        temp_inv[n] = 1 / temp;
    }
    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::cout << "Connecting to device and loading kernel..." << std::endl;
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
    cl::Kernel krnl_cf_mcmc(program, "mcmc_kernel", &err);
    logger.logCreateKernel(err);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    std::cout << "Allocating buffers..." << std::endl;
    OCL_CHECK(err, cl::Buffer buffer_temp_inv(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                              NUM_CHAINS * sizeof(KERNEL_DT), temp_inv.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_sigma(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           NUM_CHAINS * sizeof(KERNEL_DT), sigma.data(), &err));

    OCL_CHECK(err, cl::Buffer buffer_sample(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                            num_samples * sizeof(KERNEL_DT), sample.data(), &err));

    // Set the arguments
    OCL_CHECK(err, err = krnl_cf_mcmc.setArg(0, buffer_temp_inv));
    OCL_CHECK(err, err = krnl_cf_mcmc.setArg(1, buffer_sigma));
    OCL_CHECK(err, err = krnl_cf_mcmc.setArg(2, buffer_sample));
    OCL_CHECK(err, err = krnl_cf_mcmc.setArg(3, num_samples));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_temp_inv}, 0));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_sigma}, 0));

    // Launch the Kernel
    std::cout << "Launching kernel..." << std::endl;
    uint64_t nstimestart, nstimeend;
    cl::Event event;
    OCL_CHECK(err, err = q.enqueueTask(krnl_cf_mcmc, NULL, &event));
    OCL_CHECK(err, err = q.finish());
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto duration_nanosec = nstimeend - nstimestart;
    std::cout << "  Duration returned by profile API is " << (duration_nanosec * (1.0e-6)) << " ms **** " << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_sample}, CL_MIGRATE_MEM_OBJECT_HOST));

    q.finish();

    int ret = 0;
    std::cout << "Kernel done!" << std::endl;
    std::cout << "Processed " << num_samples << " samples ";
    std::cout << "with " << NUM_CHAINS << " chains" << std::endl;
    std::cout << "Samples saved to vitis_samples_out.csv" << std::endl;
    std::cout << "Use Python plot_hist.py to plot histogram " << std::endl;
    fp = fopen("vitis_samples_out.csv", "wb");
    for (unsigned int k = num_burn; k < num_samples; k++) {
        fprintf(fp, "%lf\n", sample[k]);
        if (k == num_samples - 1) {
            fprintf(fp, "%lf", sample[k]);
        }

        // quick fix for pass/fail criteria
        // This implementation shouldn't generate samples out of <-2,2>
        if (std::abs(sample[k]) > 2) {
            ret = 1;
        }
    }
    fclose(fp);
    std::cout << std::endl;

    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return ret;
}
