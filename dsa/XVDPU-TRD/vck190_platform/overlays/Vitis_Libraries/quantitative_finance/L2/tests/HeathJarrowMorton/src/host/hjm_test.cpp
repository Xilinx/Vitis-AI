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
 * @file hjm_test.cpp
 * @brief Testbench to feed input data and launch on kernel
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <getopt.h>
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

typedef double TEST_DT;
template <typename T>
using al_vec = std::vector<T, aligned_allocator<T> >;

#define TEST_MAX_TENORS (54)
#define TEST_MAX_CURVES (1280)
#define N_FACTORS (3)
#define MC_UN (4)
#define DEMO_VERSION "1.0"
#define KERNEL_NAME "hjm_kernel"

#define DEF_YEARS (10.0)
#define DEF_PATHS (500)
#define DEF_ZCBM (10.0)
#define TAU (0.5)

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

void help(const char* pgrmName) {
    std::cout << "Usage " << std::string(pgrmName)
              << " -x <xclbin_loc> -d <hist_data_loc> [-s <sim_years> -p <no_paths> -z <zcb_maturity>]" << std::endl;
    std::cout << "\t-x --xclbin_loc     Location of xclbin file" << std::endl;
    std::cout << "\t-d --data_in        Location of historical rates csv file" << std::endl;
    std::cout << "\t-s --sim_years      Number of years to simulate per HJM path (Default " << DEF_YEARS << ")"
              << std::endl;
    std::cout << "\t-p --no_paths       Number of MC HJM paths to generate per run (Default " << DEF_PATHS << ")"
              << std::endl;
    std::cout << "\t-z --zcb_mat        Maturity of the ZeroCouponBond to be priced with HJM (Default " << DEF_ZCBM
              << ")" << std::endl;
}

/**
 * @brief Calculates the price of a ZeroCouponBond analytically. This will give a reference to compare
 * with the output from HJM.
 */
TEST_DT zcbAnalyticalPrice(TEST_DT* presentFc, float zcbMaturity, float simYears) {
    const unsigned tenors = static_cast<unsigned>(simYears / TAU);
    TEST_DT accum = 0.0f;
    for (unsigned i = 0; i < tenors; i++) {
        accum += presentFc[i] / 100;
    }
    return exp(-TAU * accum);
}

/**
 * @brief Generates a set of seeds to use when running the MC simulation in the FPGA
 */
al_vec<unsigned> getFpgaSeeds() {
    al_vec<unsigned> seeds(MC_UN * N_FACTORS);
    for (unsigned i = 0; i < MC_UN * N_FACTORS; i++) {
        seeds[i] = 42 + i; // Replace here whatever seed code generation needed
    }
    return seeds;
}

/**
 * @brief Main entry point to test
 *
 * This is a command-line application to test the HJM kernel. It supports software and hardware emulation as well as
 * running on an Alveo target.
 *
 * Usage: ./hjm_test ./xclbin/<kernel_name> <historical_data_csv> [<sim_years> <no_paths>]
 */
int main(int argc, char* argv[]) {
    std::cout << "\n*************"
              << "\nHJM Demo v" << DEMO_VERSION << "\n*************" << std::endl;

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    static struct option longOps[] = {{"help", no_argument, 0, 'h'},
                                      {"xclbin_loc", required_argument, 0, 'x'},
                                      {"data_in", required_argument, 0, 'd'},
                                      {"sim_years", required_argument, 0, 's'},
                                      {"no_paths", required_argument, 0, 'p'},
                                      {"zcb_mat", required_argument, 0, 'z'},
                                      {0, 0, 0, 0}};

    // HJM Parameters
    std::string xclbinLoc, dataInLoc;
    float simYears = DEF_YEARS;
    float zcbMaturity = DEF_ZCBM;
    unsigned noPaths = DEF_PATHS;

    char opt;
    while ((opt = getopt_long(argc, argv, "hx:d:s:p:", longOps, NULL)) != -1) {
        switch (opt) {
            case 'x':
                xclbinLoc = std::string(optarg);
                break;
            case 'd':
                dataInLoc = std::string(optarg);
                break;
            case 's':
                simYears = atof(optarg);
                break;
            case 'p':
                noPaths = atoi(optarg);
                break;
            case 'z':
                zcbMaturity = atof(optarg);
                break;
            case 'h':
            case '?':
            default:
                help(argv[0]);
                return -1;
        }
    }

    if (xclbinLoc.empty()) {
        std::cerr << "Missing mandatory argument '--xclbin_loc'" << std::endl;
        return -1;
    }
    if (dataInLoc.empty()) {
        std::cerr << "Missing mandatory argument '--data_in'" << std::endl;
        return -1;
    }

    // Vectors for parameter storage.
    al_vec<TEST_DT> inputData(TEST_MAX_TENORS * TEST_MAX_CURVES);
    al_vec<TEST_DT> outPrice(1);
    al_vec<unsigned> seeds = getFpgaSeeds();
    unsigned noTenors = 0, noCurves = 0;

    // Read input data
    std::cout << "Loading input data..." << std::endl;
    std::ifstream in;
    in.open(dataInLoc.c_str());
    if (!in.is_open()) {
        std::cerr << "Failed to open file '" << dataInLoc << "'" << std::endl;
        return -1;
    }

    std::string line;
    size_t idx = 0;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string word;
        while (std::getline(iss, word, ',')) {
            inputData[idx++] = static_cast<TEST_DT>(atof(word.c_str()));
        }
        noCurves++;
    }
    in.close();
    noTenors = idx / noCurves;

    std::cout << "HJM Parameters: " << std::endl
              << "\txclbin_loc = " << xclbinLoc << std::endl
              << "\tdata_in = " << dataInLoc << std::endl
              << "\tno_tenors = " << noTenors << std::endl
              << "\tno_curves = " << noCurves << std::endl
              << "\tsim_years = " << simYears << std::endl
              << "\tno_paths = " << noPaths << std::endl
              << "\tzcb_mat = " << zcbMaturity << std::endl;

    // OPENCL HOST CODE AREA START

    std::cout << "\n\nConnecting to device and loading kernel..." << std::endl;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl_int err;

    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // Load the binary file (using function from xcl2.cpp)
    cl::Program::Binaries bins = xcl::import_binary_file(xclbinLoc);

    devices.resize(1);
    cl::Program program(context, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel krnl_hjm(program, KERNEL_NAME, &err);
    logger.logCreateKernel(err);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    std::cout << "Allocating buffers..." << std::endl;
    OCL_CHECK(err,
              cl::Buffer bufferHistData(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        TEST_MAX_CURVES * TEST_MAX_TENORS * sizeof(TEST_DT), inputData.data(), &err));
    OCL_CHECK(err, cl::Buffer bufferSeeds(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          MC_UN * N_FACTORS * sizeof(unsigned), seeds.data(), &err));
    OCL_CHECK(err, cl::Buffer bufferPriceOut(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(TEST_DT),
                                             outPrice.data(), &err));

    // Set the arguments
    OCL_CHECK(err, err = krnl_hjm.setArg(0, bufferHistData));
    OCL_CHECK(err, err = krnl_hjm.setArg(1, noTenors));
    OCL_CHECK(err, err = krnl_hjm.setArg(2, noCurves));
    OCL_CHECK(err, err = krnl_hjm.setArg(3, simYears));
    OCL_CHECK(err, err = krnl_hjm.setArg(4, noPaths));
    OCL_CHECK(err, err = krnl_hjm.setArg(5, zcbMaturity));
    OCL_CHECK(err, err = krnl_hjm.setArg(6, bufferSeeds));
    OCL_CHECK(err, err = krnl_hjm.setArg(7, bufferPriceOut));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bufferHistData, bufferSeeds}, 0));

    // Launch the kernel
    std::cout << "Launching kernel..." << std::endl;
    uint64_t nstimestart, nstimeend;
    cl::Event event;
    OCL_CHECK(err, err = q.enqueueTask(krnl_hjm, NULL, &event));
    OCL_CHECK(err, err = q.finish());
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto duration_nanosec = nstimeend - nstimestart;
    std::cout << "  Duration returned by profile API is " << (duration_nanosec * (1.0e-6)) << " ms **** " << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bufferPriceOut}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    // OPENCL HOST CODE AREA END

    TEST_DT fpgaPrice = outPrice[0];
    // For analytical price, use the last row of curves (present fc)
    const size_t lastRowIdx = (noCurves - 1) * noTenors;
    TEST_DT expectedPrice = zcbAnalyticalPrice(inputData.data() + lastRowIdx, zcbMaturity, simYears);
    double epsilon = 0.02; // 2%

    std::cout << "Kernel done!" << std::endl;
    std::cout << "Calculated ZCB FPGA price = " << fpgaPrice << std::endl;
    std::cout << "Analytical ZCB price = " << expectedPrice << std::endl;

    int error = 0;
    double diff = std::abs((double)((fpgaPrice - expectedPrice) / expectedPrice));
    if (diff > epsilon) {
        std::cerr << "ERROR with ZCB pricing! (Diff = " << diff << ")" << std::endl;
        error = 1;
    }
    error ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
          : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return error;
}
