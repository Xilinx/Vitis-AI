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

#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "hcf.hpp"
#include "hcf_host.hpp"
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

#define STR1(x) #x
#define STR(x) STR1(x)

// test variables, can be changed from the command line
static TEST_DT dw = 0.5;
static int w_max = 200;
static TEST_DT tol = 0.0001;
static std::string file = "tmp.txt";
static int run_cpu = 0;
static int gen_csv = 0;
static int display_results = 0;
static int check_expected_values = 0;
static std::string binaryFile = "";

int check_value(TEST_DT act, TEST_DT exp, TEST_DT tolerance, TEST_DT* diff) {
    *diff = my_fabs(act - exp);
    if (*diff > tolerance) {
        return 0;
    }
    return 1;
}

void usage(char* name) {
    std::cout << name << " [-f<test file> -d<dw> -w<w_max> -t<tolerance> -c -v -o -e -h]" << std::endl;
    std::cout << "dw is the integral increment (default 0.5)" << std::endl;
    std::cout << "w_max is the integration limit (default 200)" << std::endl;
    std::cout << "-c run the CPU calculation" << std::endl;
    std::cout << "-o produce csv output file" << std::endl;
    std::cout << "-v display the results" << std::endl;
    std::cout << "-e check expected values" << std::endl;
}

void generate_csv(std::string file,
                  struct xf::fintech::hcfEngineInputDataType<TEST_DT>* in,
                  TEST_DT* out,
                  int num_tests) {
    /* create output filename */
    size_t lastindex = file.find_last_of(".");
    std::string outfile = file.substr(0, lastindex) + ".csv";

    std::ofstream f(outfile, std::ios::app);
    if (!f.is_open()) {
        std::cout << "ERROR: failed to open output file: " << outfile << std::endl;
        return;
    }

    f << std::setprecision(12);
    for (int i = 0; i < num_tests; i++) {
        f << out[i] << ",";
    }
    f << std::endl;
    f.close();
}

void display_test_parameters(struct xf::fintech::hcfEngineInputDataType<TEST_DT>* p) {
    std::cout << "ERROR: ";
    std::cout << "S0=" << p->s0 << ", ";
    std::cout << "V0=" << p->v0 << ", ";
    std::cout << "K=" << p->K << ", ";
    std::cout << "rho=" << p->rho << ", ";
    std::cout << "T=" << p->T << ", ";
    std::cout << "r=" << p->r << ", ";
    std::cout << "kappa=" << p->kappa << ", ";
    std::cout << "vvol=" << p->vvol << ", ";
    std::cout << "vbar=" << p->vbar << ", ";
    std::cout << "dw=" << p->dw << ", ";
    std::cout << "w_max=" << p->w_max << std::endl;
}

int deal_with_cmd_line_args(int argc, char** argv) {
    int opt = 0;
    int b_set = 0;
    try {
        while ((opt = getopt(argc, argv, "f:d:w:t:b:cvhoe")) != -1) {
            switch (opt) {
                case 'f':
                    file = std::string(optarg);
                    break;
                case 'd':
                    dw = atof(optarg);
                    break;
                case 'w':
                    w_max = atoi(optarg);
                    break;
                case 't':
                    tol = atof(optarg);
                    break;
                case 'b':
                    binaryFile = std::string(optarg);
                    b_set = 1;
                    break;
                case 'c':
                    run_cpu = 1;
                    break;
                case 'v':
                    display_results = 1;
                    break;
                case 'o':
                    gen_csv = 1;
                    break;
                case 'e':
                    check_expected_values = 1;
                    break;
                case 'h':
                    usage(argv[0]);
                    return 0;
                    break;
                default:
                    break;
            }
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR: Failed to parse command line arg: " << opt << ": " << optarg << ": exc: " << e.what()
                  << std::endl;
        return 0;
    }
    if (!b_set) {
        std::cout << "ERROR: xclbin path is not set" << std::endl;
        return 0;
    }

    return 1;
}

class ArgParser {
   public:
    ArgParser(int& argc, char** argv) {
        for (int i = 1; i < argc; ++i) mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(std::string option, std::string& value) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end() && ++itr != this->mTokens.end()) {
            value = *itr;
            return true;
        }
        return false;
    }

   private:
    std::vector<std::string> mTokens;
};

int main(int argc, char** argv) {
    // cmd parser

    if (!deal_with_cmd_line_args(argc, argv)) {
        exit(1);
    }

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // IO data
    std::vector<struct xf::fintech::hcfEngineInputDataType<TEST_DT>,
                aligned_allocator<struct xf::fintech::hcfEngineInputDataType<TEST_DT> > >
        input_data(MAX_NUMBER_TESTS);
    std::vector<TEST_DT, aligned_allocator<TEST_DT> > output_data(MAX_NUMBER_TESTS);
    size_t bytes_in = sizeof(struct xf::fintech::hcfEngineInputDataType<TEST_DT>) * MAX_NUMBER_TESTS;
    size_t bytes_out = sizeof(TEST_DT) * MAX_NUMBER_TESTS;

    // parse the input data file
    cl_int num_tests = 0;
    TEST_DT expected_values[MAX_NUMBER_TESTS];
    auto t_start = std::chrono::high_resolution_clock::now();
    if (!parse_file(file, input_data, dw, w_max, &num_tests, expected_values, MAX_NUMBER_TESTS)) {
        if (num_tests > MAX_NUMBER_TESTS) {
            std::cout << "ERROR: too many tests, max=: " << MAX_NUMBER_TESTS << std::endl;
        } else {
            std::cout << "ERROR: failed to parse file: " << file << std::endl;
        }
        return 1;
    }
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();
    std::cout << std::setw(40) << std::left << "Parse file time"
              << "= " << duration << "us" << std::endl;

    //------------------------------ CPU ---------------------------------
    TEST_DT cpu_cp[MAX_NUMBER_TESTS];
    if (run_cpu) {
        t_start = std::chrono::high_resolution_clock::now();
        call_price(input_data, num_tests, cpu_cp);
        duration =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
                .count();
        std::cout << std::setw(40) << std::left << "CPU time"
                  << "= " << duration << "us" << std::endl;
    }

    //------------------------------ FPGA ---------------------------------
    // create the context
    cl_int err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue cq(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // create the xclbin filename
    /*std::string mode = "hw";
    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode = std::getenv("XCL_EMULATION_MODE");
    }
    std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = "xclbin/hcf_" + mode + "_" + STR(DEVICE_PART) + "_" + STR(TEST_DT) + ".xclbin";
*/
    // import and program the binary
    t_start = std::chrono::high_resolution_clock::now();
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel krnl(program, "hcf_kernel", &err);
    logger.logCreateKernel(err);
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();
    std::cout << std::setw(40) << std::left << "Import binary time"
              << "= " << duration << "us" << std::endl;

    // memory objects
    cl::Buffer dev_in(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bytes_in,
                      input_data.data()); /* read/write seems to be
                                             needed here as opposed to
                                             simple read */
    cl::Buffer dev_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bytes_out,
                       output_data.data()); /* read/write seems to be
                                               needed here as opposed
                                               to simply write */

    // copy input data to device
    t_start = std::chrono::high_resolution_clock::now();
    cq.enqueueMigrateMemObjects({dev_in}, 0);
    cq.finish();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();
    std::cout << std::setw(40) << std::left << "Input data transfer time"
              << "= " << duration << "us" << std::endl;

    // set the kernel args
    krnl.setArg(0, dev_in);
    krnl.setArg(1, dev_out);
    krnl.setArg(2, num_tests);

    // run the kernel
    cl::Event kernel_event;
    t_start = std::chrono::high_resolution_clock::now();
    cq.enqueueTask(krnl, NULL, &kernel_event);
    cq.finish();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();
    std::cout << std::setw(40) << std::left << "FPGA time"
              << "= " << duration << "us" << std::endl;

    // get the results
    t_start = std::chrono::high_resolution_clock::now();
    cq.enqueueMigrateMemObjects({dev_out}, CL_MIGRATE_MEM_OBJECT_HOST);
    cq.finish();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();
    std::cout << std::setw(40) << std::left << "Output data transfer time"
              << "= " << duration << "us" << std::endl;

    // display results
    if (display_results) {
        std::cout << std::fixed << std::setprecision(6);
        for (int i = 0; i < num_tests; i++) {
            std::cout << std::setw(40) << std::left << "FPGA Call price"
                      << "= " << output_data[i];
            if (run_cpu) {
                TEST_DT diff;
                if (!check_value(output_data[i], cpu_cp[i], tol, &diff)) {
                    std::cout << "\033[1;31m";
                }

                std::cout << "   (CPU = " << cpu_cp[i] << ")";
                std::cout << "\033[1;0m";
            }
            std::cout << std::endl;
        }
    }

    // check against QuantLib reference
    int num_fails = 0;
    if (check_expected_values) {
        std::cout << std::setprecision(6) << std::fixed;
        for (int i = 0; i < num_tests; i++) {
            TEST_DT diff;
            if (!check_value(output_data[i], expected_values[i], tol, &diff)) {
                num_fails++;
                display_test_parameters(&input_data[i]);
                std::cout << "    ERROR: FPGA Call Price = " << output_data[i] << std::endl;
                std::cout << "    ERROR: Expected        = " << expected_values[i] << std::endl;
                std::cout << "    ERROR: error           = " << diff << std::endl;
                std::cout << std::endl;
            }
        }
        std::cout << "Total Tests = " << num_tests << std::endl;
        std::cout << "Total Fails = " << num_fails << std::endl;
    }

    if (gen_csv) {
        generate_csv(file, input_data.data(), output_data.data(), num_tests);
    }

    int ret = 0;
    if (num_fails) {
        ret = 1;
    }
    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return ret;
}
