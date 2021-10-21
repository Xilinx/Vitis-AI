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

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>

#include "xcl2.hpp"
#include "m76_host.hpp"
#include "xf_utils_sw/logger.hpp"

#define STR1(x) #x
#define STR(x) STR1(x)

#define MAX_NUMBER_TESTS (2048)

// test variables, can be changed from the command line
static TEST_DT tolerance = 0.001;
static std::string file = "";
static std::string binaryFile = "";
static int check_cpu = 0;
static int check_fpga = 0;
static int check_fpga_against_cpu = 0;
static int speed = 0;
static int total_tests = 0;
static int total_fails = 0;

// check values are equal within tolerance
static int check_value(TEST_DT act, TEST_DT exp, TEST_DT* diff) {
    *diff = fabs(act - exp);
    if (*diff > tolerance) {
        return 0;
    }
    return 1;
}

void usage(char* name) {
    std::cout << name << " -f<test file> [options]" << std::endl;
    std::cout << "options:" << std::endl;
    std::cout << "    -t<tolerance> the tolerance with which to check results (default 0.001)" << std::endl;
    std::cout << "    -c check fpga results against EXP" << std::endl;
    std::cout << "    -C check cpu results against EXP" << std::endl;
    std::cout << "    -x check fpga against cpu results" << std::endl;
    std::cout << "    -s compare speed of execution" << std::endl;
    std::cout << "    -h display this message!" << std::endl;
}

void print_test_parameters(struct parsed_params* p) {
    std::cout << "S=" << p->S << ", ";
    std::cout << "K=" << p->K << ", ";
    std::cout << "sigma=" << p->sigma << ", ";
    std::cout << "T=" << p->T << ", ";
    std::cout << "r=" << p->r << ", ";
    std::cout << "lambda=" << p->lambda << ", ";
    std::cout << "kappa=" << p->kappa << ", ";
    std::cout << "delta=" << p->delta << ", ";
    std::cout << "exp=" << p->expected_value << std::endl;
    ;
}

// check the calculated results match the expected values
void check_results(std::vector<struct parsed_params*>* vect,
                   std::vector<TEST_DT, aligned_allocator<TEST_DT> >& results,
                   int num_tests,
                   int start_index) {
    for (int i = 0; i < num_tests; i++) {
        total_tests++;
        TEST_DT diff;
        if (!check_value(results.at(i), vect->at(start_index + i)->expected_value, &diff)) {
            std::cout << "FAIL (" << vect->at(start_index + i)->line_number << "): ";
            print_test_parameters(vect->at(i));
            std::cout << "      expected(" << vect->at(start_index + i)->expected_value << ") got(" << results.at(i)
                      << ") ";
            std::cout << "diff(" << diff << ")" << std::endl;
            std::cout << std::endl;
            total_fails++;
        }
    }
}

void check_results(std::vector<struct parsed_params*>* vect,
                   std::vector<TEST_DT, aligned_allocator<TEST_DT> >& fpga_results,
                   std::vector<TEST_DT, aligned_allocator<TEST_DT> >& cpu_results) {
    int i = 0;
    for (auto const& v : *vect) {
        total_tests++;
        TEST_DT diff;
        if (!check_value(fpga_results.at(i), cpu_results.at(i), &diff)) {
            std::cout << "FAIL (" << v->line_number << "): ";
            print_test_parameters(v);
            std::cout << "      fpga(" << fpga_results.at(i) << ") cpu(" << cpu_results.at(i) << ") ";
            std::cout << "diff(" << diff << ")" << std::endl;
            std::cout << std::endl;
            total_fails++;
        }
    }
}

int deal_with_cmd_line_args(int argc, char** argv) {
    int valid = 0;
    int opt = 0;
    int valid_b = 0;
    try {
        while ((opt = getopt(argc, argv, "f:t:b:Ccxsh")) != -1) {
            switch (opt) {
                case 'f':
                    file = std::string(optarg);
                    valid = 1;
                    break;
                case 't':
                    tolerance = atof(optarg);
                    break;
                case 'b':
                    binaryFile = std::string(optarg);
                    valid_b = 1;
                    break;
                case 'c':
                    check_fpga = 1;
                    break;
                case 'x':
                    check_fpga_against_cpu = 1;
                    break;
                case 'C':
                    check_cpu = 1;
                    break;
                case 's':
                    speed = 1;
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

    if (!valid) {
        std::cout << "ERROR: must provide test file" << std::endl;
        usage(argv[0]);
        return 0;
    }
    if (!valid_b) {
        std::cout << "ERROR: must provide xclbin file path" << std::endl;
        usage(argv[0]);
        return 0;
    }

    if (check_fpga + check_fpga_against_cpu + check_cpu > 1) {
        std::cout << "Only allowed one check; fpga | cpu | fpga vs cpu" << std::endl;
        usage(argv[0]);
        return 0;
    }

    return 1;
}

// copy kernel data from the parsed data structure
void copy_data(struct xf::fintech::jump_diffusion_params<TEST_DT>* dst, struct parsed_params* src) {
    // diffusion parameters
    dst->S = src->S;
    dst->K = src->K;
    dst->r = src->r;
    dst->sigma = src->sigma;
    dst->T = src->T;

    // jump parameters
    dst->lambda = src->lambda;
    dst->kappa = src->kappa;
    dst->delta = src->delta;
}

static std::vector<long int> execution_fpga;
static std::vector<long int> execution_cpu;
void run_fpga(cl::CommandQueue& cq, cl::Buffer& dev_in, cl::Buffer& dev_out, cl::Kernel& krnl, int n) {
    // copy input data to device
    auto t_start = std::chrono::high_resolution_clock::now();
    cq.enqueueMigrateMemObjects({dev_in}, 0);
    cq.finish();

    // set the kernel args
    krnl.setArg(0, dev_in);
    krnl.setArg(1, dev_out);
    krnl.setArg(2, n);

    // run the kernel
    cl::Event kernel_event;
    cq.enqueueTask(krnl, NULL, &kernel_event);
    cq.finish();

    // get the results
    cq.enqueueMigrateMemObjects({dev_out}, CL_MIGRATE_MEM_OBJECT_HOST);
    cq.finish();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();
    execution_fpga.push_back(duration);
}

int main(int argc, char** argv) {
    if (!deal_with_cmd_line_args(argc, argv)) {
        exit(1);
    }

    // parse the input test file
    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<struct parsed_params*>* vect = parse_file(file);
    if (vect == nullptr) {
        return 1;
    }
    auto parse_time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();

    // IO data structures
    std::vector<struct xf::fintech::jump_diffusion_params<TEST_DT>,
                aligned_allocator<struct xf::fintech::jump_diffusion_params<TEST_DT> > >
        input_data(MAX_NUMBER_TESTS);
    std::vector<TEST_DT, aligned_allocator<TEST_DT> > output_data(MAX_NUMBER_TESTS);
    std::vector<TEST_DT, aligned_allocator<TEST_DT> > cpu_output_data(MAX_NUMBER_TESTS);
    size_t bytes_in = sizeof(struct xf::fintech::jump_diffusion_params<TEST_DT>) * MAX_NUMBER_TESTS;
    size_t bytes_out = sizeof(TEST_DT) * MAX_NUMBER_TESTS;

    //------------------------------ CPU ---------------------------------
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    std::cout << "CPU:" << std::endl;
    int n = 0;
    int index = 0;

    for (auto const& v : *vect) {
        copy_data(&input_data.at(n), v);

        n++;
        // if we have filled the structure, run the calculations
        if (n == MAX_NUMBER_TESTS) {
            t_start = std::chrono::high_resolution_clock::now();
            cpu_merton_jump_diffusion(input_data, n, cpu_output_data);
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::high_resolution_clock::now() - t_start)
                                .count();
            execution_cpu.push_back(duration);

            if (check_cpu) {
                check_results(vect, cpu_output_data, n, index);
            }
            index += MAX_NUMBER_TESTS;
            n = 0;
            // display stats
            std::cout << std::endl;
        }
    }
    // run the remainder
    if (n > 0) {
        t_start = std::chrono::high_resolution_clock::now();
        cpu_merton_jump_diffusion(input_data, n, cpu_output_data);
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
                .count();
        execution_cpu.push_back(duration);
        if (check_cpu) {
            check_results(vect, cpu_output_data, n, index);
        }
    }

    //------------------------------ FPGA ---------------------------------
    std::cout << "FPGA:" << std::endl;
    if (check_fpga) {
        total_tests = 0;
        total_fails = 0;
    }

    // create the context
    cl_int err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue cq(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // create the xclbin filename
    std::string mode = "hw";
    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode = std::getenv("XCL_EMULATION_MODE");
    }
    std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
    // std::string binaryFile = "xclbin/m76_" + mode + "_" + STR(DEVICE_PART) + "_float" /*+ STR(TEST_DT)*/ + ".xclbin";

    // import and program the binary
    t_start = std::chrono::high_resolution_clock::now();
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel krnl(program, "m76_kernel", &err);
    logger.logCreateKernel(err);
    auto import_time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();

    // create the memory objects
    cl::Buffer dev_in(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bytes_in, input_data.data());
    cl::Buffer dev_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bytes_out, output_data.data());

    // run the kernel
    n = 0;
    index = 0;

    t_start = std::chrono::high_resolution_clock::now();
    for (auto const& v : *vect) {
        copy_data(&input_data.at(n), v);

        n++;
        // if we have filled the structure, run the calculations
        if (n == MAX_NUMBER_TESTS) {
            run_fpga(cq, dev_in, dev_out, krnl, n);
            if (check_fpga) {
                check_results(vect, output_data, n, index);
            }
            index += MAX_NUMBER_TESTS;
            n = 0;
        }
    }
    // run the remainder
    if (n > 0) {
        run_fpga(cq, dev_in, dev_out, krnl, n);
        if (check_fpga) {
            check_results(vect, output_data, n, index);
        }
    }
    auto fpga_time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start)
            .count();

    if (speed) {
        std::cout << "Parse file time    = " << parse_time << "us" << std::endl;
        std::cout << "Import binary time = " << import_time << "us" << std::endl;

        std::cout << "FPGA" << std::endl;
        long int mean_fpga = 0;
        for (auto const& t : execution_fpga) {
            std::cout << "    " << t << "us" << std::endl;
            mean_fpga += t;
        }
        mean_fpga /= execution_fpga.size();
        std::cout << "    mean = " << mean_fpga << std::endl;

        std::cout << "CPU" << std::endl;
        long int mean_cpu = 0;
        for (auto const& t : execution_cpu) {
            std::cout << "    " << t << "us" << std::endl;
            mean_cpu += t;
        }
        mean_cpu /= execution_cpu.size();
        std::cout << "    mean = " << mean_cpu << std::endl;
    }

    if (check_fpga_against_cpu) {
        check_results(vect, output_data, cpu_output_data);
    }

    if (check_fpga || check_cpu || check_fpga_against_cpu) {
        std::cout << "Total tests  = " << total_tests << std::endl;
        std::cout << "Total passes = " << total_tests - total_fails << std::endl;
        std::cout << "Total fails  = " << total_fails << std::endl;
    }

    int ret = 0;
    if (total_fails) {
        ret = 1;
    }
    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return ret;
}
