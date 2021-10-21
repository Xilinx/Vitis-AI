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
#include <string.h>
#include <sys/time.h>
#include <algorithm>

#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

#include "matrixUtility.hpp"

// Memory alignment
template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) {
        throw std::bad_alloc();
    }
    return reinterpret_cast<T*>(ptr);
}

// Compute time difference
unsigned long diff(const struct timeval* newTime, const struct timeval* oldTime) {
    return (newTime->tv_sec - oldTime->tv_sec) * 1000000 + (newTime->tv_usec - oldTime->tv_usec);
}

// Arguments parser
class ArgParser {
   public:
    ArgParser(int& argc, const char** argv) {
        for (int i = 1; i < argc; ++i) mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(const std::string option, std::string& value) const {
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

//! Core function of SVD benchmark
int main(int argc, const char* argv[]) {
    // Initialize parser
    ArgParser parser(argc, argv);

    // Initialize paths addresses
    std::string xclbin_path;
    std::string num_str;
    int num_runs, dataAM;

    // Read In paths addresses
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "INFO:input path is not set!\n";
    }
    if (!parser.getCmdOption("-runs", num_str)) {
        num_runs = 1;
        std::cout << "INFO:number runs is not set!\n";
    } else {
        num_runs = std::stoi(num_str);
    }
    if (!parser.getCmdOption("-M", num_str)) {
        dataAM = 16;
        std::cout << "INFO:matrix Size is not set!\n";
    } else {
        dataAM = std::stoi(num_str);
    }

    // Platform related operations
    xf::common::utils_sw::Logger logger;
    cl_int err = CL_SUCCESS;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);

    cl::Program program(context, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);

    cl::Kernel kernel_gtsv_0(program, "kernel_gtsv_0", &err);
    logger.logCreateKernel(err);

    // Output the inputs information
    std::cout << "INFO: Number of kernel runs: " << num_runs << std::endl;
    std::cout << "INFO: Matrix size: " << dataAM << std::endl;

    // Initialization of host buffers
    double* matDiagLow = aligned_alloc<double>(dataAM);
    double* matDiag = aligned_alloc<double>(dataAM);
    double* matDiagUp = aligned_alloc<double>(dataAM);
    double* rhs = aligned_alloc<double>(dataAM);

    for (int i = 0; i < dataAM; i++) {
        matDiagLow[i] = -1.0;
        matDiag[i] = 2.0;
        matDiagUp[i] = -1.0;
        rhs[i] = 0.0;
    };
    matDiagLow[0] = 0.0;
    matDiagUp[dataAM - 1] = 0.0;
    rhs[0] = 1.0;
    rhs[dataAM - 1] = 1.0;

    // DDR Settings
    std::vector<cl_mem_ext_ptr_t> mext_io(4);
    // mext_io[0].flags = XCL_MEM_DDR_BANK0;
    // mext_io[1].flags = XCL_MEM_DDR_BANK0;
    // mext_io[2].flags = XCL_MEM_DDR_BANK0;
    // mext_io[3].flags = XCL_MEM_DDR_BANK0;

    // mext_io[0].obj = matDiagLow;
    // mext_io[0].param = 0;
    // mext_io[1].obj = matDiag;
    // mext_io[1].param = 0;
    // mext_io[2].obj = matDiagUp;
    // mext_io[2].param = 0;
    // mext_io[3].obj = rhs;
    // mext_io[3].param = 0;

    mext_io[0] = {1, matDiagLow, kernel_gtsv_0()};
    mext_io[1] = {2, matDiag, kernel_gtsv_0()};
    mext_io[2] = {3, matDiagUp, kernel_gtsv_0()};
    mext_io[3] = {4, rhs, kernel_gtsv_0()};

    // Create device buffer and map dev buf to host buf
    cl::Buffer matdiaglow_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                              sizeof(double) * dataAM, &mext_io[0]);
    cl::Buffer matdiag_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           sizeof(double) * dataAM, &mext_io[1]);
    cl::Buffer matdiagup_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                             sizeof(double) * dataAM, &mext_io[2]);
    cl::Buffer rhs_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                       sizeof(double) * dataAM, &mext_io[3]);

    // Data transfer from host buffer to device buffer
    std::vector<std::vector<cl::Event> > kernel_evt(2);
    kernel_evt[0].resize(1);
    kernel_evt[1].resize(1);

    std::vector<cl::Memory> ob_in, ob_out;
    ob_in.push_back(matdiaglow_buffer);
    ob_in.push_back(matdiag_buffer);
    ob_in.push_back(matdiagup_buffer);
    ob_in.push_back(rhs_buffer);

    ob_out.push_back(rhs_buffer);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &kernel_evt[0][0]); // 0 : migrate from host to dev
    q.finish();
    std::cout << "INFO: Finish data transfer from host to device" << std::endl;

    // Setup kernel
    kernel_gtsv_0.setArg(0, dataAM);
    kernel_gtsv_0.setArg(1, matdiaglow_buffer);
    kernel_gtsv_0.setArg(2, matdiag_buffer);
    kernel_gtsv_0.setArg(3, matdiagup_buffer);
    kernel_gtsv_0.setArg(4, rhs_buffer);
    q.finish();
    std::cout << "INFO: Finish kernel setup" << std::endl;

    // Variables to measure time
    struct timeval tstart, tend;

    // Launch kernel and compute kernel execution time
    gettimeofday(&tstart, 0);
    for (int i = 0; i < num_runs; ++i) {
        q.enqueueTask(kernel_gtsv_0, nullptr, nullptr);
    }
    q.finish();
    gettimeofday(&tend, 0);
    std::cout << "INFO: Finish kernel execution" << std::endl;
    int exec_time = diff(&tend, &tstart);
    std::cout << "INFO: FPGA execution time of " << num_runs << " runs:" << exec_time << " us\n"
              << "INFO: Average executiom per run: " << exec_time / num_runs << " us\n";

    // Data transfer from device buffer to host buffer
    q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr); // 1 : migrate from dev to host
    q.finish();

    int rtl = 0;
    for (int i = 0; i < dataAM; i++) {
        if (std::abs(rhs[i] - 1.0) > 1e-7) rtl = 1;
    }
    if (rtl == 1) {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
        return -1;
    } else {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
        return 0;
    }

    return 0;
}
