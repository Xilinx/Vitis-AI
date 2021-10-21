/*
 * Copyright 2020 Xilinx, Inc.
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

#ifndef _HLS_TEST_
#ifndef _GENDATA_
#include "xcl2.hpp"
#endif
#endif
#ifndef __SYNTHESIS__
#include <algorithm>
#include <iostream>
#include <limits>
#include <string.h>
#include <sys/time.h>
#include "graph.hpp"
#include "xf_utils_sw/logger.hpp"
#endif

#include "xf_graph_L2.hpp"
#ifdef _HLS_TEST_
#ifndef _GENDATA_
#include "kernel_calcuDegree.hpp"
#endif
#endif

#define DT double
typedef ap_uint<512> buffType;

#ifndef __SYNTHESIS__

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

int main(int argc, const char* argv[]) {
    // Initialize parserl
    ArgParser parser(argc, argv);

    // Initialize paths addresses
    std::string xclbin_path;
    std::string num_str;

    int num_runs;
    int nrows;
    int nnz;

    // Read In paths addresses
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "INFO: input path is not set!\n";
    }
    if (!parser.getCmdOption("-runs", num_str)) {
        num_runs = 1;
        std::cout << "INFO: number runs is not set!\n";
    } else {
        num_runs = std::stoi(num_str);
    }
    if (!parser.getCmdOption("-nnz", num_str)) {
        nnz = 7;
        std::cout << "INFO: number of non-zero is not set!\n";
    } else {
        nnz = std::stoi(num_str);
    }
    if (!parser.getCmdOption("-nrows", num_str)) {
        nrows = 5;
        std::cout << "INFO: number of rows/column is not set!\n";
    } else {
        nrows = std::stoi(num_str);
    }
    std::string files;
    std::string filename, tmp, filename2_1, filename2_2;
    std::string dataSetDir;
    std::string refDir;
    if (!parser.getCmdOption("-files", num_str)) {
        files = "";
        std::cout << "INFO: dataSet name is not set!\n";
    } else {
        files = num_str;
    }
    if (!parser.getCmdOption("-dataSetDir", num_str)) {
        dataSetDir = "./data/";
        std::cout << "INFO: dataSet dir is not set!\n";
    } else {
        dataSetDir = num_str;
    }
    if (!parser.getCmdOption("-refDir", num_str)) {
        refDir = "./data/";
        std::cout << "INFO: reference dir is not set!\n";
    } else {
        refDir = num_str;
    }

    filename = dataSetDir + files + ".txt";
    filename2_1 = dataSetDir + files + "csr_offsets.txt";
    filename2_2 = dataSetDir + files + "csr_columns.txt";
    std::cout << "INFO: dataSet path is " << filename << std::endl;
    std::cout << "INFO: dataSet offset path is " << filename2_1 << std::endl;
    std::cout << "INFO: dataSet indice path is " << filename2_2 << std::endl;

    std::string fileRef;
    fileRef = refDir + "pagerank_ref.txt";
    std::cout << "INFO: reference data path is " << fileRef << std::endl;

    // Variables to measure time
    struct timeval tstart, tend;
    struct timeval tstart1, tend1;
    struct timeval tstart2, tend2;
    struct timeval tstart0, tend0;
    struct timeval tstartE2E, tendE2E;

    gettimeofday(&tstartE2E, 0);

    CscMatrix<int, DT> cscMat;
    readInWeightedDirectedGraphCV<int, DT>(filename2_2, cscMat, nnz);
    std::cout << "INFO: ReadIn succeed" << std::endl;
    readInWeightedDirectedGraphRow<int, DT>(filename2_1, cscMat, nnz, nrows);

    // Output the inputs information
    std::cout << "INFO: Number of kernel runs: " << num_runs << std::endl;
    std::cout << "INFO: Number of edges: " << nnz << std::endl;
    std::cout << "INFO: Number of nrows: " << nrows << std::endl;

    ///// declaration
    ap_uint<32>* indiceArr = aligned_alloc<ap_uint<32> >(nnz);
    for (int i = 0; i < nnz; ++i) {
        indiceArr[i] = cscMat.row.data()[i];
    }
    ap_uint<32>* degreeCSR = aligned_alloc<ap_uint<32> >(nrows + 16);

    DT* golden = new DT[nrows];

    for (int i = 0; i < nrows; ++i) {
        golden[i] = 0;
        degreeCSR[i] = 0;
    }
    for (int i = 0; i < nnz; ++i) {
        golden[indiceArr[i]] += 1;
    }

#ifdef _HLS_TEST_
    ap_uint<512>* degree = reinterpret_cast<ap_uint<512>*>(degreeCSR);
    ap_uint<512>* indiceCSC = reinterpret_cast<ap_uint<512>*>(indiceArr);
    kernel_calcuDegree_0(nrows, nnz, degree, indiceCSC);
#else
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // Platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl_int err;
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
    cl::Kernel kernel_calcuDegree(program, "kernel_calcuDegree_0", &err);
    logger.logCreateKernel(err);
    std::cout << "INFO: Kernel has been created" << std::endl;

    // DDR Settings
    std::vector<cl_mem_ext_ptr_t> mext_in(2);
    mext_in[0] = {2, degreeCSR, kernel_calcuDegree()};
    mext_in[1] = {3, indiceArr, kernel_calcuDegree()};

    // Create device buffer and map dev buf to host buf
    std::vector<cl::Buffer> buffer(2);

    buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(ap_uint<32>) * (nrows + 16), &mext_in[0]);
    buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           sizeof(ap_uint<32>) * nnz, &mext_in[1]);

    // Data transfer from host buffer to device buffer

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    ob_in.push_back(buffer[1]);
    ob_out.push_back(buffer[0]);

    kernel_calcuDegree.setArg(0, nrows);
    kernel_calcuDegree.setArg(1, nnz);
    kernel_calcuDegree.setArg(2, buffer[0]);
    kernel_calcuDegree.setArg(3, buffer[1]);

    // Setup kernel
    std::cout << "INFO: Finish kernel setup" << std::endl;
    std::vector<std::vector<cl::Event> > kernel_evt(1);
    std::vector<std::vector<cl::Event> > kernel_evt1(1);
    std::vector<std::vector<cl::Event> > kernel_evt2(1);
    std::vector<std::vector<cl::Event> > kernel_evt3(1);
    kernel_evt[0].resize(1);
    kernel_evt1[0].resize(1);
    kernel_evt2[0].resize(1);
    kernel_evt3[0].resize(1);

    for (int i = 0; i < num_runs; ++i) {
        q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &kernel_evt[i][0]); // 0 : migrate from host to dev
    }
    // Launch kernel and compute kernel execution time
    for (int i = 0; i < num_runs; ++i) {
        q.enqueueTask(kernel_calcuDegree, &kernel_evt[i], &kernel_evt1[i][0]);
    }

    // Data transfer from device buffer to host buffer
    for (int i = 0; i < num_runs; ++i) {
        q.enqueueMigrateMemObjects(ob_out, 1, &kernel_evt1[i], &kernel_evt2[i][0]); // 1 : migrate from dev to host
    }
    q.finish();

    gettimeofday(&tendE2E, 0);
    cl_ulong ts, te;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish kernel0 execution" << std::endl;
    kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    float elapsed = ((float)te - (float)ts) / 1000000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_H2D_MS, elapsed);

    kernel_evt[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    kernel_evt[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    elapsed = ((float)te - (float)ts) / 1000000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_KERNEL_MS, elapsed);

    kernel_evt2[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
    kernel_evt2[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
    elapsed = ((float)te - (float)ts) / 1000000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_D2H_MS, elapsed);

    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish E2E execution" << std::endl;
    int exec_timeE2E = diff(&tendE2E, &tstartE2E);
    std::cout << "INFO: FPGA execution time of " << num_runs << " runs:" << exec_timeE2E << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
#endif
    // Calculate err
    DT errs = 0.0;
    DT tolerance = 1e-3;
    int accurate = 0;
    for (int i = 0; i < nrows; ++i) {
        errs += (golden[i] - degreeCSR[i]) * (golden[i] - degreeCSR[i]);
        if (std::abs(degreeCSR[i] - golden[i]) < tolerance) {
            accurate += 1;
        }
    }
    DT accRate = accurate * 1.0 / nrows;
    errs = std::sqrt(err);
    std::cout << "INFO: Accurate Rate = " << accRate << std::endl;
    std::cout << "INFO: Err Geomean = " << errs << std::endl;

    free(indiceArr);
    free(degreeCSR);
    delete[] golden;

    if (errs < 1e-2) {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
        return 0;
    } else {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
        return 1;
    }
}
#endif
