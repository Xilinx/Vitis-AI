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
#endif

#include "xf_graph_L2.hpp"
#ifdef _HLS_TEST_
#ifndef _GENDATA_
#include "kernel_pagerank.hpp"
#endif
#endif

#include "xf_utils_sw/logger.hpp"

//#define BANCKMARK

// typedef double DT;
typedef float DT;
typedef ap_uint<512> buffType;

#ifndef __SYNTHESIS__

#define XCL_BANK(n) (((unsigned int)(n)) | XCL_MEM_TOPOLOGY)

#define XCL_BANK0 XCL_BANK(0)
#define XCL_BANK1 XCL_BANK(1)
#define XCL_BANK2 XCL_BANK(2)
#define XCL_BANK3 XCL_BANK(3)
#define XCL_BANK4 XCL_BANK(4)
#define XCL_BANK5 XCL_BANK(5)
#define XCL_BANK6 XCL_BANK(6)
#define XCL_BANK7 XCL_BANK(7)
#define XCL_BANK8 XCL_BANK(8)
#define XCL_BANK9 XCL_BANK(9)
#define XCL_BANK10 XCL_BANK(10)
#define XCL_BANK11 XCL_BANK(11)
#define XCL_BANK12 XCL_BANK(12)
#define XCL_BANK13 XCL_BANK(13)
#define XCL_BANK14 XCL_BANK(14)
#define XCL_BANK15 XCL_BANK(15)

template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
#if _WIN32
    ptr = (T*)malloc(num * sizeof(T));
    if (num == 0) {
#else
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) {
#endif
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
#ifndef BANCKMARK
    filename = dataSetDir + files + ".txt";
    filename2_1 = dataSetDir + files + "csc_offsets.txt";
    filename2_2 = dataSetDir + files + "csc_columns.txt";
#else
    filename = dataSetDir + files + ".mtx";
    filename2_1 = dataSetDir + files + "_csc_r.mtx";
    filename2_2 = dataSetDir + files + "_csc_cv.mtx";
#endif

    std::cout << "INFO: dataSet path is " << filename << std::endl;
    std::cout << "INFO: dataSet offset path is " << filename2_1 << std::endl;
    std::cout << "INFO: dataSet indice path is " << filename2_2 << std::endl;

#ifndef BANCKMARK
    std::string fileRef;
    fileRef = refDir + "pagerank_ref_tigergraph.txt";
    std::cout << "INFO: reference data path is " << fileRef << std::endl;
#else
    std::string fileRef1;
    fileRef1 = refDir + files + ".tiger";
    std::cout << "INFO: reference data path is " << fileRef1 << std::endl;
#endif
    // Variables to measure time
    struct timeval start_time; // End to end time clock start
    gettimeofday(&start_time, 0);

    CscMatrix<int, float> cscMat;
    readInWeightedDirectedGraphCV<int, float>(filename2_2, cscMat, nnz);
    std::cout << "INFO: ReadIn succeed" << std::endl;
    readInWeightedDirectedGraphRow<int, float>(filename2_1, cscMat, nnz, nrows);

    // Output the inputs information
    std::cout << "INFO: Number of kernel runs: " << num_runs << std::endl;
    std::cout << "INFO: Number of edges: " << nnz << std::endl;
    std::cout << "INFO: Number of nrows: " << nrows << std::endl;

#ifndef BANCKMARK
    DT alpha = 0.85;
    DT tolerance = 1e-3f;
    int maxIter = 20;
#else
    DT alpha = 0.85;
    DT tolerance = 1e-3f;
    int maxIter = 500;
#endif

    int depthNrow = (nrows + 1 + 15) / 16;
    int depthNNZ = (nnz + 15) / 16;
    int sizeNrow = depthNrow * 16;
    int sizeNNZ = depthNNZ * 16;

    // for type width and size
    const int sizeT = sizeof(DT);
    const int widthT = sizeof(DT) * 8;

    ///// declaration
    ap_uint<32>* offsetArr = aligned_alloc<ap_uint<32> >(sizeNrow);
    ap_uint<32>* indiceArr = aligned_alloc<ap_uint<32> >(sizeNNZ);
    float* weightArr = aligned_alloc<float>(sizeNNZ);
    for (int i = 0; i < nnz; ++i) {
        if (i < nrows + 1) {
            offsetArr[i] = cscMat.columnOffset.data()[i];
        } else if (i < sizeNrow) {
            offsetArr[i] = 0;
        }

        indiceArr[i] = cscMat.row.data()[i];
        weightArr[i] = 1.0;
    }

    int iteration = (sizeof(DT) == 8) ? (nrows + 7) / 8 : (nrows + 16 - 1) / 16;
    int unrollNm2 = (sizeof(DT) == 4) ? 16 : 8;
    int iteration2 = (nrows + unrollNm2 - 1) / unrollNm2;
    buffType* cntValFull = aligned_alloc<buffType>(iteration2);
    buffType* buffPing = aligned_alloc<buffType>(iteration2);
    buffType* buffPong = aligned_alloc<buffType>(iteration2);
    DT* pagerank = aligned_alloc<DT>(nrows);
    int* resultInfo = aligned_alloc<int>(2);
    int depthDegree = (nrows + 16 + 15) / 16;
    int sizeDegree = depthDegree * 16;
    int depthOrder = (nrows + 16 + 7) / 8;
    int sizeOrder = depthOrder * 8;
    ap_uint<32>* degreeCSR = aligned_alloc<ap_uint<32> >(sizeDegree);
    ap_uint<32>* orderUnroll = aligned_alloc<ap_uint<32> >(sizeOrder);

    DT* golden = new DT[nrows];

    for (int i = 0; i < nrows; ++i) {
        golden[i] = 0;
        degreeCSR[i] = 0;
        pagerank[i] = 0;
    }

#ifndef BANCKMARK
    readInRef<int, DT>(fileRef, golden, nrows);
#else
    std::vector<int> row;
    std::vector<DT> value;
    readInTigerRef<int, DT>(fileRef1, row, value, golden, nrows);
#endif

#ifdef _HLS_TEST_
    ap_uint<512>* pagerank1 = aligned_alloc<ap_uint<512> >(iteration2);
    ap_uint<512>* degree = reinterpret_cast<ap_uint<512>*>(degreeCSR);
    ap_uint<512>* offsetCSC = reinterpret_cast<ap_uint<512>*>(offsetArr);
    ap_uint<512>* indiceCSC = reinterpret_cast<ap_uint<512>*>(indiceArr);
    ap_uint<512>* weightCSC = reinterpret_cast<ap_uint<512>*>(weightArr);
    const int widthOR = (sizeof(DT) == 8) ? 256 : 512;
    ap_uint<widthOR>* orderUnroll2 = reinterpret_cast<ap_uint<widthOR>*>(orderUnroll);
    std::cout << "kernel start" << std::endl;
    kernel_pagerank_0(nrows, nnz, alpha, tolerance, maxIter, offsetCSC, indiceCSC, weightCSC, degree, cntValFull,
                      buffPing, buffPong, resultInfo, orderUnroll2); // pagerank1,
    bool resultinPong = (bool)(*resultInfo);
    int iterations = (int)(*(resultInfo + 1));
    std::cout << "kernel end" << std::endl;
    int cnt = 0;

    for (int i = 0; i < iteration2; ++i) {
        xf::graph::internal::calc_degree::f_cast<DT> tt;
        ap_uint<512> tmp11 = resultinPong ? buffPong[i] : buffPing[i]; // pagerank1[i];
        for (int k = 0; k < unrollNm2; ++k) {
            if (cnt < nrows) {
                tt.i = tmp11.range(widthT * (k + 1) - 1, widthT * k);
                if (sizeT == 8) {
                    pagerank[cnt] = (DT)(tt.f);
                } else {
                    pagerank[cnt] = (DT)(tt.f);
                }
                cnt++;
            }
        }
    }
    free(pagerank1);
#else
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // Get CL devices.
    cl_int fail;

    // Platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &fail);
    logger.logCreateContext(fail);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &fail);
    logger.logCreateCommandQueue(fail);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &fail);
    logger.logCreateProgram(fail);
    cl::Kernel kernel_pagerank(program, "kernel_pagerank_0", &fail);
    logger.logCreateKernel(fail);
    std::cout << "INFO: Kernel has been created" << std::endl;

#ifndef USE_HBM
    // DDR Settings
    std::vector<cl_mem_ext_ptr_t> mext_in(9);
    mext_in[0].flags = XCL_MEM_DDR_BANK0;
    mext_in[0].obj = offsetArr; // pagerank2;
    mext_in[0].param = 0;
    mext_in[1].flags = XCL_MEM_DDR_BANK0;
    mext_in[1].obj = indiceArr;
    mext_in[1].param = 0;
    mext_in[2].flags = XCL_MEM_DDR_BANK0;
    mext_in[2].obj = weightArr;
    mext_in[2].param = 0;
    mext_in[3].flags = XCL_MEM_DDR_BANK0;
    mext_in[3].obj = degreeCSR;
    mext_in[3].param = 0;
    mext_in[4].flags = XCL_MEM_DDR_BANK0;
    mext_in[4].obj = cntValFull;
    mext_in[4].param = 0;
    mext_in[5].flags = XCL_MEM_DDR_BANK0;
    mext_in[5].obj = buffPing;
    mext_in[5].param = 0;
    mext_in[6].flags = XCL_MEM_DDR_BANK0;
    mext_in[6].obj = buffPong;
    mext_in[6].param = 0;
    mext_in[7].flags = XCL_MEM_DDR_BANK0;
    mext_in[7].obj = resultInfo;
    mext_in[7].param = 0;
    mext_in[8].flags = XCL_MEM_DDR_BANK0;
    mext_in[8].obj = orderUnroll;
    mext_in[8].param = 0;
#else

    std::vector<cl_mem_ext_ptr_t> mext_in(9);
    mext_in[0].flags = XCL_BANK0;
    mext_in[0].obj = offsetArr;
    mext_in[0].param = 0;
    mext_in[1].flags = XCL_BANK2;
    mext_in[1].obj = indiceArr;
    mext_in[1].param = 0;
    mext_in[2].flags = XCL_BANK4;
    mext_in[2].obj = weightArr;
    mext_in[2].param = 0;
    mext_in[3].flags = XCL_BANK6;
    mext_in[3].obj = degreeCSR;
    mext_in[3].param = 0;
    mext_in[4].flags = XCL_BANK8;
    mext_in[4].obj = cntValFull;
    mext_in[4].param = 0;
    mext_in[5].flags = XCL_BANK10;
    mext_in[5].obj = buffPing;
    mext_in[5].param = 0;
    mext_in[6].flags = XCL_BANK12;
    mext_in[6].obj = buffPong;
    mext_in[6].param = 0;
    mext_in[7].flags = XCL_BANK12;
    mext_in[7].obj = resultInfo;
    mext_in[7].param = 0;
    mext_in[8].flags = XCL_BANK1;
    mext_in[8].obj = orderUnroll;
    mext_in[8].param = 0;
#endif

    // clang-format off
    // Create device buffer and map dev buf to host buf
    std::vector<cl::Buffer> buffer(9);

    buffer[0] = cl::Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(ap_uint<32>) * (nrows + 1), offsetArr); // offset// for band to one axi
    buffer[1] = cl::Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           sizeof(ap_uint<32>) * nnz, indiceArr); // indice
    buffer[2] = cl::Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(float) * nnz,
                           weightArr); // weight
    buffer[3] = cl::Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(ap_uint<32>) * nrows, degreeCSR); // degree

    buffer[4] = cl::Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(buffType) * iteration2, cntValFull); // const
    buffer[5] = cl::Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(buffType) * iteration2, buffPing); // buffp
    buffer[6] = cl::Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(buffType) * iteration2, buffPong); // buffq

    buffer[7] = cl::Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * (2),
                           resultInfo); // resultInfo
    buffer[8] = cl::Buffer(context,  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(ap_uint<32>) * (nrows + 16), orderUnroll); // order
    // clang-format on

    // add buffers to migrate
    std::vector<cl::Memory> init;
    for (int i = 0; i < 9; i++) {
        init.push_back(buffer[i]);
    }

    // Data transfer from host buffer to device buffer
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    ob_in.push_back(buffer[0]);
    ob_in.push_back(buffer[1]);
    ob_in.push_back(buffer[2]);

    ob_in.push_back(buffer[3]);
    ob_in.push_back(buffer[4]);
    ob_out.push_back(buffer[5]);
    ob_out.push_back(buffer[6]);
    ob_out.push_back(buffer[7]);

    ob_in.push_back(buffer[8]);

    kernel_pagerank.setArg(0, nrows);
    kernel_pagerank.setArg(1, nnz);
    kernel_pagerank.setArg(2, alpha);
    kernel_pagerank.setArg(3, tolerance);
    kernel_pagerank.setArg(4, maxIter);
    kernel_pagerank.setArg(5, buffer[0]);
    kernel_pagerank.setArg(6, buffer[1]);
    kernel_pagerank.setArg(7, buffer[2]);
    kernel_pagerank.setArg(8, buffer[3]);
    kernel_pagerank.setArg(9, buffer[4]);

    kernel_pagerank.setArg(10, buffer[5]);
    kernel_pagerank.setArg(11, buffer[6]);
    kernel_pagerank.setArg(12, buffer[7]);
    kernel_pagerank.setArg(13, buffer[8]);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);
    q.finish();

    // Setup kernel
    std::cout << "INFO: Finish kernel setup" << std::endl;
    std::vector<cl::Event> events_write(1);
    std::vector<std::vector<cl::Event> > events_kernel(1);
    std::vector<cl::Event> events_read(1);

    events_kernel[0].resize(1);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]); // 0 : migrate from host to dev

    // Launch kernel and compute kernel execution time
    q.enqueueTask(kernel_pagerank, &events_write, &events_kernel[0][0]);

    // Data transfer from device buffer to host buffer
    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel[0], &events_read[0]); // 1 : migrate from dev to host

    q.finish();

    struct timeval end_time;
    gettimeofday(&end_time, 0);
    std::cout << "INFO: Finish kernel execution" << std::endl;
    std::cout << "INFO: Finish E2E execution" << std::endl;

    // print related times
    unsigned long timeStart, timeEnd, exec_time0, write_time, read_time;
    std::cout << "-------------------------------------------------------" << std::endl;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    write_time = (timeEnd - timeStart) / 1000.0;
    std::cout << "INFO: Data transfer from host to device: " << write_time << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    read_time = (timeEnd - timeStart) / 1000.0;
    std::cout << "INFO: Data transfer from device to host: " << read_time << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;

    exec_time0 = 0;
    for (int i = 0; i < num_runs; ++i) {
        events_kernel[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
        events_kernel[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
        exec_time0 += (timeEnd - timeStart) / 1000.0;
    }
    std::cout << "INFO: Average kernel execution per run: " << exec_time0 / num_runs << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    unsigned long exec_timeE2E = diff(&end_time, &start_time);
    std::cout << "INFO: Average execution per run: " << (write_time + exec_time0 + read_time) << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;

    bool resultinPong = (bool)(*resultInfo);
    int iterations = (int)(*(resultInfo + 1));
    std::cout << "resultinPong = " << resultinPong << std::endl;
    std::cout << "iterations = " << iterations << std::endl;

    int cnt = 0;
    for (int i = 0; i < iteration2; ++i) {
        xf::graph::internal::calc_degree::f_cast<DT> tt;
        ap_uint<512> tmp11 = resultinPong ? buffPong[i] : buffPing[i]; // pagerank1[i];
        for (int k = 0; k < unrollNm2; ++k) {
            if (cnt < nrows) {
                tt.i = tmp11.range(widthT * (k + 1) - 1, widthT * k);
                if (sizeT == 8) {
                    pagerank[cnt] = (DT)(tt.f);
                } else {
                    pagerank[cnt] = (DT)(tt.f);
                }
                cnt++;
            }
        }
    }

#endif
    //    for (int i = 0; (i < nrows) && (i < 100); ++i) {
    //        std::cout << "pagerank i = " << i << "\t our = " << pagerank[i] << "\t golden = " << golden[i] <<
    //        std::endl;
    //    }

    double sum2 = 0.0;
    for (int i = 0; i < nrows; ++i) {
        sum2 += golden[i];
    }

    std::fstream fin("pagerank1.output", std::ios::out);
    if (!fin) {
        std::cout << "Error : file doesn't exist !" << std::endl;
        exit(1);
    }
    double sum3 = 0.0;
    for (int i = 0; i < nrows; ++i) {
        sum3 += pagerank[i];
        fin << i << "  ";
        fin << pagerank[i] << "\n";
    }

    fin.close();

    std::cout << "INFO: sum_golden = " << sum2 << std::endl;
    std::cout << "INFO: sum_pagerank = " << sum3 << std::endl;

    // Calculate err
    DT err = 0.0;
    int accurate = 0;
    for (int i = 0; i < nrows; ++i) {
        err += (golden[i] - pagerank[i]) * (golden[i] - pagerank[i]);
        if (std::abs(pagerank[i] - golden[i]) < tolerance) {
            accurate += 1;
        } else {
            std::cout << "pagerank i = " << i << "\t our = " << pagerank[i] << "\t golden = " << golden[i] << std::endl;
        }
    }
    DT accRate = accurate * 1.0 / nrows;
    err = std::sqrt(err);
    std::cout << "INFO: Accurate Rate = " << accRate << std::endl;
    std::cout << "INFO: Err Geomean = " << err << std::endl;

    free(offsetArr);
    free(indiceArr);
    free(weightArr);
    free(orderUnroll);
    free(cntValFull);
    free(buffPing);
    free(buffPong);
    free(pagerank);
    free(degreeCSR);
    delete[] golden;

    if (err < nrows * tolerance) {
        std::cout << "INFO: Result is correct" << std::endl;
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
        return 0;
    } else {
        std::cout << "INFO: Result is wrong" << std::endl;
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
        return 1;
    }
}
#endif
