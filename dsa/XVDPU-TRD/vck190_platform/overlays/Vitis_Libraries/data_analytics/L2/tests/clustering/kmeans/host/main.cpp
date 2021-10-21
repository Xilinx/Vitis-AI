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

#include <sys/time.h>
#include <new>
#include <cstdlib>
#include <algorithm>

#ifndef HLS_TEST
#include <xcl2.hpp>
#endif

#include <vector>

#define XF_DATA_ANALYTICS_DEBUG 1
#include "../kernel/config.hpp"
#include <ap_int.h>
#include <iostream>
#include "eval.hpp"
#include "iris.hpp"
#include "xf_utils_sw/logger.hpp"

#ifdef HLS_TEST
extern "C" void kmeansKernel(ap_uint<512> inputData[(1 << 20) + 100], ap_uint<512> centers[1 << 20]);
#endif
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

inline int tvdiff(struct timeval* tv0, struct timeval* tv1) {
    return (tv1->tv_sec - tv0->tv_sec) * 1000000 + (tv1->tv_usec - tv0->tv_usec);
}
/*
void combineConfig( ap_uint<512> &config,  int kcluster,int dim, int  nsample,int maxIter,DType eps){
    config.range(31,0)=kcluster;
    config.range(63,32)=dim;
    config.range(95,64)=nsample;
    config.range(127,96)=maxIter;
    const int sz = sizeof(DType) * 8;
    conv<DType,sz> nt;
    nt.dt=eps;
    config.range(sz+127,128)=nt.ut;
}
*/
template <typename T>
T* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
}

int main(int argc, char* argv[]) {
    // cmd parser
    ArgParser parser(argc, (const char**)argv);
#ifndef HLS_TEST
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
#endif
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // set repeat time
    int num_rep = 1;
    std::string num_str;
    if (parser.getCmdOption("-rep", num_str)) {
        try {
            num_rep = std::stoi(num_str);
        } catch (...) {
            num_rep = 2;
        }
    }
    if (num_rep < 2) {
        num_rep = 2;
        std::cout << "WARNING: ping-pong buffer shoulde be updated at least 1 time.\n";
    }
    if (num_rep > 20) {
        num_rep = 20;
        std::cout << "WARNING: limited repeat to " << num_rep << " times.\n";
    }
    // num_rep = 1;

    int spNum = NS; // 20000;
    std::string num_str2;
    if (parser.getCmdOption("-num", num_str2)) {
        try {
            spNum = std::stoi(num_str2);
        } catch (...) {
            spNum = NS; // 2000 * 1000;
            std::cout << "WARNING: spNum=" << spNum << std::endl;
        }
    }
    //    num_rep = 1;

    int dim = DIM;
    int kcluster = KC;
    int nsample = NS;
    int maxNS = spNum;
    nsample = maxNS;
    // dim = DIM - 2;
    // kcluster = KC - 2;
    const int sz = sizeof(DType) * 8;
    int dsz = dim * sz;
    int numIn512 = 512 / dsz;
    int ND = (maxNS * DIM * sizeof(DType) * 8 + 511) / 512;
    const int NC = 1 + (KC + PCU) * ((DIM * sizeof(DType) * 8 + 511) / 512);
    int buffSize = ND + NC + 1;
    int numDataBlock = (nsample * dsz + 511) / 512;
    int numCenterBlock = (kcluster * dsz + 511) / 512;
    int maxIter = 30; // 1000;
    DType eps = 1e-8;

#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "numIn512=" << numIn512 << std::endl;
    std::cout << "numCenterBlock=" << numCenterBlock << std::endl;
    std::cout << "numDataBlock=" << numDataBlock << std::endl;
    std::cout << "KC=" << KC << "   NS=" << NS << std::endl;
    std::cout << "ND=" << ND << "   NC=" << NC << std::endl;
    std::cout << "maxNs=" << maxNS << " nsample=" << nsample << "  kcluster=" << kcluster << "  dim=" << dim
              << "  buffSize=" << buffSize << std::endl;
#endif
    ap_uint<512> config = 0;
    ap_uint<512>* inputData = aligned_alloc<ap_uint<512> >(buffSize);
    ap_uint<512>* centers = aligned_alloc<ap_uint<512> >(NC);
    ap_uint<32>* gtag = (ap_uint<32>*)malloc(sizeof(ap_uint<32>) * maxNS);
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    int cdsz = DIM * sz;
    DType** x = (DType**)malloc(sizeof(DType*) * maxNS);
    for (int i = 0; i < nsample; i++) {
        x[i] = (DType*)malloc(cdsz);
    }
    DType** c = (DType**)malloc(sizeof(DType*) * KC);
    for (int i = 0; i < KC; i++) {
        c[i] = (DType*)malloc(cdsz);
    }
    DType** nc = (DType**)malloc(sizeof(DType*) * KC);
    for (int i = 0; i < KC; i++) {
        nc[i] = (DType*)malloc(cdsz);
    }
    DType** gnc = (DType**)malloc(sizeof(DType*) * KC);
    for (int i = 0; i < KC; i++) {
        gnc[i] = (DType*)malloc(cdsz);
    }
#else

    ap_uint<512> data[1 + ND + NC];
    ap_uint<512> center[NC];
    ap_uint<32> gtag[NS];
    DType x[NS][DIM];
    DType c[KC][DIM];
    DType nc[KC][DIM];
    DType gnc[KC][DIM];

#endif
    combineConfig(config, kcluster, dim, nsample, maxIter, eps);
    for (int i = 0; i < nsample; ++i) {
        for (int j = 0; j < dim; ++j) {
            // x[i][j] = 0.5 + (i * 131 + j) % 1000;
            int ii = i % 150;
            int jj = j % 4;
            x[i][j] = irisVec[ii][jj];
        }
    }

    for (int i = 0; i < KC; ++i) {
        for (int j = 0; j < dim; ++j) {
            // DType t = 0.5 + (i * 131 + j) % 1000;
            // c[i][j] = t;
            int ii = i % 150;
            int jj = j % 4;
            c[i][j] = irisVec[ii][jj];
        }
    }

    int kid = 0;
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "numIn512=" << numIn512 << std::endl;
    std::cout << "numCenterBlock=" << numCenterBlock << std::endl;
    std::cout << "numDataBlock=" << numDataBlock << std::endl;
    std::cout << "KC=" << KC << "   NS=" << NS << std::endl;
    std::cout << "ND=" << ND << "   NC=" << NC << std::endl;
    std::cout << "nsample=" << nsample << "  kcluster=" << kcluster << "  dim=" << dim << "  buffSize=" << buffSize
              << std::endl;
    for (int k = 0; k < KC; ++k) {
        std::cout << "k=" << k << "   c=(";
        for (int j = 0; j < DIM; ++j) {
            DType cd = c[k][j];
            std::cout << cd;
            if (j < DIM - 1) std::cout << ",";
        }
        std::cout << ")" << std::endl;
    }
#endif
    inputData[0] = config;
    convertVect2axi(c, dim, kcluster, 1, inputData);
    int numCB = (kcluster * dsz + 511) / 512;
    convertVect2axi(x, dim, nsample, numCB + 1, inputData);

    std::cout << "Host map buffer has been allocated and set.\n";
#ifndef HLS_TEST
    cl_int cl_err;
    // Get CL devices.
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Create context and command queue for selected device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected Device " << devName << "\n";

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);

    cl::Kernel kernel0(program, "kmeansKernel", &cl_err);
    logger.logCreateKernel(cl_err);

#ifdef USE_DDR
    cl_mem_ext_ptr_t mext_in;
    mext_in = {XCL_MEM_DDR_BANK0, inputData, 0};

    cl_mem_ext_ptr_t mext_out;
    mext_out = {XCL_MEM_DDR_BANK0, centers, 0};
#else
    cl_mem_ext_ptr_t mext_in;
    mext_in = {(unsigned int)(0), inputData, 0};

    cl_mem_ext_ptr_t mext_out;
    mext_out = {(unsigned int)(0), centers, 0};
#endif
    cl::Buffer in_buff;
    cl::Buffer out_buff;

    // Map buffers
    in_buff = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                         (size_t)(sizeof(ap_uint<512>) * (buffSize)), &mext_in);
    out_buff = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                          (size_t)(sizeof(ap_uint<512>) * NC), &mext_out);

    std::cout << "DDR buffers have been mapped/copy-and-mapped\n";
    q.flush();
    q.finish();
#endif

    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);

#ifndef HLS_TEST
    std::vector<std::vector<cl::Event> > write_events(num_rep);
    std::vector<std::vector<cl::Event> > kernel_events(num_rep);
    std::vector<std::vector<cl::Event> > read_events(num_rep);
    for (int i = 0; i < num_rep; i++) {
        write_events[i].resize(1);
        kernel_events[i].resize(1);
        read_events[i].resize(1);
    }
    std::cout << "num_rep " << num_rep << std::endl;
    /*
     */

    for (int i = 0; i < num_rep; i++) {
        // write inputData to DDR
        std::vector<cl::Memory> ib;
        ib.push_back(in_buff);
        q.enqueueMigrateMemObjects(ib, 0, nullptr, &write_events[i][0]);
        // set args and enqueue kernel

        int j = 0;
        kernel0.setArg(j++, in_buff);
        kernel0.setArg(j++, out_buff);

        q.enqueueTask(kernel0, &write_events[i], &kernel_events[i][0]);
        // read data from DDR
        std::vector<cl::Memory> ob;
        ob.push_back(out_buff);

        q.enqueueMigrateMemObjects(ob, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[i], &read_events[i][0]);
    }
#endif
    // wait all to finish
    q.flush();
    q.finish();

#ifdef HLS_TEST
    kmeansKernel(inputData, centers);
#endif
    gettimeofday(&end_time, 0);

    int res = 0;
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "-------  cal golden ----" << std::endl;
    goldenTrain(x, c, gnc, dim, kcluster, nsample, gtag, eps, maxIter);
    convert2array(dim, kcluster, centers, nc);
    std::cout << "KC=" << KC << "   NS=" << NS << std::endl;
    for (int k = 0; k < KC; ++k) {
        std::cout << "k=" << k << "   c=(";
        for (int j = 0; j < DIM; ++j) {
            DType c1 = nc[k][j];
            DType c2 = gnc[k][j];
            if (c1 != c2) res++;
            std::cout << std::dec << c1 << "(" << c2 << ")";
            if (j < DIM - 1) std::cout << ",";
        }
        std::cout << ")" << std::endl;
    }
    if (res == 0)
        std::cout << "PASS" << std::endl;
    else
        std::cout << "FAIL" << std::endl;
    std::cout << "Kernel has been run for " << std::dec << num_rep << " times." << std::endl;
    std::cout << "Total execution time " << tvdiff(&start_time, &end_time) << "us" << std::endl;
#endif

#ifndef HLS_TEST
    free(inputData);
    free(centers);
    for (int i = 0; i < maxNS; i++) free(x[i]);

    for (int i = 0; i < KC; i++) {
        free(c[i]);
        free(nc[i]);
        free(gnc[i]);
    }
    free(x);
    free(c);
    free(nc);
    free(gnc);
    free(gtag);

#endif
    res ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return res;
}
