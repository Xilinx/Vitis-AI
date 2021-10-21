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
#ifndef HLS_TEST
#include "xcl2.hpp"
#endif
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "ap_int.h"
#include "utils.hpp"
#include "cpi_capfloor_engine_kernel.hpp"
#include "xf_utils_sw/logger.hpp"

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
    std::cout << "\n----------------------CPI CapFloor Engine-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }
    // Allocate Memory in Host Memory
    double* times_alloc = aligned_alloc<double>(LEN);
    double* strikes_alloc = aligned_alloc<double>(LEN);
    double* prices_alloc = aligned_alloc<double>(LEN * LEN);
    DT* output = aligned_alloc<DT>(1);

    // -------------setup k0 params---------------
    int err = 0;
    DT minErr = 10e-10;
    int xSize = 7;
    int ySize = 8;

    double golden = 0.022759999999999999;

    DT cfMaturityTimes[7] = {3.0054794520547947, 5, 7, 10.001601916311101, 15.002739726027396, 20.005479452054796,
                             30.001601916311103};
    DT cfStrikes[8] = {-0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06};
    DT cPriceB[56] = {0.1189026727532132,
                      0.19234884598336022,
                      0.25893190845247194,
                      0.34716101441352776,
                      0.45149515082144476,
                      0.54984156444044396,
                      0.68506774112094337,
                      0.092142126147838899,
                      0.15086931377480772,
                      0.20609223629086393,
                      0.28298752968117147,
                      0.37776538469755194,
                      0.47195914772445069,
                      0.60883382967607202,
                      0.066062203731440894,
                      0.10909964072251221,
                      0.15160923940178606,
                      0.21468337452033526,
                      0.29520946272431492,
                      0.37985640392630282,
                      0.50921195588585078,
                      0.041986241745908481,
                      0.068822646085233807,
                      0.097601702943668323,
                      0.14444396141115279,
                      0.20582824893879248,
                      0.27381429347506581,
                      0.38239007859277974,
                      0.022759999999999999,
                      0.034532,
                      0.047794999999999997,
                      0.075781000000000001,
                      0.11407300000000001,
                      0.15375999999999998,
                      0.221167,
                      0.010026999999999999,
                      0.012790000000000001,
                      0.017018999999999999,
                      0.030394999999999998,
                      0.048188999999999996,
                      0.060772,
                      0.083923999999999999,
                      0.0038799999999999998,
                      0.0040590000000000001,
                      0.0050619999999999997,
                      0.010762000000000001,
                      0.016840000000000001,
                      0.017227000000000003,
                      0.018474999999999998,
                      0.0014939999999999999,
                      0.0014109999999999999,
                      0.0016879999999999998,
                      0.0043610000000000003,
                      0.006365,
                      0.0054869999999999997,
                      0.0045030000000000001};
    DT fPriceB_[56] = {0.001562,
                       0.0021449999999999998,
                       0.0024450000000000001,
                       0.0039249999999999997,
                       0.0036819999999999999,
                       0.0039700000000000004,
                       0.0041479999999999998,
                       0.0028379999999999998,
                       0.0036729999999999996,
                       0.0042079999999999999,
                       0.006352,
                       0.0063619999999999996,
                       0.0067469999999999995,
                       0.0073900000000000007,
                       0.0053610000000000003,
                       0.0066660000000000001,
                       0.0077040000000000008,
                       0.010920000000000001,
                       0.011696999999999999,
                       0.012179000000000001,
                       0.013975,
                       0.010459999999999999,
                       0.012959999999999999,
                       0.015224000000000001,
                       0.020344000000000001,
                       0.023272999999999999,
                       0.023855999999999999,
                       0.028674999999999999,
                       0.020986423566868195,
                       0.027102914440439996,
                       0.030672544709074989,
                       0.038692580066551074,
                       0.047326314871280251,
                       0.04562055531365905,
                       0.055501689276864274,
                       0.038589455964999964,
                       0.055712359250414734,
                       0.069065840642156262,
                       0.088262673232410438,
                       0.11411008563692915,
                       0.12317631764911008,
                       0.16744489410569519,
                       0.063367519206596779,
                       0.099307091363403766,
                       0.13038590020945451,
                       0.1721686783337858,
                       0.23454395102070713,
                       0.2843548261064861,
                       0.43130230651061563,
                       0.092501277049768627,
                       0.15101660895125213,
                       0.20459803401350229,
                       0.27857239514041832,
                       0.3974986330688387,
                       0.5179412029467938,
                       0.85136429970596605};

    DT r = 0.03;
    DT t = 3.0054794520547947;

    for (int i = 0; i < xSize; i++) {
        times_alloc[i] = cfMaturityTimes[i];
    }

    for (int i = 0; i < ySize; i++) {
        strikes_alloc[i] = cfStrikes[i];
    }

    for (int i = 0; i < xSize * ySize; i++) {
        prices_alloc[i] = cPriceB[i];
    }

#ifndef HLS_TEST
    cl_int cl_err;
    // do pre-process on CPU
    struct timeval start_time, end_time, test_time;
    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    // cl::Program::Binaries xclBins = xcl::import_binary_file("../xclbin/MCAE_u250_hw.xclbin");
    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);
    cl::Kernel kernel_CPIEngine(program, "CPI_k0", &cl_err);
    logger.logCreateKernel(cl_err);
    std::cout << "kernel has been created" << std::endl;

    cl_mem_ext_ptr_t mext_o[4];
    mext_o[0] = {7, output, kernel_CPIEngine()};
    mext_o[1] = {2, times_alloc, kernel_CPIEngine()};
    mext_o[2] = {3, strikes_alloc, kernel_CPIEngine()};
    mext_o[3] = {4, prices_alloc, kernel_CPIEngine()};

    // create device buffer and map dev buf to host buf
    cl::Buffer output_buf;
    cl::Buffer times_buf, strikes_buf, prices_buf;
    output_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * N,
                            &mext_o[0]);
    times_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * LEN,
                           &mext_o[1]);
    strikes_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(DT) * LEN,
                             &mext_o[2]);
    prices_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            sizeof(DT) * LEN * LEN, &mext_o[3]);

    std::vector<cl::Memory> ob_out;
    ob_out.push_back(output_buf);

    q.finish();
    // launch kernel and calculate kernel execution time
    std::cout << "kernel start------" << std::endl;
    gettimeofday(&start_time, 0);
    for (int i = 0; i < 1; ++i) {
        kernel_CPIEngine.setArg(0, xSize);
        kernel_CPIEngine.setArg(1, ySize);
        kernel_CPIEngine.setArg(2, times_buf);
        kernel_CPIEngine.setArg(3, strikes_buf);
        kernel_CPIEngine.setArg(4, prices_buf);
        kernel_CPIEngine.setArg(5, t);
        kernel_CPIEngine.setArg(6, r);
        kernel_CPIEngine.setArg(7, output_buf);
        q.enqueueTask(kernel_CPIEngine, nullptr, nullptr);
    }

    q.finish();
    gettimeofday(&end_time, 0);
    std::cout << "kernel end------" << std::endl;
    std::cout << "Execution time " << tvdiff(&start_time, &end_time) << "us" << std::endl;
    q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr);
    q.finish();

#else
    CPI_k0(xSize, ySize, cfMaturityTimes, cfStrikes, cPriceB, t, r, output);
#endif
    DT out = output[0];
    if (std::fabs(out - golden) > minErr) err++;
    std::cout << "NPV= " << out << " ,diff/NPV= " << (out - golden) / golden << std::endl;
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return err;
}
