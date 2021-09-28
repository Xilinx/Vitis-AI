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
#include "utils.hpp"
#ifndef HLS_TEST
#include "xcl2.hpp"
#endif

#include <math.h>
#include "kernel_mceuropeanengine.hpp"
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
int print_result(int cu_number, std::vector<double*>& out, double golden, double max_tol) {
    std::cout << "FPGA result:\n";
    for (int i = 0; i < cu_number; ++i) {
        if (std::fabs(out[i][0] - golden) > max_tol) {
            std::cout << "            Kernel " << i << " - " << out[i][0] << "            golden - " << golden
                      << std::endl;
            return 1;
        } else {
            std::cout << "            Kernel " << i << " - " << out[i][0] << std::endl;
        }
    }
    return 0;
}

int main(int argc, const char* argv[]) {
    std::cout << "\n----------------------MC(European) Engine-----------------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // cmd parser
    ArgParser parser(argc, argv);
    std::string xclbin_path;

    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

    struct timeval st_time, end_time;

    // test data
    unsigned int timeSteps = 1;
    DtUsed requiredTolerance = 0.02;
    DtUsed underlying = 36;
    DtUsed riskFreeRate = 0.06;
    DtUsed volatility = 0.20;
    DtUsed dividendYield = 0.0;
    DtUsed strike = 40;
    unsigned int optionType = 1;
    DtUsed timeLength = 1;
    unsigned int seeds[4] = {4332, 441242, 42, 13342};
    unsigned int requiredSamples = 0; // 262144; // 48128;//0;//1024;//0;
    unsigned int maxSamples = 0;
    //
    unsigned int loop_nm = 1024;
    std::string mode_emu = "hw";
    if (std::getenv("XCL_EMULATION_MODE") != nullptr) {
        mode_emu = std::getenv("XCL_EMULATION_MODE");
    }

    int num_rep = 20;
    std::string num_str;
    if (parser.getCmdOption("-rep", num_str)) {
        try {
            num_rep = std::stoi(num_str);
        } catch (...) {
            num_rep = 1;
        }
    }
    DtUsed max_diff = requiredTolerance;

    cl_int cl_err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);

#ifdef SW_EMU_TEST
    // hls::exp and hls::log have bug in multi-thread.
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE,
                       &cl_err); // | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
#else
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
#endif
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();

    std::cout << "Selected Device " << devName << "\n";

    cl::Program::Binaries xclbins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);

    cl::Program program(context, devices, xclbins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);

    std::string krnl_name = "kernel_mc";
    cl_uint cu_number;
    {
        cl::Kernel k(program, krnl_name.c_str());
        k.getInfo(CL_KERNEL_COMPUTE_UNIT_COUNT, &cu_number);
    }
    if (mode_emu.compare("hw_emu") == 0) {
        loop_nm = 1;
        num_rep = cu_number;
        requiredSamples = 1024 * MCM_NM;
        max_diff = 0.06;
    } else if (mode_emu.compare("sw_emu") == 0) {
        loop_nm = 1;
        num_rep = cu_number * 3;
    }

    std::cout << "loop_nm = " << loop_nm << std::endl;
    std::cout << "num_rep = " << num_rep << std::endl;
    std::cout << "cu_number = " << cu_number << std::endl;
    std::vector<cl::Kernel> krnl0(cu_number);
    std::vector<cl::Kernel> krnl1(cu_number);

    for (cl_uint i = 0; i < cu_number; ++i) {
        std::string krnl_full_name = krnl_name + ":{" + krnl_name + "_" + std::to_string(i + 1) + "}";
        krnl0[i] = cl::Kernel(program, krnl_full_name.c_str(), &cl_err);
        krnl1[i] = cl::Kernel(program, krnl_full_name.c_str(), &cl_err);
        logger.logCreateKernel(cl_err);
    }
    std::cout << "Kernel has been created\n";

    std::vector<DtUsed*> out_a(cu_number);
    std::vector<DtUsed*> out_b(cu_number);
    for (int i = 0; i < cu_number; ++i) {
        out_a[i] = aligned_alloc<DtUsed>(OUTDEP);
        out_b[i] = aligned_alloc<DtUsed>(OUTDEP);
    }
    std::vector<cl_mem_ext_ptr_t> mext_out_a(cu_number);
    std::vector<cl_mem_ext_ptr_t> mext_out_b(cu_number);
    for (int i = 0; i < cu_number; ++i) {
        mext_out_a[i] = {9, out_a[i], krnl0[i]()};
        mext_out_b[i] = {9, out_b[i], krnl1[i]()};
    }
    std::vector<cl::Buffer> out_buff_a(cu_number);
    std::vector<cl::Buffer> out_buff_b(cu_number);
    for (int i = 0; i < cu_number; i++) {
        out_buff_a[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   (size_t)(OUTDEP * sizeof(DtUsed)), &mext_out_a[i]);
        out_buff_b[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   (size_t)(OUTDEP * sizeof(DtUsed)), &mext_out_b[i]);
    }
    std::vector<std::vector<cl::Event> > kernel_events(num_rep);
    std::vector<std::vector<cl::Event> > read_events(num_rep);
    for (int i = 0; i < num_rep; ++i) {
        kernel_events[i].resize(cu_number);
        read_events[i].resize(1);
    }
    for (int i = 0; i < cu_number; ++i) {
        int j = 0;
        krnl0[i].setArg(j++, loop_nm);
        krnl0[i].setArg(j++, seeds[i]);
        krnl0[i].setArg(j++, underlying);
        krnl0[i].setArg(j++, volatility);
        krnl0[i].setArg(j++, dividendYield);
        krnl0[i].setArg(j++, riskFreeRate);
        krnl0[i].setArg(j++, timeLength);
        krnl0[i].setArg(j++, strike);
        krnl0[i].setArg(j++, optionType);
        krnl0[i].setArg(j++, out_buff_a[i]);
        krnl0[i].setArg(j++, requiredTolerance);
        krnl0[i].setArg(j++, requiredSamples);
        krnl0[i].setArg(j++, timeSteps);
        krnl0[i].setArg(j++, maxSamples);
    }
    for (int i = 0; i < cu_number; ++i) {
        int j = 0;
        krnl1[i].setArg(j++, loop_nm);
        krnl1[i].setArg(j++, seeds[i]);
        krnl1[i].setArg(j++, underlying);
        krnl1[i].setArg(j++, volatility);
        krnl1[i].setArg(j++, dividendYield);
        krnl1[i].setArg(j++, riskFreeRate);
        krnl1[i].setArg(j++, timeLength);
        krnl1[i].setArg(j++, strike);
        krnl1[i].setArg(j++, optionType);
        krnl1[i].setArg(j++, out_buff_b[i]);
        krnl1[i].setArg(j++, requiredTolerance);
        krnl1[i].setArg(j++, requiredSamples);
        krnl1[i].setArg(j++, timeSteps);
        krnl1[i].setArg(j++, maxSamples);
    }

    std::vector<cl::Memory> out_vec_a; //{out_buff[0]};
    std::vector<cl::Memory> out_vec_b; //{out_buff[0]};
    for (int i = 0; i < cu_number; ++i) {
        out_vec_a.push_back(out_buff_a[i]);
        out_vec_b.push_back(out_buff_b[i]);
    }
    q.finish();
    gettimeofday(&st_time, 0);
    for (int i = 0; i < num_rep / cu_number; ++i) {
        int use_a = i & 1;
        if (use_a) {
            if (i > 1) {
                for (int c = 0; c < cu_number; ++c) {
                    q.enqueueTask(krnl0[c], &read_events[i - 2], &kernel_events[i][c]);
                }
            } else {
                for (int c = 0; c < cu_number; ++c) {
                    q.enqueueTask(krnl0[c], nullptr, &kernel_events[i][c]);
                }
            }
        } else {
            if (i > 1) {
                for (int c = 0; c < cu_number; ++c) {
                    q.enqueueTask(krnl1[c], &read_events[i - 2], &kernel_events[i][c]);
                }
            } else {
                for (int c = 0; c < cu_number; ++c) {
                    q.enqueueTask(krnl1[c], nullptr, &kernel_events[i][c]);
                }
            }
        }
        if (use_a) {
            q.enqueueMigrateMemObjects(out_vec_a, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[i], &read_events[i][0]);
        } else {
            q.enqueueMigrateMemObjects(out_vec_b, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[i], &read_events[i][0]);
        }
    }

    q.flush();
    q.finish();
    gettimeofday(&end_time, 0);
    int exec_time = tvdiff(&st_time, &end_time);
    double time_elapsed = double(exec_time) / 1000 / 1000;
    std::cout << "FPGA execution time: " << time_elapsed << " s\n"
              << "options number: " << loop_nm * num_rep << " \n"
              << "opt/sec: " << double(loop_nm * num_rep) / time_elapsed << std::endl;
    DtUsed golden = 3.834522;
    std::cout << "Expected value: " << golden << ::std::endl;
    int err = 0;
    if (num_rep > cu_number) {
        err += print_result(cu_number, out_a, golden, max_diff);
    }
    err += print_result(cu_number, out_b, golden, max_diff);
    err ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return err;
}
