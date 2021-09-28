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

#include "xf_data_analytics/classification/svm_train.hpp"
#include "xf_data_analytics/common/utils.hpp"
#include "utils.hpp"
#include <CL/cl_ext_xilinx.h>
#include <xcl2.hpp>
#include "xf_utils_sw/logger.hpp"

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

int main(int argc, const char* argv[]) {
    //
    std::cout << "\n--------- SVM Test ---------\n";

    // cmd arg parser.
    ArgParser parser(argc, argv);
    std::string xclbin_path; // eg. q5kernel_VCU1525_hw.xclbin
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }

    std::string in_dir;
    //    if (!parser.getCmdOption("-in", in_dir) || !is_dir(in_dir)) {
    if (!parser.getCmdOption("-in", in_dir)) {
        std::cout << "ERROR: input dir is not specified or not valid.\n";
        return 1;
    }
    std::string trn;
    if (!parser.getCmdOption("-trn", trn)) {
        std::cout << "ERROR: trn is not specified or not valid.\n";
        return 1;
        // samples_num = std::stoi(trn);
    }
    std::string ten;
    if (!parser.getCmdOption("-ten", ten)) {
        std::cout << "ERROR: ten is not specified or not valid.\n";
        return 1;
        // test_num = std::stoi(samples_num);
    }
    std::string fn;
    if (!parser.getCmdOption("-fn", fn)) {
        std::cout << "ERROR: fn is not specified or not valid.\n";
        return 1;
        // test_num = std::stoi(samples_num);
    }
    std::string itrn;
    if (!parser.getCmdOption("-itrn", itrn)) {
        std::cout << "ERROR: cn is not specified or not valid.\n";
        return 1;
        // test_num = std::stoi(samples_num);
    }
    std::string bn_s;
    int bn;
    if (!parser.getCmdOption("-bn", bn_s)) {
        bn_s = "1";
    }
    bn = std::stoi(bn_s);
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int cl_err;
    // Get CL devices.
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Create context and command queue for selected device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    cl::CommandQueue q(context, device,
                       // CL_QUEUE_PROFILING_ENABLE);
                       CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected Device " << devName << "\n";

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    std::vector<cl::Device> devices_h;
    devices_h.push_back(device);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);

    cl::Kernel kernel;
    kernel = cl::Kernel(program, "SVM", &cl_err);
    logger.logCreateKernel(cl_err);

    // Allocate Memory in Host Memory
    ap_uint<64> samples_num = std::stoi(trn);
    ap_uint<64> features_num = std::stoi(fn);
    ap_uint<64>* datasets = (ap_uint<64>*)malloc(sizeof(ap_uint<64>) * samples_num * (features_num + 1));
    double* datasets_D = (double*)malloc(sizeof(double) * samples_num * (features_num + 1));
    std::ifstream fin(in_dir);
    std::string line;
    int row = 0;
    int col = 0;
    while (getline(fin, line)) {
        std::istringstream sin(line);
        std::string attr_val;
        col = 0;
        while (getline(sin, attr_val, ' ')) {
            size_t pos = attr_val.find(':');
            if (pos != attr_val.npos) {
                attr_val = attr_val.substr(pos + 1);
            }
            f_cast<double> w;
            w.f = (std::atof(attr_val.c_str()));
            datasets[(features_num + 1) * row + col] = w.i;
            datasets_D[(features_num + 1) * row + col] = (std::atof(attr_val.c_str()));
            col++;
        }
        row++;
    }
    // test csv read
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << datasets_D[i * (features_num + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "rows num:" << row << std::endl;
    std::cout << "cols num:" << col << std::endl;
    ap_uint<512> configs[20];
    ap_uint<512>* data = aligned_alloc<ap_uint<512> >(sizeof(ap_uint<512>) * samples_num * (features_num / 8 + 1));
    // config:
    ap_uint<512> cfg;
    cfg(63, 0) = features_num;      // feature_number
    cfg(127, 64) = std::stoi(itrn); // max_iteration_number
    cfg(191, 128) = samples_num;    // sample_number
    f_cast<double> w;
    w.f = 1.0;
    cfg(255, 192) = w.i; // step_size
    w.f = 0.01;
    cfg(319, 256) = w.i; // reg_para
    w.f = 0.001;
    cfg(383, 320) = w.i;              // tolerence
    cfg(447, 384) = 2;                // offset
    cfg(511, 448) = features_num + 1; // columns
    data[0] = cfg;

    cfg = 0;
    f_cast<float> ff;
    ff.f = 1.0;
    cfg(31, 0) = ff.i; // fraction
    cfg(63, 32) = 0;   // ifJump
    cfg(95, 64) = 3;   // bucket_size
    cfg(127, 96) = 43; // seed+1
    data[1] = cfg;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int position = (features_num + 1) * i + j;
            if (j != 0) {
                data[(position - 1) / 8 + 2]((position - 1) % 8 * 64 + 63, (position - 1) % 8 * 64) =
                    datasets[(features_num + 1) * i + j];
            } else {
                if (datasets[(features_num + 1) * i] == 0)
                    data[(position + col - 1) / 8 + 2]((position + col - 1) % 8 * 64 + 63,
                                                       (position + col - 1) % 8 * 64) = 0;
                else
                    data[(position + col - 1) / 8 + 2]((position + col - 1) % 8 * 64 + 63,
                                                       (position + col - 1) % 8 * 64) = 1;
            }
        }
    }
    ap_uint<512>* weight = aligned_alloc<ap_uint<512> >(features_num / 8 + 1);
    double* init_weight = aligned_alloc<double>(features_num);
    for (int i = 0; i < features_num; i++) {
        init_weight[i] = 0.0;
        f_cast<double> w;
        w.f = init_weight[i];
        weight[i / 8](i % 8 * 64 + 63, i % 8 * 64) = w.i;
    }
#ifdef USE_DDR
    cl_mem_ext_ptr_t mext_data = {XCL_BANK0, data, 0};
    cl_mem_ext_ptr_t mext_weight = {XCL_BANK0, weight, 0};
#else
    cl_mem_ext_ptr_t mext_data = {(unsigned int)(0), data, 0};
    cl_mem_ext_ptr_t mext_weight = {(unsigned int)(1), weight, 0};
#endif
    // Map buffers
    int datasize = samples_num * (features_num / 8 + 1);
    int weightsize = (features_num / 8 + 1);
    int err;
    cl::Buffer buf_data(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,

                        (size_t)(sizeof(ap_uint<512>) * datasize), &mext_data, &err);
    printf("creating buf_data\n");

    cl::Buffer buf_weight(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                          (size_t)(sizeof(ap_uint<512>) * weightsize), &mext_weight, &err);
    printf("creating buf_weight\n");
    q.finish();
    std::cout << "DDR buffers have been mapped/copy-and-mapped\n";

    int num_rep = 1;
    std::vector<std::vector<cl::Event> > write_events(num_rep);
    std::vector<std::vector<cl::Event> > kernel_events(num_rep);
    std::vector<std::vector<cl::Event> > read_events(num_rep);
    for (int i = 0; i < num_rep; ++i) {
        write_events[i].resize(1);
        kernel_events[i].resize(1);
        read_events[i].resize(1);
    }
    std::vector<cl::Memory> buffwrite;
    buffwrite.push_back(buf_data);
    buffwrite.push_back(buf_weight);

    int j = 0;
    kernel.setArg(j++, buf_data);
    kernel.setArg(j++, buf_weight);

    struct timeval tv_r_s, tv_r_e;
    gettimeofday(&tv_r_s, 0);

    q.enqueueMigrateMemObjects(buffwrite, 0, nullptr, &write_events[0][0]);
    q.enqueueTask(kernel, &write_events[0], &kernel_events[0][0]);
    std::vector<cl::Memory> buffread;
    buffread.push_back(buf_weight);
    q.enqueueMigrateMemObjects(buffread, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[0], &read_events[0][0]);
    q.finish();

    gettimeofday(&tv_r_e, 0);

    cl_ulong start, end;
    kernel_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    kernel_events[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    long long kernel_ns = end - start;
    std::cout << "Kernel execution timn: " << kernel_ns / 1000000 << "." << (kernel_ns) % 1000000 / 10000 << "ms"
              << std::endl;
    std::cout << std::dec << "Decision Tree FPGA times:" << tvdiff(&tv_r_s, &tv_r_e) / 1000 << " ms" << std::endl;

    //    LinearSVM(data, weight);
    const double golden[28] = {0.187245,    -0.00602651, 0.00680521, 0.109098,   -0.00730226, 0.21383,    -0.0371468,
                               -0.00824233, 0.176195,    0.185988,   -0.0111435, 0.0269147,   0.17515,    0.219899,
                               0.0128187,   -0.0165276,  0.155597,   0.184356,   -0.019142,   -0.0355461, 0.153701,
                               0.187469,    0.205479,    0.20453,    0.155229,   0.124429,    0.164835,   0.134081};
    int ret = 0;
    for (int i = 0; i < features_num; i++) {
        f_cast<double> w;
        w.i = weight[i / 8](i % 8 * 64 + 63, i % 8 * 64);
        std::cout << i << ": " << w.f << " golden " << golden[i] << std::endl;
        if (fabs((w.f - golden[i]) / golden[i]) > 0.00001) ret++;
    }
    free(datasets);
    free(datasets_D);

    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return ret;
}
