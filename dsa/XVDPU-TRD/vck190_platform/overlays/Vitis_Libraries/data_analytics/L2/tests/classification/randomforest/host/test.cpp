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
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include "xf_data_analytics/classification/decision_tree_quantize.hpp"
#include "utils.hpp"
#include <CL/cl_ext_xilinx.h>
#include <xcl2.hpp>
#include "test.hpp"
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

typedef struct print_buf_result_data_ {
    int i;
    int nodes_num;
    ap_uint<512>* tree;
} print_buf_result_data_t;
void CL_CALLBACK print_buf_result(cl_event event, cl_int cmd_exec_status, void* user_data) {
    print_buf_result_data_t* d = (print_buf_result_data_t*)user_data;
    ap_uint<512>* tree = d->tree;
    struct Node_H<DataType> nodes_0[MAX_NODES_NUM];
    int nodes_num_0 = 1;
    for (int i = 0; i < MAX_NODES_NUM; i++) {
        nodes_0[i].chl = INVALID_NODEID;
        nodes_0[i].isLeaf = 0;
    }
    GetTreeFromBits<DataType, 64>(nodes_0, tree, nodes_num_0);
    d->nodes_num = nodes_num_0;
    std::cout << "nodes_num_0:" << nodes_num_0 << std::endl;
    // printTree(nodes_0, nodes_num_0);
    // test
    /*bool unorderedFeatures[9] = {0};
    precisonAndRecall(testsets, test_num, features_num, nodes_0, unorderedFeatures, numClass);
*/
}
const int dw = sizeof(DataType) * 8;
const ap_uint<32> data_out_header_len = 1024;
int main(int argc, const char* argv[]) {
    std::cout << "\n--------- RF Sampling Test ---------\n";
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // cmd arg parser.
    ArgParser parser(argc, argv);
    std::string xclbin_path; // eg. q5kernel_VCU1525_hw.xclbin
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }

    std::string in_dir;
    if (!parser.getCmdOption("-in", in_dir) || !is_dir(in_dir)) {
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
    std::string cn;
    if (!parser.getCmdOption("-cn", cn)) {
        std::cout << "ERROR: cn is not specified or not valid.\n";
        return 1;
        // test_num = std::stoi(samples_num);
    }
    std::string loopnum_str;
    if (!parser.getCmdOption("-ln", loopnum_str)) {
        loopnum_str = "1";
    }
    int num_rep = std::stoi(loopnum_str);
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

    cl::Kernel kernel_sp_0[2];

    cl::Kernel kernel_tree_0[2];

    for (int i = 0; i < 2; i++) {
        kernel_tree_0[i] = cl::Kernel(program, "DecisionTreeQT_0", &cl_err);
        logger.logCreateKernel(cl_err);

        kernel_sp_0[i] = cl::Kernel(program, "randomForestSP", &cl_err);
        logger.logCreateKernel(cl_err);
    }

    struct Paras paras;
    paras.max_tree_depth = 9;
    paras.min_leaf_size = 1;
    paras.max_leaf_cat_per = 0.998;
    paras.min_info_gain = 0;
    paras.maxBins = 4;
    paras.cretiea = 0;
    ap_uint<32> samples_num = std::stoi(trn);
    ap_uint<32> test_num = std::stoi(ten);
    ap_uint<32> features_num = std::stoi(fn);
    ap_uint<32> numClass = std::stoi(cn);

    int elem_per_line = 64 / sizeof(DataType);
    int elem_per_line_qt = 64 / sizeof(ap_uint<8>);

    int total = samples_num * (features_num + 1);
    int datasize = (total + elem_per_line - 1) / elem_per_line + 1;
    int datasize_qt = (total + elem_per_line_qt - 1) / elem_per_line_qt + data_out_header_len;

    printf("data_size:%d\n", datasize);

    ap_uint<512>* data;
    data = aligned_alloc<ap_uint<512> >(datasize);
    ap_uint<512>* configs;
    configs = aligned_alloc<ap_uint<512> >(32);
    ap_uint<512>* data_out_0;
    data_out_0 = aligned_alloc<ap_uint<512> >(datasize_qt);
    ap_uint<512>* data_out_1;
    data_out_1 = aligned_alloc<ap_uint<512> >(datasize_qt);
    ap_uint<512>* trees[2];
    trees[0] = aligned_alloc<ap_uint<512> >(treesize);
    trees[1] = aligned_alloc<ap_uint<512> >(treesize);

    cl_mem_ext_ptr_t mext_data = {1, data, kernel_sp_0[0]()};
    cl_mem_ext_ptr_t mext_configs = {2, configs, kernel_sp_0[0]()};
    cl_mem_ext_ptr_t mext_data_out_0 = {3, data_out_0, kernel_sp_0[0]()};
    cl_mem_ext_ptr_t mext_data_out_1 = {3, data_out_1, kernel_sp_0[1]()};
    cl_mem_ext_ptr_t mext_tree_0 = {1, trees[0], kernel_tree_0[0]()};
    cl_mem_ext_ptr_t mext_tree_1 = {1, trees[1], kernel_tree_0[1]()};

    load_dat<ap_uint<512> >(data + 1, "train", in_dir, datasize - 1, sizeof(ap_uint<512>));

    // for test reasult
    DataType* testsets = (DataType*)malloc(sizeof(DataType) * test_num * (features_num + 1));
    std::string line;
    int row = 0;
    int col = 0;
    // test csv read
    std::ifstream fin_test(in_dir + "/test.txt");
    row = 0;
    col = 0;
    // getline(fin, line);;if there is header
    while (getline(fin_test, line)) {
        std::istringstream sin(line);
        std::string attr_val;
        col = 0;
        while (getline(sin, attr_val, ',')) {
            testsets[(features_num + 1) * row + col] = std::atof(attr_val.c_str());
            col++;
        }
        row++;
    }
    std::cout << "testing data rows num:" << row << std::endl;
    std::cout << "testing data cols num:" << col << std::endl;

    f_cast<float> instance_fraction_;
    f_cast<float> feature_fraction_0_;
    f_cast<float> feature_fraction_1_;
    instance_fraction_.f = 1;
    feature_fraction_1_.f = 1;

    data[0].range(31, 0) = samples_num;              //
    data[0].range(63, 32) = features_num;            //
    data[0].range(95, 64) = numClass;                //
    data[0].range(127, 96) = instance_fraction_.i;   // instance_fraction
    data[0].range(159, 128) = feature_fraction_0_.i; // feature_fraction
    data[0].range(191, 160) = feature_fraction_1_.i; // feature_fraction
    data[0].range(223, 192) = 43;                    // instance_seed
    data[0].range(255, 224) = 90;                    // instance_seed

    printf("\n");
    printf("\n");
    printf("\n");

    GenConfAll<double, 64>(samples_num, features_num, numClass, paras, in_dir + "/config.txt", configs);

    int err;
    cl::Buffer buf_data_0_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                            (size_t)(sizeof(ap_uint<512>) * datasize), &mext_data, &err);
    cl::Buffer buf_data_0_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                            (size_t)(sizeof(ap_uint<512>) * datasize), &mext_data, &err);
    printf("creating buf_data\n");

    cl::Buffer buf_configs_0_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                               (size_t)(sizeof(ap_uint<512>) * 32), &mext_configs, &err);
    cl::Buffer buf_configs_0_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                               (size_t)(sizeof(ap_uint<512>) * 32), &mext_configs, &err);
    printf("creating buf_configs\n");

    cl::Buffer buf_data_out_0_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                (size_t)(sizeof(ap_uint<512>) * (datasize_qt)), &mext_data_out_0, &err);
    cl::Buffer buf_data_out_0_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                (size_t)(sizeof(ap_uint<512>) * (datasize_qt)), &mext_data_out_1, &err);

    std::vector<cl::Memory> tb;
    tb.push_back(buf_data_out_0_a);
    tb.push_back(buf_data_out_0_b);
    q.enqueueMigrateMemObjects(tb, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);

    printf("creating buf_data_out\n");

    cl::Buffer buf_trees[2];
    buf_trees[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                              (size_t)(sizeof(ap_uint<512>) * treesize), &mext_tree_0, &err);
    buf_trees[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                              (size_t)(sizeof(ap_uint<512>) * treesize), &mext_tree_1, &err);
    printf("creating buf_tree\n");

    q.finish();
    std::cout << "DDR buffers have been mapped/copy-and-mapped\n";

    std::vector<std::vector<cl::Event> > write_events_0(num_rep);
    std::vector<std::vector<cl::Event> > read_events_0(num_rep);
    std::vector<std::vector<cl::Event> > kernel_tree_events_0(num_rep);
    std::vector<std::vector<cl::Event> > kernel_sp_events_0(num_rep);

    for (int i = 0; i < num_rep; ++i) {
        write_events_0[i].resize(1);
        kernel_tree_events_0[i].resize(1);
        kernel_sp_events_0[i].resize(1);
        read_events_0[i].resize(1);
    }
    std::vector<cl::Memory> ibtable_0[2];

    ibtable_0[0].push_back(buf_data_0_a);
    ibtable_0[0].push_back(buf_configs_0_a);
    ibtable_0[0].push_back(buf_trees[0]);

    ibtable_0[1].push_back(buf_data_0_b);
    ibtable_0[1].push_back(buf_configs_0_b);
    ibtable_0[1].push_back(buf_trees[1]);

    const int seedid_0 = 0;
    const int seedid_1 = 1;
    int j = 0;
    kernel_sp_0[0].setArg(j++, seedid_0);
    kernel_sp_0[0].setArg(j++, buf_data_0_a);
    kernel_sp_0[0].setArg(j++, buf_configs_0_a);
    kernel_sp_0[0].setArg(j++, buf_data_out_0_a);

    j = 0;
    kernel_sp_0[1].setArg(j++, seedid_1);
    kernel_sp_0[1].setArg(j++, buf_data_0_b);
    kernel_sp_0[1].setArg(j++, buf_configs_0_b);
    kernel_sp_0[1].setArg(j++, buf_data_out_0_b);

    j = 0;
    kernel_tree_0[0].setArg(j++, buf_data_out_0_a);
    kernel_tree_0[0].setArg(j++, buf_trees[0]);

    j = 0;
    kernel_tree_0[1].setArg(j++, buf_data_out_0_b);
    kernel_tree_0[1].setArg(j++, buf_trees[1]);
    std::cout << "Kernel has been setup\n";

    q.enqueueMigrateMemObjects(ibtable_0[0], 0, nullptr, &write_events_0[0][0]);
    q.enqueueMigrateMemObjects(ibtable_0[1], 0, nullptr, &write_events_0[1][0]);

    std::cout << "First Migration for data\n";

    std::vector<print_buf_result_data_t> cbd(num_rep);
    std::vector<print_buf_result_data_t>::iterator it = cbd.begin();
    print_buf_result_data_t* cbd_ptr = &(*it);

    for (int i = 0; i < num_rep; ++i) {
        int k_id = i & 0x01;
        if (i > 1) {
            q.enqueueTask(kernel_sp_0[k_id], &read_events_0[i - 2], &kernel_sp_events_0[i][0]);
        } else {
            q.enqueueTask(kernel_sp_0[k_id], &write_events_0[i], &kernel_sp_events_0[i][0]);
        }
        q.enqueueTask(kernel_tree_0[k_id], &kernel_sp_events_0[i], &kernel_tree_events_0[i][0]);

        // read data from DDR
        std::vector<cl::Memory> obtable_0;
        obtable_0.push_back(buf_trees[k_id]);
        q.enqueueMigrateMemObjects(obtable_0, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_tree_events_0[i],
                                   &read_events_0[i][0]);
        cbd_ptr[i].i = i;
        cbd_ptr[i].tree = trees[k_id];
        cbd_ptr[i].nodes_num = 0;
        read_events_0[i][0].setCallback(CL_COMPLETE, print_buf_result, cbd_ptr + i);
    }
    q.finish();

    // check the result
    int nerr = 0;
    for (int i = 0; i < num_rep; i++) {
        print_buf_result_data_t* d = (print_buf_result_data_t*)(cbd_ptr + i);
        int nodes_num = d->nodes_num;
        // golden nodes_num : 227
        if (nodes_num != 227) nerr++;
    }
    nerr ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return nerr;
}
