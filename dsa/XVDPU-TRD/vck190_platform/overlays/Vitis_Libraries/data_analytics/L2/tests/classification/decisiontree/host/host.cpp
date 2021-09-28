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
#include "xf_data_analytics/classification/decision_tree_L2.hpp"
#include "utils.hpp"
#include <CL/cl_ext_xilinx.h>
#include <xcl2.hpp>
#include "host.hpp"
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
    std::cout << "\n--------- Decision Tree Test ---------\n";

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
    std::string bn_s;
    int bn;
    if (!parser.getCmdOption("-bn", bn_s)) {
        bn_s = "1";
    }
    bn = std::stoi(bn_s);

    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // Get CL devices.
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Create context and command queue for selected device
    cl_int cl_err;
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
    kernel = cl::Kernel(program, "DecisionTree", &cl_err);
    logger.logCreateKernel(cl_err);

    // Allocate Memory in Host Memory
    struct Paras paras;
    paras.max_tree_depth = 9;
    paras.min_leaf_size = 1;
    paras.max_leaf_cat_per = 0.995;
    paras.min_info_gain = 0;
    paras.maxBins = 4;
    paras.cretiea = 0;
    struct Node_H<DataType> nodes[MAX_NODES_NUM];
    int nodes_num = 1;
    for (int i = 0; i < MAX_NODES_NUM; i++) {
        nodes[i].chl = INVALID_NODEID;
        nodes[i].isLeaf = 0;
    }
    ap_uint<30> samples_num_ = std::stoi(trn);
    ap_uint<30> samples_num = samples_num_ * bn;
    ap_uint<30> test_num = std::stoi(ten) * bn;
    ap_uint<8> features_num = std::stoi(fn);
    ap_uint<8> numClass = std::stoi(cn);
    ap_uint<8> numCategories[MAX_CAT_NUM_] = {0};

    DataType* datasets = (DataType*)malloc(sizeof(DataType) * samples_num * (features_num + 1));
    DataType* testsets = (DataType*)malloc(sizeof(DataType) * test_num * (features_num + 1));
    // read train data
    std::ifstream fin(in_dir + "/train.txt");
    std::string line;
    int row = 0;
    int col = 0;
    // getline(fin, line);if there is header
    while (getline(fin, line)) {
        std::istringstream sin(line);
        std::string attr_val;
        col = 0;
        while (getline(sin, attr_val, ',')) {
            datasets[(features_num + 1) * row + col] = std::atof(attr_val.c_str());
            col++;
        }
        row++;
    }
    // testing
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << datasets[i * (features_num + 1) + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "traing data rows num:" << row << std::endl;
    std::cout << "traing data cols num:" << col << std::endl;
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

    int elem_per_line = 64 / sizeof(DataType);
    int total = samples_num * (features_num + 1);
    int datasize = (total + elem_per_line - 1) / elem_per_line + 1;
    printf("data_size:%d\n", datasize);

    // ap_uint<512> data[DATASIZE];
    ap_uint<512>* data;
    data = aligned_alloc<ap_uint<512> >(datasize);
    ap_uint<512>* configs;
    configs = aligned_alloc<ap_uint<512> >(30);
    ap_uint<512>* tree;
    tree = aligned_alloc<ap_uint<512> >(treesize);
#ifdef USE_DDR
    cl_mem_ext_ptr_t mext_data = {XCL_BANK0, data, 0};
    cl_mem_ext_ptr_t mext_configs = {XCL_BANK0, configs, 0};
    cl_mem_ext_ptr_t mext_tree = {XCL_BANK0, tree, 0};
#else
    cl_mem_ext_ptr_t mext_data = {(unsigned int)(0), data, 0};
    cl_mem_ext_ptr_t mext_configs = {(unsigned int)(0), configs, 0};
    cl_mem_ext_ptr_t mext_tree = {(unsigned int)(0), tree, 0};
#endif
    // preprocess, now in host
    DataPreocess<DataType, 64>(datasets, numCategories, paras, in_dir, samples_num, features_num, numClass, configs,
                               data);
    // Map buffers
    int err;
    cl::Buffer buf_data(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,

                        (size_t)(sizeof(ap_uint<512>) * datasize), &mext_data, &err);
    printf("creating buf_data\n");

    cl::Buffer buf_configs(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                           (size_t)(sizeof(ap_uint<512>) * 30), &mext_configs, &err);
    printf("creating buf_table_l_a\n");

    cl::Buffer buf_tree(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                        (size_t)(sizeof(ap_uint<512>) * treesize), &mext_tree, &err);
    printf("creating buf_tree\n");
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
    buffwrite.push_back(buf_configs);
    buffwrite.push_back(buf_tree);

    int j = 0;
    kernel.setArg(j++, buf_data);
    kernel.setArg(j++, buf_configs);
    kernel.setArg(j++, buf_tree);

    q.enqueueMigrateMemObjects(buffwrite, 0, nullptr, &write_events[0][0]);
    q.enqueueTask(kernel, &write_events[0], &kernel_events[0][0]);
    std::vector<cl::Memory> buffread;
    buffread.push_back(buf_tree);
    q.enqueueMigrateMemObjects(buffread, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[0], &read_events[0][0]);
    // DecisionTree(data, configs, tree);
    // get the tree
    q.finish();
    GetTreeFromBits<DataType, 64>(nodes, tree, nodes_num);
    // check nodes_num
    if (nodes_num != 227) {
        std::cout << "check failed" << std::endl;
        return 1;
    }
    printTree(nodes, nodes_num);
    // test
    bool unorderedFeatures[9] = {0};
    // check tree by computing precision and recall
    int check = precisonAndRecall(testsets, test_num, features_num, nodes, unorderedFeatures, numClass);

    check ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
          : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return check;
}
