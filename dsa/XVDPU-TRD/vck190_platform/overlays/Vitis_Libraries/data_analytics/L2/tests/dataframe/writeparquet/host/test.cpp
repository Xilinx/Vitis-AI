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
#include <hls_stream.h>
#include "xf_data_analytics/common/obj_interface.hpp"
#include "xf_data_analytics/dataframe/df_utils.hpp"
#include "xf_data_analytics/dataframe/write_to_dataframe.hpp"
#include "x_utils.hpp"
#include "xclhost.hpp"
#include "meta.hpp"
#include "xf_utils_sw/logger.hpp"
#define PAGE_SZ (8 << 20)

using namespace xf::data_analytics::dataframe;

int main(int argc, const char* argv[]) {
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "------------------ in main ------------------" << std::endl;
    // Get CL devices.
    // cmd arg parser.
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    x_utils::ArgParser parser(argc, argv);
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "No input xclbin" << std::endl;
        exit(1);
    };
    std::string exp;
    int exp_v = 15;
    int num = (1 << 26);
    if (!parser.getCmdOption("-e", exp)) {
        std::cout << "default input lines: " << (1 << exp_v) << std::endl;
    } else {
        exp_v = std::stoi(exp);
    }
    int lines = (1 << exp_v);
    std::cout << "input lines: " << lines << std::endl;
    cl_int err;
    cl_context ctx;
    cl_device_id dev_id;
    cl_command_queue cq;
    cl_program prg;

    err = xclhost::init_hardware(&ctx, &dev_id, &cq, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                 MSTR(XDEVICE));
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to init OpenCL with " MSTR(XDEVICE) "\n");
        return err;
    }

    err = xclhost::load_binary(&prg, ctx, dev_id, xclbin_path.c_str());
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to program PL\n");
        return err;
    }

    std::cout << "--- generate data in object stream struct ---" << std::endl;

    hls::stream<Object> obj_strm("in_obj_strm");
    int data_size[5];
    char* datas[5];
    for (int i = 0; i < 5; i++) {
        data_size[i] = 0;
        datas[i] = (char*)malloc(sizeof(char) * (1 << 30));
    }
    Object obj_data;

    // 3 rounds to finish 1 file
    int round = 1;
    int str_page_num = 0;
    int str_page_size[1024];
    int str_page_count[1024];

    for (int i = 0; i < 1024; i++) {
        str_page_size[i] = 0;
        str_page_count[i] = 0;
    }
    for (int r = 0; r < round; r++) {
        for (int l = 0; l < lines; l++) {
            // each json line has 5 data
            for (int n = 0; n < 5; n++) {
                if (n != 4) {
                    ap_uint<64> dat = l + n;  // 64bit double data
                    ap_uint<16> field_id = n; // 4 fields
                    ap_uint<4> valid = 8;     // 64-bit valid

                    ap_uint<4> type = TInt64; // int64,1
                    if (n == 2) {
                        type = TDouble; // double, 3
                    }
                    if (n == 3) {
                        type = TDouble; // 2
                        // type = TFloat32; // 2
                        // valid = 4;
                    }

                    memcpy(datas[n] + data_size[n], &dat, valid);
                    data_size[n] += valid;

                    obj_data.set_data(dat);
                    obj_data.set_id(field_id);
                    obj_data.set_valid(valid);
                    obj_data.set_type(type);
                    obj_strm.write(obj_data);
                } else if (n == 4) {
                    ap_uint<4> type = TString;
                    ap_uint<16> field_id = n; // 4 fields
                    int str_len = 4;          //+ l % 20;
                    if (str_page_size[str_page_num] + 4 + str_len > (PAGE_SZ)) {
                        data_size[n] += (str_page_size[str_page_num]);
                        str_page_num++;
                    }
                    memcpy(datas[n] + data_size[n] + str_page_size[str_page_num], &str_len, 4);
                    str_page_size[str_page_num] += 4;
                    str_page_count[str_page_num] += 1;

                    int batch_num = (str_len + sizeof(int64_t) - 1) / sizeof(int64_t);
                    for (int i = 0; i < batch_num; i++) {
                        std::string str = TestData::GetString(l);
                        ap_uint<64> dat = 0;
                        memcpy(&dat, str.c_str(), str.length());
                        ap_uint<4> valid = 8;
                        if (i == batch_num - 1) {
                            valid = str_len - i * 8;
                        }

                        if (valid < 8) dat.range(63, valid * 8) = 0;
                        memcpy(datas[n] + data_size[n] + str_page_size[str_page_num], &dat, valid);
                        str_page_size[str_page_num] += valid;

                        obj_data.set_data(dat);
                        obj_data.set_id(field_id);
                        obj_data.set_valid(valid);
                        obj_data.set_type(type);
                        obj_strm.write(obj_data);
                    }
                }
            }

            ap_uint<4> type = 13; // end of json line
            obj_data.set_id(4);
            obj_data.set_type(type);
            obj_strm.write(obj_data);

            ap_uint<4> tf = obj_data.get_type();
            std::string tf_str = (tf == FEOF) ? "EOF" : (tf == FEOC ? "EOC" : (tf == FEOL) ? "EOL" : tf.to_string());
        }

        ap_uint<4> type = 14; // end of col
        obj_data.set_type(type);
        obj_strm.write(obj_data);

        ap_uint<4> tf = obj_data.get_type();
        std::string tf_str = (tf == FEOF) ? "EOF" : (tf == FEOC ? "EOC" : (tf == FEOL) ? "EOL" : tf.to_string());
    }
    ap_uint<4> type = 15; // end of file
    obj_data.set_type(type);
    obj_strm.write(obj_data);
    str_page_num++;

    std::cout << "----------------------------- Finish stream data generation -----------------------------"
              << std::endl;
    ap_uint<4> tf = obj_data.get_type();
    std::string tf_str = (tf == FEOF) ? "EOF" : (tf == FEOC ? "EOC" : (tf == FEOL) ? "EOL" : tf.to_string());
    ap_uint<88>* ddr_obj = (ap_uint<88>*)malloc(sizeof(ap_uint<88>) * (num));
    {
        Object obj_data;
        obj_data = obj_strm.read();
        ap_uint<88> dat = obj_data.get_all();
        ap_uint<4> type = obj_data.get_type();
        int id = 1;
        while (type != FEOF) {
            ddr_obj[id++] = dat;
            obj_data = obj_strm.read();
            dat = obj_data.get_all();
            type = obj_data.get_type();
        }
        ddr_obj[id] = dat;
        ddr_obj[0] = 0;
        ddr_obj[0].range(31, 0) = id;
#ifdef _DF_DEBUG_V2
#ifndef __SYNTHESIS__
        std::cout << "final stream elem: " << id << std::endl;
#endif
#endif
    }
    int size = (num) * sizeof(ap_uint<64>);
    int size_ = sizeof(ap_uint<88>) * (num);
    std::cout << "----------------------------- Finish dataframe generation -----------------------------" << std::endl;

    ap_uint<64>* buff0;
    buff0 = (ap_uint<64>*)malloc((num) * sizeof(ap_uint<64>));
    memset(buff0, 0, sizeof(ap_uint<64>) * (num));
    ap_uint<8> schema_1[16];
    schema_1[0] = 1;
    schema_1[1] = 1;
    schema_1[2] = 3;
    schema_1[3] = 3;
    schema_1[4] = 5;
    //    ObjToParquet(ddr_obj, schema_1, buff0);
    cl_kernel kernel_write = clCreateKernel(prg, "ObjToParquet", &err);
    logger.logCreateKernel(err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: failed to create kernel.\n");
        return 1;
    }
    cl_mem_ext_ptr_t mext_out, mext_schema_1, mext_tmp;
    mext_tmp = {0, ddr_obj, kernel_write}; //{0,tmp,kernel_write}
    mext_schema_1 = {1, schema_1, kernel_write};
    mext_out = {2, buff0, kernel_write};

    cl_mem buf_out, buf_schema_1, buf_tmp;
    buf_schema_1 = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, //
                                  16, &mext_schema_1, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: failed to create memory.\n");
        return 1;
    }

    buf_tmp = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, //
                             size_, &mext_tmp, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: failed to create memory.\n");
        return 1;
    }

    buf_out = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, //
                             size, &mext_out, &err);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: failed to create memory.\n");
        return 1;
    }
    std::cout << "Successfully create memories" << std::endl;

    int j = 0;
    clSetKernelArg(kernel_write, j++, sizeof(cl_mem), &buf_tmp);
    clSetKernelArg(kernel_write, j++, sizeof(cl_mem), &buf_schema_1);
    clSetKernelArg(kernel_write, j++, sizeof(cl_mem), &buf_out);

    std::vector<cl_mem> bufs;
    bufs.push_back(buf_schema_1);
    bufs.push_back(buf_tmp);
    bufs.push_back(buf_out);
    clEnqueueMigrateMemObjects(cq, bufs.size(), bufs.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr,

                               nullptr);
    clFinish(cq);
    std::vector<cl_mem> bufs_in;
    bufs_in.push_back(buf_schema_1);
    bufs_in.push_back(buf_tmp);

    std::vector<cl_mem> bufs_out;
    bufs_out.push_back(buf_out);
    std::cout << "Finish Initialization" << std::endl;

    std::array<cl_event, 1> evt_r;
    std::array<cl_event, 1> evt_kw;
    std::array<cl_event, 1> evt_w;
    clEnqueueMigrateMemObjects(cq, bufs_in.size(), bufs_in.data(), 0, 0, nullptr, &evt_r[0]);

    // clFinish(cq);
    // std::cout << "finish h2d" << std::endl;
    clEnqueueTask(cq, kernel_write, 1, evt_r.data(), &evt_kw[0]);
    // clFinish(cq);
    // std::cout << "finish kernel" << std::endl;

    clEnqueueMigrateMemObjects(cq, bufs_out.size(), bufs_out.data(), CL_MIGRATE_MEM_OBJECT_HOST, 1, evt_kw.data(),
                               &evt_w[0]);
    clFinish(cq);
    // std::cout << "finish d2h" << std::endl;

    cl_ulong start1, end1;
    clGetEventProfilingInfo(evt_kw[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start1, NULL);
    clGetEventProfilingInfo(evt_kw[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end1, NULL);
    double kerneltime1 = (double)(end1 - start1) / 1000000;
    std::cout << std::dec << "Kernel execution time " << kerneltime1 << " ms" << std::endl;

    std::cout << "----------------------------- Collect pages -----------------------------" << std::endl;
    WriteParquetFile<4>(buff0, 5);
    int nerr = ReadParquetFile();
    nerr ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);

    return nerr;
}
