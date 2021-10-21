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

#include "xclhost.hpp"
#include "db_intpair_sort_1g.hpp"

namespace xf {
namespace database {
namespace intpair_sort {
void sortAPI::init(std::string xclbin_path, int device_id, bool user_setting) {
    evs_write.resize(2);
    evs_insert.resize(2);
    evs_m0.resize(2);
    evs_m1.resize(2);
    evs_m2.resize(2);

    // device init
    std::string dsa_name;
    err = xclhost::init_hardware(&ctx, &dev_id, &cq, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                 dsa_name, device_id, user_setting);
    if (xclbin_path == "") {
        xclbin_path = "/opt/xilinx/apps/vt_database/sort/share/xclbin/vt_database_sort-" + dsa_name + ".xclbin";
        std::cout << "Auto select XCLBIN: " << xclbin_path << std::endl;
    }
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to init OpenCL \n");
        exit(1);
    }

    std::cout << "platform inited" << std::endl;
    err = xclhost::load_binary(&prg, ctx, dev_id, xclbin_path.c_str());
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to program PL\n");
        exit(1);
    }

    cl_mem_ext_ptr_t mext_in[2], mext_k0[2], mext_k1[2], mext_k2[2], mext_k3, mext_k4;
    for (int i = 0; i < 2; i++) {
        mext_in[i] = {XCL_MEM_TOPOLOGY | unsigned(3), nullptr, 0};
        mext_k0[i] = {XCL_MEM_TOPOLOGY | unsigned(3), nullptr, 0};
        mext_k1[i] = {XCL_MEM_TOPOLOGY | unsigned(3), nullptr, 0};
        mext_k2[i] = {XCL_MEM_TOPOLOGY | unsigned(3), nullptr, 0};
    }
    mext_k3 = {XCL_MEM_TOPOLOGY | unsigned(3), nullptr, 0};
    mext_k4 = {XCL_MEM_TOPOLOGY | unsigned(2), nullptr, 0};

    // Map buffers
    ap_uint<64>* init_buf = mm.aligned_alloc<ap_uint<64> >(sz_test);
    // memset(init_buf, 0, sz_test * sizeof(ap_uint<64>));
    for (int i = 0; i < 2; i++) {
        buf_in[i] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sz_64m * sizeof(ap_uint<64>),
                                   &mext_in[i], &err);
        if (err != CL_SUCCESS) {
            std::cout << "Create Device buf_in " << i << " Failed!" << std::endl;
            exit(1);
        }
        buf_k0[i] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, sz_64m * sizeof(ap_uint<64>),
                                   &mext_k0[i], &err);
        if (err != CL_SUCCESS) {
            std::cout << "Create Device buf_k0 " << i << " Failed!" << std::endl;
            exit(1);
        }
        buf_k1[i] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, sz_64m * sizeof(ap_uint<64>),
                                   &mext_k1[i], &err);
        if (err != CL_SUCCESS) {
            std::cout << "Create Device buf_k1 " << i << " Failed!" << std::endl;
            exit(1);
        }
        buf_k2[i] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, sz_64m * sizeof(ap_uint<64>),
                                   &mext_k2[i], &err);
        if (err != CL_SUCCESS) {
            std::cout << "Create Device buf_k2 " << i << " Failed!" << std::endl;
            exit(1);
        }
        err = clEnqueueWriteBuffer(cq, buf_in[i], 0, 0, sizeof(ap_uint<64>) * sz_64m, init_buf, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(cq, buf_k0[i], 0, 0, sizeof(ap_uint<64>) * sz_64m, init_buf, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(cq, buf_k1[i], 0, 0, sizeof(ap_uint<64>) * sz_64m, init_buf, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(cq, buf_k2[i], 0, 0, sizeof(ap_uint<64>) * sz_64m, init_buf, 0, NULL, NULL);
        clFinish(cq);
    }
    buf_k3 =
        clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, sz_test * sizeof(ap_uint<64>), &mext_k3, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Create Device buf_k3 Failed!" << std::endl;
        exit(1);
    }

    buf_k4 =
        clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, sz_test * sizeof(ap_uint<64>), &mext_k4, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Create Device buf_k4 Failed!" << std::endl;
        exit(1);
    }

    err = clEnqueueWriteBuffer(cq, buf_k3, 0, 0, sizeof(ap_uint<64>) * sz_test, init_buf, 0, NULL, NULL);
    clFinish(cq);

    err = clEnqueueWriteBuffer(cq, buf_k4, 0, 0, sizeof(ap_uint<64>) * sz_test, init_buf, 0, NULL, NULL);
    clFinish(cq);
    // init for
    sort_64m_once(init_buf, init_buf, sz_64m, 1, 0);
    sort_64m_once(init_buf, init_buf, sz_64m, 1, 1);
    std::cout << "start Initial done" << std::endl;
}

std::vector<int> sortAPI::get_merge_loop_num(size_t allsize) {
    int round = allsize / sz_64m;
    int leftsize = allsize % sz_64m;

    int m_lp = -1;
    int size_counter = sz_128k;
    if (leftsize > 0) m_lp++;
    while (size_counter < leftsize) {
        m_lp++;
        size_counter *= merge_num[m_lp];
    }
    if (m_lp > 3) {
        std::cerr << "l_level max " << std::endl;
        exit(1);
    }
    // if m_lp == 0, then need one insert
    // if m_lp == 1, then need one insert and one merge
    // if m_lp == 2, then need one insert and two merge
    // if m_lp == 3, then need one insert and three merge
    //
    std::vector<int> user_sche(2);
    user_sche[0] = round;
    user_sche[1] = m_lp;
#ifdef API_DEBUG
    std::cout << "round:" << round << ",l_level:" << m_lp << std::endl;
#endif
    return user_sche;
}

// Sort function
std::future<ErrCode> sortAPI::sort(uint64_t* user_in, uint64_t* user_out, size_t allsize, int order) {
    // timeval t_0, t_1;
    // promise and future
    std::promise<ErrCode> prom;
    std::future<ErrCode> user_future = prom.get_future();
    // create kernels
    std::vector<int> merge_loop_num = get_merge_loop_num(allsize);
    int round = merge_loop_num[0];
    int round_ = (allsize + sz_64m - 1) / sz_64m;
    int l_level = merge_loop_num[1];

    std::vector<cl_kernel> insert_kernel(round + 1);
    std::vector<cl_kernel> merge_kernels_0(round + 1);
    std::vector<cl_kernel> merge_kernels_1(round + 1);
    std::vector<cl_kernel> merge_kernels_2(round + 1);
    cl_kernel merge_kernels_3;
    for (int i = 0; i < round; i++) {
#ifdef API_DEBUG
        std::cout << "WARNING: in round kernel create" << std::endl;
#endif
        int ppid = (i + counter) % 0x02;
        insert_kernel[i] = clCreateKernel(prg, "gqe_insert_kernel_0", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
            exit(1);
        }
        int j = 0;
        int batchsize = sz_64m;
        int offset = 0;
        clSetKernelArg(insert_kernel[i], j++, sizeof(cl_mem), &buf_in[ppid]);
        clSetKernelArg(insert_kernel[i], j++, sizeof(unsigned int), &batchsize);
        clSetKernelArg(insert_kernel[i], j++, sizeof(unsigned int), &order);
        clSetKernelArg(insert_kernel[i], j++, sizeof(unsigned int), &offset);
        clSetKernelArg(insert_kernel[i], j++, sizeof(cl_mem), &buf_k0[ppid]);
#ifdef API_DEBUG
        std::cout << "WARNING: insert_0 kernel " << i << " created" << std::endl;
#endif

        merge_kernels_0[i] = clCreateKernel(prg, "gqe_merge_kernel_0", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
        }
        j = 0;
        offset = 0;
        int each_len = sz_128k;
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(cl_mem), &buf_k0[ppid]);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(unsigned int), &each_len);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(unsigned int), &batchsize);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(unsigned int), &order);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(unsigned int), &offset);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(cl_mem), &buf_k1[ppid]);
#ifdef API_DEBUG
        std::cout << "WARNING: merge_0 kernel " << i << " created" << std::endl;
#endif

        merge_kernels_1[i] = clCreateKernel(prg, "gqe_merge_kernel_1", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
            exit(1);
        }
        j = 0;
        offset = 0;
        each_len = sz_1m;
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(cl_mem), &buf_k1[ppid]);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(unsigned int), &each_len);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(unsigned int), &batchsize);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(unsigned int), &order);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(unsigned int), &offset);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(cl_mem), &buf_k2[ppid]);
#ifdef API_DEBUG
        std::cout << "WARNING: merge_1 kernel " << i << " created" << std::endl;
#endif

        merge_kernels_2[i] = clCreateKernel(prg, "gqe_merge_kernel_2", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
            exit(1);
        }
        j = 0;
        offset = i * sz_64m / 2;
        each_len = sz_8m;
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(cl_mem), &buf_k2[ppid]);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(unsigned int), &each_len);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(unsigned int), &batchsize);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(unsigned int), &order);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(unsigned int), &offset);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(cl_mem), &buf_k3);
#ifdef API_DEBUG
        std::cout << "WARNING: merge_2 kernel " << i << " created" << std::endl;
#endif
    }

    merge_kernels_3 = clCreateKernel(prg, "gqe_merge_kernel_3", &err);
    if (allsize > sz_64m) {
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
            exit(1);
        }
        int j = 0;
        int each_len = sz_64m;
        int offset = 0;
        clSetKernelArg(merge_kernels_3, j++, sizeof(cl_mem), &buf_k3);
        clSetKernelArg(merge_kernels_3, j++, sizeof(unsigned int), &each_len);
        clSetKernelArg(merge_kernels_3, j++, sizeof(unsigned int), &allsize);
        clSetKernelArg(merge_kernels_3, j++, sizeof(unsigned int), &order);
        clSetKernelArg(merge_kernels_3, j++, sizeof(unsigned int), &offset);
        clSetKernelArg(merge_kernels_3, j++, sizeof(cl_mem), &buf_k4);
#ifdef API_DEBUG
        std::cout << "WARNING: merge_3 kernel created" << std::endl;
#endif
    }

    for (int i = 0; i <= l_level; i++) {
        int rid = round;
        int ppid = (counter + rid) % 0x02;
        int batchsize = allsize - round * sz_64m;
        if (i == 0) {
            insert_kernel[rid] = clCreateKernel(prg, "gqe_insert_kernel_0", &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create insert kernel.\n");
                exit(1);
            }
            int j = 0;
            int offset = 0;
            clSetKernelArg(insert_kernel[rid], j++, sizeof(cl_mem), &buf_in[ppid]);
            clSetKernelArg(insert_kernel[rid], j++, sizeof(unsigned int), &batchsize);
            clSetKernelArg(insert_kernel[rid], j++, sizeof(unsigned int), &order);
            cl_mem buf_k0_ = buf_k0[ppid];
            if (rid != 0 && i == l_level) {
                buf_k0_ = buf_k3;
                offset = rid * sz_64m / 2;
            }
            clSetKernelArg(insert_kernel[rid], j++, sizeof(unsigned int), &offset);
            clSetKernelArg(insert_kernel[rid], j++, sizeof(cl_mem), &buf_k0_);
#ifdef API_DEBUG
            std::cout << "WARNING: insert kernel l_level created,rid is: " << rid << std::endl;
#endif
        } else if (i == 1) {
            merge_kernels_0[rid] = clCreateKernel(prg, "gqe_merge_kernel_0", &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create insert kernel.\n");
                exit(1);
            }
            int j = 0;
            int each_len = sz_128k;
            int offset = 0;
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(cl_mem), &buf_k0[ppid]);
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(unsigned int), &each_len);
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(unsigned int), &batchsize);
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(unsigned int), &order);
#ifdef API_DEBUG
            std::cout << "ndebug subbuffer in m0" << std::endl;
#endif
            cl_mem buf_k1_ = buf_k1[ppid];
            if (rid != 0 && i == l_level) {
                buf_k1_ = buf_k3;
                offset = rid * sz_64m / 2;
            }
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(unsigned int), &offset);
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(cl_mem), &buf_k1_);

#ifdef API_DEBUG
            std::cout << "WARNING: merge_0 kernel l_level created,rid is: " << rid << std::endl;
#endif
        } else if (i == 2) {
            merge_kernels_1[rid] = clCreateKernel(prg, "merge_kernel_1", &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create insert kernel.\n");
                exit(1);
            }
            int j = 0;
            int each_len = sz_1m;
            int offset = 0;
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(cl_mem), &buf_k1[ppid]);
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(unsigned int), &each_len);
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(unsigned int), &batchsize);
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(unsigned int), &order);
            cl_mem buf_k2_ = buf_k2[ppid];
            if (rid != 0 && i == l_level) {
                buf_k2_ = buf_k3;
                offset = rid * sz_64m / 2;
            }
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(unsigned int), &offset);
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(cl_mem), &buf_k2_);

        } else if (i == 3) {
            merge_kernels_2[rid] = clCreateKernel(prg, "merge_kernel_2", &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create insert kernel.\n");
                exit(1);
            }
            int j = 0;
            int each_len = sz_8m;
            int offset = rid * sz_64m / 2;
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(cl_mem), &buf_k2[ppid]);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(unsigned int), &each_len);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(unsigned int), &batchsize);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(unsigned int), &order);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(unsigned int), &offset);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(cl_mem), &buf_k3);
        }
    }
    std::cout << "Kernel has been created\n";

    // create events
    std::vector<cl_event> ev_w(round + 1);
    std::vector<cl_event> ev_i(round + 1);
    std::vector<cl_event> ev_m0(round + 1);
    std::vector<cl_event> ev_m1(round + 1);
    std::vector<cl_event> ev_m2(round + 1);

    cl_event ev_m3;
    cl_event ev_read;
    std::array<cl_event, sz_test_ch> for_last_merge;
    int dep_last_merge = 0;
    // read
    int loop_num = round;
    int loop_num_m0 = round;
    int loop_num_m1 = round;
    int loop_num_m2 = round;

    if (l_level >= 0) {
        loop_num += 1;
    }
    if (l_level >= 1) {
        loop_num_m0 += 1;
    }
    if (l_level >= 2) {
        loop_num_m1 += 1;
    }
    if (l_level >= 3) {
        loop_num_m2 += 1;
    }
#ifdef API_DEBUG
    std::cout << "start run kernels" << std::endl;
    std::cout << "loop_num " << loop_num << std::endl;
    std::cout << "loop_num_m0 " << loop_num_m0 << std::endl;
    std::cout << "loop_num_m1 " << loop_num_m1 << std::endl;
    std::cout << "loop_num_m2 " << loop_num_m2 << std::endl;
#endif

    for (int i = 0; i < loop_num; i++) {
        int ppid = (i + counter) % 0x02;
        int size_left = allsize - sz_64m * i;
        int size = size_left > sz_64m ? sz_64m : size_left;
#ifdef API_DEBUG
        std::cout << "run size in loop " << i << " is:" << size << std::endl;
#endif
        // clwrite
        std::array<cl_event, 2> for_w;
        int num_for_w = 0;
        if (evs_write[ppid].size() != 0) {
            for_w[num_for_w++] = evs_write[ppid][evs_write[ppid].size() - 1];
        }
        if (evs_insert[ppid].size() != 0) {
            for_w[num_for_w++] = evs_insert[ppid][evs_insert[ppid].size() - 1];
        }
        if (num_for_w == 0) {
            err = clEnqueueWriteBuffer(cq, buf_in[ppid], 0, 0, sizeof(ap_uint<64>) * (size), user_in + i * sz_64m, 0,
                                       NULL, &ev_w[i]);
        } else {
            err = clEnqueueWriteBuffer(cq, buf_in[ppid], 0, 0, sizeof(ap_uint<64>) * (size), user_in + i * sz_64m,
                                       num_for_w, for_w.data(), &ev_w[i]);
        }
        if (err != CL_SUCCESS) {
            std::cout << "EnqueueWriteBuffer FAILED" << std::endl;
            exit(1);
        } else {
#ifdef API_DEBUG
            std::cout << "EnqueueWriteBuffer SUCCESS" << std::endl;
#endif
        }
        evs_write[ppid].push_back(ev_w[i]);
        /*
        gettimeofday(&t_1, 0);
        clFinish(cq);
        x_utils::tvdiff(t_0, t_1, "Sort_future: write time");
        */

        // insert kernel
        std::array<cl_event, 2> for_insert_evs;
        for_insert_evs[0] = ev_w[i];
        int rep_num = 1;

        if (l_level == 0 && i == loop_num - 1) {
            if (loop_num != 1) {
                int m3_size = evs_m3.size();
                if (m3_size != 0) {
                    int last_m3_ind = m3_size - 1;
                    for_insert_evs[1] = evs_m3[last_m3_ind];
                    rep_num++;
                }
#ifdef API_DEBUG
                if (m3_size == 0) {
                    std::cout << "no event in evs_m3" << std::endl;
                }
#endif
            } else {
                int read_size = evs_read.size();
                if (read_size != 0) {
                    int last_read_ind = read_size - 1;
                    for_insert_evs[1] = evs_read[last_read_ind];
                    rep_num++;
                }
            }
        } else {
            int m0_size = evs_m0[ppid].size();
            if (m0_size != 0) {
                int last_m0_ind = m0_size - 1;
                for_insert_evs[1] = evs_m0[ppid][last_m0_ind];
                rep_num++;
            }
        }

        err = clEnqueueTask(cq, insert_kernel[i], rep_num, for_insert_evs.data(), &ev_i[i]);
        if (l_level == 0 && i == loop_num - 1 && loop_num != 1) {
            for_last_merge[dep_last_merge++] = ev_i[i];
        }
        if (err != CL_SUCCESS) {
            std::cout << "0 EnqueueTask insert FAILED" << std::endl;
            exit(1);
        } else {
#ifdef API_DEBUG
            std::cout << "0 EnqueueTask insert SUCCESS" << std::endl;
#endif
        }
        evs_insert[ppid].push_back(ev_i[i]);
        /*
        gettimeofday(&t_2, 0);
        clFinish(cq);
        x_utils::tvdiff(t_1, t_2, "Sort_future: insert kernel time");
        */

        // merge_0 kenrel
        if (i < loop_num_m0) {
            std::array<cl_event, 2> for_m0;
            for_m0[0] = ev_i[i];
            int rep_num = 1;

            if (l_level == 1 && i == loop_num - 1) {
                if (loop_num != 1) {
                    int m3_size = evs_m3.size();
                    if (m3_size != 0) {
                        int last_m3_ind = m3_size - 1;
                        for_m0[1] = evs_m3[last_m3_ind];
                        rep_num++;
                    }
                } else {
                    int read_size = evs_read.size();
                    if (read_size != 0) {
                        int last_read_ind = read_size - 1;
                        for_m0[1] = evs_read[last_read_ind];
                        rep_num++;
                    }
                }
            } else {
                int m1_size = evs_m1[ppid].size();
                if (m1_size != 0) {
                    int last_m1_ind = m1_size - 1;
                    for_m0[1] = evs_m1[ppid][last_m1_ind];
                    rep_num++;
                }
            }

            err = clEnqueueTask(cq, merge_kernels_0[i], rep_num, for_m0.data(), &ev_m0[i]);
            if (l_level == 1 && i == loop_num - 1 && loop_num != 1) {
                for_last_merge[dep_last_merge++] = ev_m0[i];
            }
            if (err != CL_SUCCESS) {
                std::cout << "0 EnqueueTask merge_0 FAILED" << std::endl;
                exit(1);
            } else {
#ifdef API_DEBUG
                std::cout << "0 EnqueueTask merge_0 SUCCESS" << std::endl;
#endif
            }
            evs_m0[ppid].push_back(ev_m0[i]);
        }

        // merge_1 kenrel
        if (i < loop_num_m1) {
            std::array<cl_event, 2> for_m1;
            for_m1[0] = ev_m0[i];
            int rep_num = 1;
            if (l_level == 2 && i == loop_num - 1) {
                if (loop_num != 1) {
                    int m3_size = evs_m3.size();
                    if (m3_size != 0) {
                        int last_m3_ind = m3_size - 1;
                        for_m1[1] = evs_m3[last_m3_ind];
                        rep_num++;
                    }
                } else {
                    int read_size = evs_read.size();
                    if (read_size != 0) {
                        int last_read_ind = read_size - 1;
                        for_m1[1] = evs_read[last_read_ind];
                        rep_num++;
                    }
                }
            } else {
                int m2_size = evs_m2[ppid].size();
                if (m2_size != 0) {
                    int last_m2_ind = m2_size - 1;
                    for_m1[1] = evs_m2[ppid][last_m2_ind];
                    rep_num++;
                }
            }

            err = clEnqueueTask(cq, merge_kernels_1[i], rep_num, for_m1.data(), &ev_m1[i]);
            if (l_level == 2 && i == loop_num - 1 && loop_num != 1) {
                for_last_merge[dep_last_merge++] = ev_m1[i];
            }
            if (err != CL_SUCCESS) {
                std::cout << "0 EnqueueTask merge_1 FAILED" << std::endl;
                exit(1);
            } else {
#ifdef API_DEBUG
                std::cout << "0 EnqueueTask merge_1 SUCCESS" << std::endl;
#endif
            }
            evs_m1[ppid].push_back(ev_m1[i]);
        }

        // merge_2 kenrel
        if (i < loop_num_m2) {
            std::array<cl_event, 2> for_m2;
            for_m2[0] = ev_m1[i];
            int rep_num = 1;

            if (l_level == 3 && loop_num == 1) {
                int read_size = evs_read.size();
                if (read_size != 0) {
                    int last_read_ind = read_size - 1;
                    for_m2[1] = evs_read[last_read_ind];
                    rep_num++;
                }
            } else {
                int m3_size = evs_m3.size();
                if (m3_size != 0) {
                    int last_m3_ind = m3_size - 1;
                    for_m2[1] = evs_m3[last_m3_ind];
                    rep_num++;
                }
            }

            err = clEnqueueTask(cq, merge_kernels_2[i], rep_num, for_m2.data(), &ev_m2[i]);
            for_last_merge[dep_last_merge++] = ev_m2[i];
            if (err != CL_SUCCESS) {
                std::cout << "0 EnqueueTask merge_2 FAILED" << std::endl;
                exit(1);
            } else {
#ifdef API_DEBUG
                std::cout << "0 EnqueueTask merge_2 SUCCESS" << std::endl;
#endif
            }
            evs_m2[ppid].push_back(ev_m2[i]);
        }
    }

    if (allsize > sz_64m) {
        int read_size = evs_read.size();
        if (read_size != 0) {
            int last_read_ind = read_size - 1;
            for_last_merge[dep_last_merge++] = evs_read[last_read_ind];
        }

        err = clEnqueueTask(cq, merge_kernels_3, dep_last_merge, for_last_merge.data(), &ev_m3);
        if (err != CL_SUCCESS) {
            std::cout << "0 EnqueueTask merge_3 FAILED" << std::endl;
            exit(1);
        } else {
#ifdef API_DEBUG
            std::cout << "0 EnqueueTask merge_3 SUCCESS" << std::endl;
#endif
        }
        evs_m3.push_back(ev_m3);
    }
    // read
    cl_mem buf_out;
    cl_event for_ev_read;
    if (round == 0) {
        if (l_level == 0) {
            for_ev_read = ev_i[0];
            buf_out = buf_k0[0];
        } else if (l_level == 1) {
            for_ev_read = ev_m0[0];
            buf_out = buf_k1[0];
        } else if (l_level == 2) {
            for_ev_read = ev_m1[0];
            buf_out = buf_k2[0];
        } else if (l_level == 3) {
            for_ev_read = ev_m2[0];
            buf_out = buf_k3;
        } else {
            for_ev_read = ev_m2[0];
            buf_out = buf_k3;
            std::cout << "l_level went wrong!" << std::endl;
            exit(1);
        }
#ifdef API_DEBUG
        std::cout << "buf_out is buf_k3" << std::endl;
#endif
    } else if (round == 1 && l_level == -1) {
        for_ev_read = ev_m2[0];
        buf_out = buf_k3;
#ifdef API_DEBUG
        std::cout << "1 buf_out is buf_k3_subs[0]" << std::endl;
#endif
    } else {
        for_ev_read = ev_m3;
        buf_out = buf_k4;
#ifdef API_DEBUG
        std::cout << "buf_out is buf_k4" << std::endl;
#endif
    }

#ifdef API_DEBUG
// clFinish(cq);
#endif
    clEnqueueReadBuffer(cq, buf_out, 0 /* non-blocking */, //
                        0, sizeof(ap_uint<64>) * allsize,
                        user_out, //
                        1, &for_ev_read, &ev_read);

    // callback_data cdata;
    // cdata.prom = std::move(prom);
    // clSetEventCallback(ev_read, CL_COMPLETE, callback_func, &cdata);
    evs_read.push_back(ev_read);
    // counter++;
    counter = (counter + round_) % 2;
    std::thread(&sortAPI::setProm, this, ev_read, std::move(prom)).detach();
    return user_future;
}
// std::future<ErrCode> sort(ap_uint<64>* user_in, ap_uint<64>* user_out, size_t
// allsize, int order = 1) {
std::future<ErrCode> sortAPI::sort_server(std::vector<const uint64_t*> user_in,
                                          std::vector<uint64_t*> user_out,
                                          size_t allsize,
                                          int order) {
    std::cout << "integrating in service!" << std::endl;
    // promise and future
    std::promise<ErrCode> prom;
    std::future<ErrCode> user_future = prom.get_future();
    // create kernels
    std::vector<int> merge_loop_num = get_merge_loop_num(allsize);
    int round = merge_loop_num[0];
    int round_ = (allsize + sz_64m - 1) / sz_64m;
    int l_level = merge_loop_num[1];

    std::vector<cl_kernel> insert_kernel(round + 1);
    std::vector<cl_kernel> merge_kernels_0(round + 1);
    std::vector<cl_kernel> merge_kernels_1(round + 1);
    std::vector<cl_kernel> merge_kernels_2(round + 1);
    cl_kernel merge_kernels_3;
    for (int i = 0; i < round; i++) {
#ifdef API_DEBUG
        std::cout << "WARNING: in round kernel create" << std::endl;
#endif
        int ppid = (i + counter) % 0x02;
        insert_kernel[i] = clCreateKernel(prg, "insert_kernel_0", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
            exit(1);
        }
        int j = 0;
        int batchsize = sz_64m;
        int offset = 0;
        clSetKernelArg(insert_kernel[i], j++, sizeof(cl_mem), &buf_in[ppid]);
        clSetKernelArg(insert_kernel[i], j++, sizeof(unsigned int), &batchsize);
        clSetKernelArg(insert_kernel[i], j++, sizeof(unsigned int), &order);
        clSetKernelArg(insert_kernel[i], j++, sizeof(unsigned int), &offset);
        clSetKernelArg(insert_kernel[i], j++, sizeof(cl_mem), &buf_k0[ppid]);
#ifdef API_DEBUG
        std::cout << "WARNING: insert_0 kernel " << i << " created" << std::endl;
#endif

        merge_kernels_0[i] = clCreateKernel(prg, "merge_kernel_0", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
            exit(1);
        }
        j = 0;
        offset = 0;
        int each_len = sz_128k;
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(cl_mem), &buf_k0[ppid]);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(unsigned int), &each_len);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(unsigned int), &batchsize);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(unsigned int), &order);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(unsigned int), &offset);
        clSetKernelArg(merge_kernels_0[i], j++, sizeof(cl_mem), &buf_k1[ppid]);
#ifdef API_DEBUG
        std::cout << "WARNING: merge_0 kernel " << i << " created" << std::endl;
#endif

        merge_kernels_1[i] = clCreateKernel(prg, "merge_kernel_1", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
            exit(1);
        }
        j = 0;
        offset = 0;
        each_len = sz_1m;
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(cl_mem), &buf_k1[ppid]);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(unsigned int), &each_len);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(unsigned int), &batchsize);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(unsigned int), &order);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(unsigned int), &offset);
        clSetKernelArg(merge_kernels_1[i], j++, sizeof(cl_mem), &buf_k2[ppid]);
#ifdef API_DEBUG
        std::cout << "WARNING: merge_1 kernel " << i << " created" << std::endl;
#endif

        merge_kernels_2[i] = clCreateKernel(prg, "merge_kernel_2", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
            exit(1);
        }
        j = 0;
        offset = i * sz_64m / 2;
        each_len = sz_8m;
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(cl_mem), &buf_k2[ppid]);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(unsigned int), &each_len);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(unsigned int), &batchsize);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(unsigned int), &order);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(unsigned int), &offset);
        clSetKernelArg(merge_kernels_2[i], j++, sizeof(cl_mem), &buf_k3);
#ifdef API_DEBUG
        std::cout << "WARNING: merge_2 kernel " << i << " created" << std::endl;
#endif
    }

    merge_kernels_3 = clCreateKernel(prg, "gqe_merge_kernel_3", &err);
    if (allsize > sz_64m) {
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create insert kernel.\n");
            exit(1);
        }
        int j = 0;
        int each_len = sz_64m;
        int offset = 0;
        clSetKernelArg(merge_kernels_3, j++, sizeof(cl_mem), &buf_k3);
        clSetKernelArg(merge_kernels_3, j++, sizeof(unsigned int), &each_len);
        clSetKernelArg(merge_kernels_3, j++, sizeof(unsigned int), &allsize);
        clSetKernelArg(merge_kernels_3, j++, sizeof(unsigned int), &order);
        clSetKernelArg(merge_kernels_3, j++, sizeof(unsigned int), &offset);
        clSetKernelArg(merge_kernels_3, j++, sizeof(cl_mem), &buf_k4);
#ifdef API_DEBUG
        std::cout << "WARNING: merge_3 kernel created" << std::endl;
#endif
    }

    for (int i = 0; i <= l_level; i++) {
        int rid = round;
        int ppid = (counter + rid) % 0x02;
        int batchsize = allsize - round * sz_64m;
        if (i == 0) {
            insert_kernel[rid] = clCreateKernel(prg, "insert_kernel_0", &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create insert kernel.\n");
                exit(1);
            }
            int j = 0;
            int offset = 0;
            clSetKernelArg(insert_kernel[rid], j++, sizeof(cl_mem), &buf_in[ppid]);
            clSetKernelArg(insert_kernel[rid], j++, sizeof(unsigned int), &batchsize);
            clSetKernelArg(insert_kernel[rid], j++, sizeof(unsigned int), &order);
            cl_mem buf_k0_ = buf_k0[ppid];
            if (rid != 0 && i == l_level) {
                buf_k0_ = buf_k3;
                offset = rid * sz_64m / 2;
            }
            clSetKernelArg(insert_kernel[rid], j++, sizeof(unsigned int), &offset);
            clSetKernelArg(insert_kernel[rid], j++, sizeof(cl_mem), &buf_k0_);
#ifdef API_DEBUG
            std::cout << "WARNING: insert kernel l_level created,rid is: " << rid << std::endl;
#endif
        } else if (i == 1) {
            merge_kernels_0[rid] = clCreateKernel(prg, "merge_kernel_0", &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create insert kernel.\n");
                exit(1);
            }
            int j = 0;
            int each_len = sz_128k;
            int offset = 0;
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(cl_mem), &buf_k0[ppid]);
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(unsigned int), &each_len);
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(unsigned int), &batchsize);
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(unsigned int), &order);
            std::cout << "ndebug subbuffer in m0" << std::endl;
            cl_mem buf_k1_ = buf_k1[ppid];
            if (rid != 0 && i == l_level) {
                buf_k1_ = buf_k3;
                offset = rid * sz_64m / 2;
            }
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(unsigned int), &offset);
            clSetKernelArg(merge_kernels_0[rid], j++, sizeof(cl_mem), &buf_k1_);

#ifdef API_DEBUG
            std::cout << "WARNING: merge_0 kernel l_level created,rid is: " << rid << std::endl;
#endif
        } else if (i == 2) {
            merge_kernels_1[rid] = clCreateKernel(prg, "merge_kernel_1", &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create insert kernel.\n");
                exit(1);
            }
            int j = 0;
            int each_len = sz_1m;
            int offset = 0;
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(cl_mem), &buf_k1[ppid]);
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(unsigned int), &each_len);
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(unsigned int), &batchsize);
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(unsigned int), &order);
            cl_mem buf_k2_ = buf_k2[ppid];
            if (rid != 0 && i == l_level) {
                buf_k2_ = buf_k3;
                offset = rid * sz_64m / 2;
            }
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(unsigned int), &offset);
            clSetKernelArg(merge_kernels_1[rid], j++, sizeof(cl_mem), &buf_k2_);

        } else if (i == 3) {
            merge_kernels_2[rid] = clCreateKernel(prg, "gqe_merge_kernel_2", &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create insert kernel.\n");
                exit(1);
            }
            int j = 0;
            int each_len = sz_8m;
            int offset = rid * sz_64m / 2;
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(cl_mem), &buf_k2[ppid]);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(unsigned int), &each_len);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(unsigned int), &batchsize);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(unsigned int), &order);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(unsigned int), &offset);
            clSetKernelArg(merge_kernels_2[rid], j++, sizeof(cl_mem), &buf_k3);
        }
    }
    std::cout << "Kernel has been created\n";

    // create events
    std::vector<cl_event> ev_w(round + 1);
    std::vector<cl_event> ev_i(round + 1);
    std::vector<cl_event> ev_m0(round + 1);
    std::vector<cl_event> ev_m1(round + 1);
    std::vector<cl_event> ev_m2(round + 1);

    cl_event ev_m3;
    cl_event ev_read;
    std::array<cl_event, sz_test_ch> for_last_merge;
    int dep_last_merge = 0;
    // read
    int loop_num = round;
    int loop_num_m0 = round;
    int loop_num_m1 = round;
    int loop_num_m2 = round;

    if (l_level >= 0) {
        loop_num += 1;
    }
    if (l_level >= 1) {
        loop_num_m0 += 1;
    }
    if (l_level >= 2) {
        loop_num_m1 += 1;
    }
    if (l_level >= 3) {
        loop_num_m2 += 1;
    }
#ifdef API_DEBUG
    std::cout << "start run kernels" << std::endl;
    std::cout << "loop_num " << loop_num << std::endl;
    std::cout << "loop_num_m0 " << loop_num_m0 << std::endl;
    std::cout << "loop_num_m1 " << loop_num_m1 << std::endl;
    std::cout << "loop_num_m2 " << loop_num_m2 << std::endl;
#endif

    for (int i = 0; i < loop_num; i++) {
        int ppid = (i + counter) % 0x02;
        int size_left = allsize - sz_64m * i;
        int size = size_left > sz_64m ? sz_64m : size_left;
#ifdef API_DEBUG
        std::cout << "run size in loop " << i << " is:" << size << std::endl;
#endif
        // clwrite
        std::array<cl_event, 2> for_w;
        int num_for_w = 0;
        if (evs_write[ppid].size() != 0) {
            for_w[num_for_w++] = evs_write[ppid][evs_write[ppid].size() - 1];
        }
        if (evs_insert[ppid].size() != 0) {
            for_w[num_for_w++] = evs_insert[ppid][evs_insert[ppid].size() - 1];
        }
        if (num_for_w == 0) {
            err = clEnqueueWriteBuffer(cq, buf_in[ppid], 0, 0, sizeof(ap_uint<64>) * (size), user_in[i], 0, NULL,
                                       &ev_w[i]);
        } else {
            err = clEnqueueWriteBuffer(cq, buf_in[ppid], 0, 0, sizeof(ap_uint<64>) * (size), user_in[i], num_for_w,
                                       for_w.data(), &ev_w[i]);
        }
        if (err != CL_SUCCESS) {
            std::cout << "EnqueueWriteBuffer FAILED" << std::endl;
            exit(1);
        } else {
#ifdef API_DEBUG
            std::cout << "EnqueueWriteBuffer SUCCESS" << std::endl;
#endif
        }
        evs_write[ppid].push_back(ev_w[i]);
        /*
        gettimeofday(&t_1, 0);
        clFinish(cq);
        x_utils::tvdiff(t_0, t_1, "Sort_future: write time");
        */

        // insert kernel
        std::array<cl_event, 2> for_insert_evs;
        for_insert_evs[0] = ev_w[i];
        int rep_num = 1;

        if (l_level == 0 && i == loop_num - 1) {
            if (loop_num != 1) {
                int m3_size = evs_m3.size();
                if (m3_size != 0) {
                    int last_m3_ind = m3_size - 1;
                    for_insert_evs[1] = evs_m3[last_m3_ind];
                    rep_num++;
                }
#ifdef API_DEBUG
                if (m3_size == 0) {
                    std::cout << "no event in evs_m3" << std::endl;
                }
#endif
            } else {
                int read_size = evs_read.size();
                if (read_size != 0) {
                    int last_read_ind = read_size - 1;
                    for_insert_evs[1] = evs_read[last_read_ind];
                    rep_num++;
                }
            }
        } else {
            int m0_size = evs_m0[ppid].size();
            if (m0_size != 0) {
                int last_m0_ind = m0_size - 1;
                for_insert_evs[1] = evs_m0[ppid][last_m0_ind];
                rep_num++;
            }
        }

        err = clEnqueueTask(cq, insert_kernel[i], rep_num, for_insert_evs.data(), &ev_i[i]);
        if (l_level == 0 && i == loop_num - 1 && loop_num != 1) {
            for_last_merge[dep_last_merge++] = ev_i[i];
        }
        if (err != CL_SUCCESS) {
            std::cout << "0 EnqueueTask insert FAILED" << std::endl;
            exit(1);
        } else {
#ifdef API_DEBUG
            std::cout << "0 EnqueueTask insert SUCCESS" << std::endl;
#endif
        }
        evs_insert[ppid].push_back(ev_i[i]);
        /*
        gettimeofday(&t_2, 0);
        clFinish(cq);
        x_utils::tvdiff(t_1, t_2, "Sort_future: insert kernel time");
        */

        // merge_0 kenrel
        if (i < loop_num_m0) {
            std::array<cl_event, 2> for_m0;
            for_m0[0] = ev_i[i];
            int rep_num = 1;

            if (l_level == 1 && i == loop_num - 1) {
                if (loop_num != 1) {
                    int m3_size = evs_m3.size();
                    if (m3_size != 0) {
                        int last_m3_ind = m3_size - 1;
                        for_m0[1] = evs_m3[last_m3_ind];
                        rep_num++;
                    }
                } else {
                    int read_size = evs_read.size();
                    if (read_size != 0) {
                        int last_read_ind = read_size - 1;
                        for_m0[1] = evs_read[last_read_ind];
                        rep_num++;
                    }
                }
            } else {
                int m1_size = evs_m1[ppid].size();
                if (m1_size != 0) {
                    int last_m1_ind = m1_size - 1;
                    for_m0[1] = evs_m1[ppid][last_m1_ind];
                    rep_num++;
                }
            }

            err = clEnqueueTask(cq, merge_kernels_0[i], rep_num, for_m0.data(), &ev_m0[i]);
            if (l_level == 1 && i == loop_num - 1 && loop_num != 1) {
                for_last_merge[dep_last_merge++] = ev_m0[i];
            }
            if (err != CL_SUCCESS) {
                std::cout << "0 EnqueueTask merge_0 FAILED" << std::endl;
                exit(1);
            } else {
#ifdef API_DEBUG
                std::cout << "0 EnqueueTask merge_0 SUCCESS" << std::endl;
#endif
            }
            evs_m0[ppid].push_back(ev_m0[i]);
        }

        // merge_1 kenrel
        if (i < loop_num_m1) {
            std::array<cl_event, 2> for_m1;
            for_m1[0] = ev_m0[i];
            int rep_num = 1;
            if (l_level == 2 && i == loop_num - 1) {
                if (loop_num != 1) {
                    int m3_size = evs_m3.size();
                    if (m3_size != 0) {
                        int last_m3_ind = m3_size - 1;
                        for_m1[1] = evs_m3[last_m3_ind];
                        rep_num++;
                    }
                } else {
                    int read_size = evs_read.size();
                    if (read_size != 0) {
                        int last_read_ind = read_size - 1;
                        for_m1[1] = evs_read[last_read_ind];
                        rep_num++;
                    }
                }
            } else {
                int m2_size = evs_m2[ppid].size();
                if (m2_size != 0) {
                    int last_m2_ind = m2_size - 1;
                    for_m1[1] = evs_m2[ppid][last_m2_ind];
                    rep_num++;
                }
            }

            err = clEnqueueTask(cq, merge_kernels_1[i], rep_num, for_m1.data(), &ev_m1[i]);
            if (l_level == 2 && i == loop_num - 1 && loop_num != 1) {
                for_last_merge[dep_last_merge++] = ev_m1[i];
            }
            if (err != CL_SUCCESS) {
                std::cout << "0 EnqueueTask merge_1 FAILED" << std::endl;
                exit(1);
            } else {
#ifdef API_DEBUG
                std::cout << "0 EnqueueTask merge_1 SUCCESS" << std::endl;
#endif
            }
            evs_m1[ppid].push_back(ev_m1[i]);
        }

        // merge_2 kenrel
        if (i < loop_num_m2) {
            std::array<cl_event, 2> for_m2;
            for_m2[0] = ev_m1[i];
            int rep_num = 1;

            if (l_level == 3 && loop_num == 1) {
                int read_size = evs_read.size();
                if (read_size != 0) {
                    int last_read_ind = read_size - 1;
                    for_m2[1] = evs_read[last_read_ind];
                    rep_num++;
                }
            } else {
                int m3_size = evs_m3.size();
                if (m3_size != 0) {
                    int last_m3_ind = m3_size - 1;
                    for_m2[1] = evs_m3[last_m3_ind];
                    rep_num++;
                }
            }

            err = clEnqueueTask(cq, merge_kernels_2[i], rep_num, for_m2.data(), &ev_m2[i]);
            for_last_merge[dep_last_merge++] = ev_m2[i];
            if (err != CL_SUCCESS) {
                std::cout << "0 EnqueueTask merge_2 FAILED" << std::endl;
                exit(1);
            } else {
#ifdef API_DEBUG
                std::cout << "0 EnqueueTask merge_2 SUCCESS" << std::endl;
#endif
            }
            evs_m2[ppid].push_back(ev_m2[i]);
        }
    }

    if (allsize > sz_64m) {
        int read_size = evs_read.size();
        if (read_size != 0) {
            int last_read_ind = read_size - 1;
            for_last_merge[dep_last_merge++] = evs_read[last_read_ind];
        }

        err = clEnqueueTask(cq, merge_kernels_3, dep_last_merge, for_last_merge.data(), &ev_m3);
        if (err != CL_SUCCESS) {
            std::cout << "0 EnqueueTask merge_3 FAILED" << std::endl;
            exit(1);
        } else {
#ifdef API_DEBUG
            std::cout << "0 EnqueueTask merge_3 SUCCESS" << std::endl;
#endif
        }
        evs_m3.push_back(ev_m3);
    }
    // read
    cl_mem buf_out;
    cl_event for_ev_read;
    if (round == 0) {
        if (l_level == 0) {
            for_ev_read = ev_i[0];
            buf_out = buf_k0[0];
        } else if (l_level == 1) {
            for_ev_read = ev_m0[0];
            buf_out = buf_k1[0];
        } else if (l_level == 2) {
            for_ev_read = ev_m1[0];
            buf_out = buf_k2[0];
        } else if (l_level == 3) {
            for_ev_read = ev_m2[0];
            buf_out = buf_k3;
        } else {
            for_ev_read = ev_m2[0];
            buf_out = buf_k3;
            std::cout << "l_level went wrong!" << std::endl;
            exit(1);
        }
#ifdef API_DEBUG
        std::cout << "buf_out is buf_k3" << std::endl;
#endif
    } else if (round == 1 && l_level == -1) {
        for_ev_read = ev_m2[0];
        buf_out = buf_k3;
#ifdef API_DEBUG
        std::cout << "1 buf_out is buf_k3_subs[0]" << std::endl;
#endif
    } else {
        for_ev_read = ev_m3;
        buf_out = buf_k4;
#ifdef API_DEBUG
        std::cout << "buf_out is buf_k4" << std::endl;
#endif
    }

#ifdef API_DEBUG
// clFinish(cq);
#endif
    std::vector<cl_event> ev_read_(loop_num);
    for (int i = 0; i < loop_num; i++) {
        int size_left = allsize - sz_64m * i;
        int size = size_left > sz_64m ? sz_64m : size_left;

        std::array<cl_event, 2> for_ev_reads;
        for_ev_reads[0] = for_ev_read;
        int dep_num = 1;

        if (i > 0) {
            for_ev_reads[1] = ev_read_[i - 1];
            dep_num = 2;
        }

        clEnqueueReadBuffer(cq, buf_out, 0 /* non-blocking */, //
                            sizeof(ap_uint<64>) * i * sz_64m, sizeof(ap_uint<64>) * size,
                            user_out[i], //
                            dep_num, for_ev_reads.data(), &ev_read_[i]);
    }
    ev_read = ev_read_[loop_num - 1];

    // callback_data cdata;
    // cdata.prom = std::move(prom);
    // clSetEventCallback(ev_read, CL_COMPLETE, callback_func, &cdata);
    evs_read.push_back(ev_read);
    // counter++;
    counter = (counter + round_) % 2;
    std::thread(&sortAPI::setProm, this, ev_read, std::move(prom)).detach();
    return user_future;
}
void sortAPI::sort_64m_once(ap_uint<64>* user_in, ap_uint<64>* user_out, size_t size, int order, int ppid) {
    // timeval t_0, t_1, t_2;
    // gettimeofday(&t_0, 0);
    cl_kernel insert_kernel = clCreateKernel(prg, "gqe_insert_kernel_0", &err);
    cl_kernel merge_kernel_0 = clCreateKernel(prg, "gqe_merge_kernel_0", &err);
    cl_kernel merge_kernel_1 = clCreateKernel(prg, "gqe_merge_kernel_1", &err);
    cl_kernel merge_kernel_2 = clCreateKernel(prg, "gqe_merge_kernel_2", &err);

    int batchsize = size;
    int j = 0;
    int offset = 0;
    clSetKernelArg(insert_kernel, j++, sizeof(cl_mem), &buf_in[ppid]);
    clSetKernelArg(insert_kernel, j++, sizeof(unsigned int), &batchsize);
    clSetKernelArg(insert_kernel, j++, sizeof(unsigned int), &order);
    clSetKernelArg(insert_kernel, j++, sizeof(unsigned int), &offset);
    clSetKernelArg(insert_kernel, j++, sizeof(cl_mem), &buf_k0[ppid]);

    j = 0;
    offset = 0;
    int each_len = sz_128k;
    clSetKernelArg(merge_kernel_0, j++, sizeof(cl_mem), &buf_k0[ppid]);
    clSetKernelArg(merge_kernel_0, j++, sizeof(unsigned int), &each_len);
    clSetKernelArg(merge_kernel_0, j++, sizeof(unsigned int), &batchsize);
    clSetKernelArg(merge_kernel_0, j++, sizeof(unsigned int), &order);
    clSetKernelArg(merge_kernel_0, j++, sizeof(unsigned int), &offset);
    clSetKernelArg(merge_kernel_0, j++, sizeof(cl_mem), &buf_k1[ppid]);

    j = 0;
    offset = 0;
    each_len = sz_1m;
    clSetKernelArg(merge_kernel_1, j++, sizeof(cl_mem), &buf_k1[ppid]);
    clSetKernelArg(merge_kernel_1, j++, sizeof(unsigned int), &each_len);
    clSetKernelArg(merge_kernel_1, j++, sizeof(unsigned int), &batchsize);
    clSetKernelArg(merge_kernel_1, j++, sizeof(unsigned int), &order);
    clSetKernelArg(merge_kernel_1, j++, sizeof(unsigned int), &offset);
    clSetKernelArg(merge_kernel_1, j++, sizeof(cl_mem), &buf_k2[ppid]);

    j = 0;
    offset = ppid * sz_64m / 2;
    each_len = sz_8m;
    clSetKernelArg(merge_kernel_2, j++, sizeof(cl_mem), &buf_k2[ppid]);
    clSetKernelArg(merge_kernel_2, j++, sizeof(unsigned int), &each_len);
    clSetKernelArg(merge_kernel_2, j++, sizeof(unsigned int), &batchsize);
    clSetKernelArg(merge_kernel_2, j++, sizeof(unsigned int), &order);
    clSetKernelArg(merge_kernel_2, j++, sizeof(unsigned int), &offset);
    clSetKernelArg(merge_kernel_2, j++, sizeof(cl_mem), &buf_k3);

    // gettimeofday(&t_1, 0);
    cl_event ev_w, ev_r, ev_k, ev_m0, ev_m1, ev_m2;
    err = clEnqueueWriteBuffer(cq, buf_in[0], 0, 0, sizeof(ap_uint<64>) * (size), user_in, 0, NULL, &ev_w);
    err = clEnqueueTask(cq, insert_kernel, 1, &ev_w, &ev_k);
    err = clEnqueueTask(cq, merge_kernel_0, 1, &ev_k, &ev_m0);
    err = clEnqueueTask(cq, merge_kernel_1, 1, &ev_m0, &ev_m1);
    err = clEnqueueTask(cq, merge_kernel_2, 1, &ev_m1, &ev_m2);
    clEnqueueReadBuffer(cq, buf_k3, 0, 0, sizeof(ap_uint<64>) * size, user_out, 1, &ev_m2, &ev_r);
    clFinish(cq);
    // gettimeofday(&t_2, 0);
    // x_utils::tvdiff(t_0, t_1, "Sort_once: set kernel time");
    // x_utils::tvdiff(t_1, t_2, "Sort_once: run kernel time");
    // x_utils::tvdiff(t_0, t_2, "Sort_once: all kernel time");
}
} // namespace intpair_sort
} // namespace database
} // namespace xf

#ifdef L3_WITH_SERVER
namespace arrow {
namespace flight {

using BatchVector = std::vector<std::shared_ptr<RecordBatch> >;

class IntSortServer : public FlightServerBase {
   private:
    using SortBufferVector = std::vector<uint64_t*>;
    int counter_;
    BatchVector cur_tsk;
    std::map<std::string, BatchVector> flights;
    std::map<std::string, std::future<xf::database::intpair_sort::ErrCode> > futures;

    xf::database::intpair_sort::sortAPI sortapi;

    MemoryPool* pool;
    // std::map<FlightInfo, BatchVector> flights;

   public:
    IntSortServer(std::string xclbin_path) {
        sortapi.init(xclbin_path);
        pool = default_memory_pool();
    }
    Status ListFlights(const ServerCallContext& context,
                       const Criteria* criteria,
                       std::unique_ptr<FlightListing>* listings) override {
        // reserve todo
        return Status::OK();
    }

    Status GetFlightInfo(const ServerCallContext& context,
                         const FlightDescriptor& request,
                         std::unique_ptr<FlightInfo>* out) override {
        // reserve todo
        return Status::Invalid("Flight not found: ", request.ToString());
    }

    FlightDescriptor makeDescriptor(FlightDescriptor descriptor_) {
        Location location;
        FlightEndpoint endpoint;
        // loction not in use
        ARROW_EXPECT_OK(Location::ForGrpcTcp("foo1.bar.com", 12345, &location));
        std::string ticket;
        if (descriptor_.type == FlightDescriptor::DescriptorType::CMD) {
            ticket = descriptor_.cmd; // use timestamp and filename as ticket
            endpoint = FlightEndpoint({{ticket}, {location}});
        } else {
            // ticket = descriptor_.path;
        }
        FlightDescriptor descr{FlightDescriptor::CMD, "int sort", {}};
        std::cout << "do_put: ticket is " << ticket << std::endl;
        //------construct a new FlightDescriptor-------//
        return descr;
    }

    Status DoPut(const ServerCallContext& context,
                 std::unique_ptr<FlightMessageReader> reader,
                 std::unique_ptr<FlightMetadataWriter> writer) override {
        // user descriptor, get ticket from the
        FlightDescriptor descriptor_ = reader->descriptor();
        std::string flight_info = descriptor_.cmd;
        int sort_info = (flight_info.c_str())[0] - '0';
        std::cout << "sort_info:" << sort_info << std::endl;
        BatchVector batches_;
        Status s = reader->ReadAll(&batches_);
        if (s != Status::OK()) {
            std::cout << "Read RecordBatch Error!" << std::endl;
            exit(1);
        }
        std::cout << "Read RecordBatch Success!" << std::endl;
        int32_t bufferLen = 0;
        int batch_num = batches_.size();
        std::vector<const uint64_t*> buffer_in(batch_num);
        std::vector<uint64_t*> buffer_out(batch_num);
        std::vector<int32_t> buffer_out_len(batch_num);

        for (int i = 0; i < batch_num; i++) {
            std::shared_ptr<ArrayData> arraydata = batches_[i]->column_data(0);

            const int32_t bufferLen_each = batches_[i]->num_rows();
            const int32_t bufferSize_each = sizeof(uint64_t) * bufferLen_each;
            buffer_out_len[i] = bufferLen_each;
            bufferLen += bufferLen_each;
            ::arrow::Status st_out = pool->Allocate(bufferSize_each, (uint8_t**)(&buffer_out[i]));
            if (!st_out.ok()) {
                std::cout << "failed with allocate buffer_out" << std::endl;
            }

            buffer_in[i] = arraydata->GetValues<uint64_t>(1); // buffer 0 is nullptr
            std::cout << "Round:" << i << ",ArrayData length:" << bufferLen_each
                      << ",ArrayData buffer num:" << (arraydata->buffers).size()
                      << ",ArrayData buffe1 size:" << (arraydata->buffers)[1]->size()
                      << ",ArrayData offset:" << arraydata->offset << std::endl;
            std::cout << "Round " << i << " done" << std::endl;
        }
        std::future<xf::database::intpair_sort::ErrCode> f =
            sortapi.sort_server(buffer_in, buffer_out, bufferLen, sort_info);
        f.wait();
        /*
        for (int j = 0; j < batch_num; j++) {
          memcpy(buffer_out[j], buffer_in[j], sizeof(uint64_t) * buffer_out_len[j]);
        }
        */
        BatchVector batches_out;
        for (int j = 0; j < batch_num; j++) {
            std::cout << "Sort Done, Round " << j << std::endl;
            for (int i = 0; i < 10; i++) {
                std::cout << buffer_out[j][i] << ", ";
            }
            std::cout << std::endl;

            std::shared_ptr<Buffer> buffer_out_ = std::make_shared<Buffer>(
                reinterpret_cast<const uint8_t*>(buffer_out[j]), buffer_out_len[j] * sizeof(uint64_t));
            // DEBUG
            // const uint64_t* buffer_out_data =
            //     reinterpret_cast<const uint64_t*>(buffer_out_->data());

            auto arraydata_out = ArrayData::Make(uint64(), buffer_out_len[j], {nullptr, buffer_out_});

            auto f1 = field("data", uint64());
            std::vector<std::shared_ptr<Field> > fields = {f1};
            std::shared_ptr<Schema> schema = std::make_shared<Schema>(fields);
            std::shared_ptr<RecordBatch> rb = RecordBatch::Make(schema, buffer_out_len[j], {arraydata_out});
            batches_out.push_back(rb);

            std::cout << "RecordBatch num_rows: " << rb->num_rows() << ",RecordBatch num_columns:" << rb->num_columns()
                      << ",RecordBatch Schema" << (rb->schema())->ToString() << std::endl;
        }
        flights[flight_info] = batches_out;
        // todo: ticket can descript batch or single tasks
        //------construct a new FlightDescriptor-------//
        /* for test
        cur_tsk.clear();
        Status s = reader->ReadAll(&cur_tsk);
        */
        return Status::OK();
    }

    Status DoGet(const ServerCallContext& context,
                 const Ticket& request,
                 std::unique_ptr<FlightDataStream>* data_stream) override {
        std::shared_ptr<RecordBatchReader> batch_reader;
        std::cout << "do_get: ticket is " << request.ticket << std::endl;
        BatchVector batches_ = flights[request.ticket];
        // FlightInfo flight_info = makeFlightInfo(descriptor_, batches_);

        batch_reader = std::make_shared<BatchIterator>(batches_[0]->schema(), batches_);
        *data_stream = std::unique_ptr<FlightDataStream>(new RecordBatchStream(batch_reader));

        return Status::OK();
    }
    Status ListActions(const ServerCallContext& context, std::vector<ActionType>* out) override {
        // todo: reserve
        return Status::OK();
    }

    Status DoAction(const ServerCallContext& context,
                    const Action& action,
                    std::unique_ptr<ResultStream>* result) override {
        // todo: reserve
        return Status::OK();
    }
};

std::unique_ptr<FlightServerBase> IntSortServerInst(std::string xclbin_path) {
    return std::unique_ptr<FlightServerBase>(new IntSortServer(xclbin_path));
}

} // namespace flight
} // namespace arrow
#endif
