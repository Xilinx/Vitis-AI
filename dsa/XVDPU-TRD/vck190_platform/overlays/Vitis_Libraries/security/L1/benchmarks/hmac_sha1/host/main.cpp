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
#include <iostream>

#include <openssl/sha.h>

#include <sys/time.h>
#include <new>
#include <cstdlib>

#ifndef HLS_TEST
#include <xcl2.hpp>
#include "xf_utils_sw/logger.hpp"
#endif

#include <vector>

#include "kernel_config.hpp"

// max text length
#define MAX_N_MSG 4096

// text length for each task in Byte
#define N_MSG 1024
// number of tasks for a single PCIe block
#define N_TASK 20 // 8192
// cipher key size in bytes
#define KEY_SIZE 32

#ifdef HLS_TEST
extern "C" void hmacSha1Kernel_1(ap_uint<512> inputData[(1 << 20) + 100], ap_uint<512> outputData[1 << 20]);
#endif

ap_uint<GRP_WIDTH> char2ap_uint(unsigned char* data, int ptr) {
    ap_uint<GRP_WIDTH> tmp = 0;
    for (int i = 0; i < GRP_SIZE; i++) {
        tmp.range(i * 8 + 7, i * 8) = data[ptr + i];
    }
    return tmp;
}

void hmacSHA1(const unsigned char* key,
              unsigned int keyLen,
              const unsigned char* message,
              unsigned int msgLen,
              unsigned char* h) {
    const int MSG_SIZE = 4;             // size of each message word in byte
    const int HASH_SIZE = 5 * MSG_SIZE; // size of hash value in byte
    const int MAX_MSG = 4096;           // max size of message in byte
    const int BLOCK_SIZE = 64;          // size of SHA1 block

    unsigned char kone[BLOCK_SIZE + 8] = {0};
    unsigned char kipad[BLOCK_SIZE + 8] = {0};
    unsigned char kopad[BLOCK_SIZE + 8] = {0};
    unsigned char kmsg[BLOCK_SIZE + MAX_MSG + 8] = {0};
    unsigned char khsh[BLOCK_SIZE + HASH_SIZE + 8] = {0};
    unsigned char h1[HASH_SIZE + 8] = {0};
    unsigned char h2[HASH_SIZE + 8] = {0};

    if (keyLen > BLOCK_SIZE) {
        SHA1((const unsigned char*)key, keyLen, (unsigned char*)h1);
        memcpy(kone, h1, HASH_SIZE);
    } else
        memcpy(kone, key, keyLen);

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        kipad[i] = (unsigned int)(kone[i]) ^ 0x36;
        kopad[i] = (unsigned int)(kone[i]) ^ 0x5c;
    }

    memcpy(kmsg, kipad, BLOCK_SIZE);
    memcpy(kmsg + BLOCK_SIZE, message, msgLen);
    SHA1((const unsigned char*)kmsg, BLOCK_SIZE + msgLen, (unsigned char*)h2);

    memcpy(khsh, kopad, BLOCK_SIZE);
    memcpy(khsh + BLOCK_SIZE, h2, HASH_SIZE);
    SHA1((const unsigned char*)khsh, BLOCK_SIZE + HASH_SIZE, (unsigned char*)h);
}

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
    // set repeat time
    int num_rep = 2;
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

    // set N_TASK and N_MSG
    int n_task, n_msg;
    if (parser.getCmdOption("-task", num_str)) {
        try {
            n_task = std::stoi(num_str);
        } catch (...) {
            n_task = N_TASK;
        }
    }
    if (parser.getCmdOption("-msg", num_str)) {
        try {
            n_msg = std::stoi(num_str);
        } catch (...) {
            n_msg = N_MSG;
        }
    }
    n_task = N_TASK;
    n_msg = N_MSG;
    std::cout << "task num : " << n_task << std::endl;
    std::cout << "msg length in byte : " << n_msg << std::endl;

    // input data
    unsigned char datain[256];
    for (int i = 0; i < 256; i++) {
        datain[i] = (unsigned char)i;
    }

    unsigned char messagein[MAX_N_MSG];
    for (int i = 0; i < n_msg; i++) {
        messagein[i] = datain[i % 256];
    }

    // cipher key
    const unsigned char key[] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a,
                                 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
                                 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f};

    // generate golden
    unsigned char hmacResult[20];
    hmacSHA1(key, 32, messagein, n_msg, hmacResult);

    // ouput length of the result

    ap_uint<512> golden = 0;
    for (unsigned int j = 0; j < 20; j++) {
        golden.range(j * 8 + 7, j * 8) = hmacResult[j];
    }

    ap_uint<8 * KEY_SIZE> keyReg;
    for (unsigned int i = 0; i < KEY_SIZE; i++) {
        keyReg.range(i * 8 + 7, i * 8) = key[i];
    }

    ap_uint<128> dataReg;
    for (unsigned int i = 0; i < 16; i++) {
        dataReg.range(i * 8 + 7, i * 8) = datain[i];
    }

    std::cout << "Goldens have been created using OpenSSL.\n";

    // Host buffers
    ap_uint<512>* hb_in1 = aligned_alloc<ap_uint<512> >(n_msg * n_task * CH_NM / 64 + CH_NM);
    ap_uint<512>* hb_in2 = aligned_alloc<ap_uint<512> >(n_msg * n_task * CH_NM / 64 + CH_NM);
    ap_uint<512>* hb_in3 = aligned_alloc<ap_uint<512> >(n_msg * n_task * CH_NM / 64 + CH_NM);
    ap_uint<512>* hb_in4 = aligned_alloc<ap_uint<512> >(n_msg * n_task * CH_NM / 64 + CH_NM);

    ap_uint<512>* hb_out_a[4];
    for (int i = 0; i < 4; i++) {
        hb_out_a[i] = aligned_alloc<ap_uint<512> >(n_task * CH_NM);
    }
    ap_uint<512>* hb_out_b[4];
    for (int i = 0; i < 4; i++) {
        hb_out_b[i] = aligned_alloc<ap_uint<512> >(n_task * CH_NM);
    }

    // generate configurations
    for (unsigned int j = 0; j < CH_NM; j++) {
        // massage length in 128-bit for each task
        hb_in1[j].range(511, 448) = n_msg;
        hb_in2[j].range(511, 448) = n_msg;
        hb_in3[j].range(511, 448) = n_msg;
        hb_in4[j].range(511, 448) = n_msg;

        // number of tasks in a single PCIe block
        hb_in1[j].range(447, 384) = n_task;
        hb_in2[j].range(447, 384) = n_task;
        hb_in3[j].range(447, 384) = n_task;
        hb_in4[j].range(447, 384) = n_task;

        // cipherkey
        hb_in1[j].range(255, 0) = keyReg.range(255, 0);
        hb_in2[j].range(255, 0) = keyReg.range(255, 0);
        hb_in3[j].range(255, 0) = keyReg.range(255, 0);
        hb_in4[j].range(255, 0) = keyReg.range(255, 0);
    }
    // generate texts
    for (int i = 0; i < n_task; i++) {
        for (int j = 0; j < n_msg; j += GRP_SIZE) {
            int pos = (j / GRP_SIZE) + i * (n_msg / GRP_SIZE) + CH_NM;
            for (int l = 0; l < SUB_GRP_SZ; l++) {
                hb_in1[pos].range(l * GRP_WIDTH + GRP_WIDTH - 1, l * GRP_WIDTH) = char2ap_uint(messagein, j);
                hb_in2[pos].range(l * GRP_WIDTH + GRP_WIDTH - 1, l * GRP_WIDTH) = char2ap_uint(messagein, j);
                hb_in3[pos].range(l * GRP_WIDTH + GRP_WIDTH - 1, l * GRP_WIDTH) = char2ap_uint(messagein, j);
                hb_in4[pos].range(l * GRP_WIDTH + GRP_WIDTH - 1, l * GRP_WIDTH) = char2ap_uint(messagein, j);
            }
        }
    }
    std::cout << "Host map buffer has been allocated and set.\n";

#ifndef HLS_TEST
    // Get CL devices.
    xf::common::utils_sw::Logger logger;
    cl_int err = CL_SUCCESS;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Create context and command queue for selected device
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected Device " << devName << "\n";

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);

    cl::Kernel kernel0(program, "hmacSha1Kernel_1", &err);
    logger.logCreateKernel(err);
    cl::Kernel kernel1(program, "hmacSha1Kernel_2", &err);
    logger.logCreateKernel(err);
    cl::Kernel kernel2(program, "hmacSha1Kernel_3", &err);
    logger.logCreateKernel(err);
    cl::Kernel kernel3(program, "hmacSha1Kernel_4", &err);
    logger.logCreateKernel(err);
    std::cout << "Kernel has been created.\n";

    cl_mem_ext_ptr_t mext_in[4];
    mext_in[0] = {XCL_MEM_DDR_BANK0, hb_in1, 0};
    mext_in[1] = {XCL_MEM_DDR_BANK1, hb_in2, 0};
    mext_in[2] = {XCL_MEM_DDR_BANK2, hb_in3, 0};
    mext_in[3] = {XCL_MEM_DDR_BANK3, hb_in4, 0};

    cl_mem_ext_ptr_t mext_out_a[4];
    mext_out_a[0] = {XCL_MEM_DDR_BANK0, hb_out_a[0], 0};
    mext_out_a[1] = {XCL_MEM_DDR_BANK1, hb_out_a[1], 0};
    mext_out_a[2] = {XCL_MEM_DDR_BANK2, hb_out_a[2], 0};
    mext_out_a[3] = {XCL_MEM_DDR_BANK3, hb_out_a[3], 0};

    cl_mem_ext_ptr_t mext_out_b[4];
    mext_out_b[0] = {XCL_MEM_DDR_BANK0, hb_out_b[0], 0};
    mext_out_b[1] = {XCL_MEM_DDR_BANK1, hb_out_b[1], 0};
    mext_out_b[2] = {XCL_MEM_DDR_BANK2, hb_out_b[2], 0};
    mext_out_b[3] = {XCL_MEM_DDR_BANK3, hb_out_b[3], 0};

    // ping buffer
    cl::Buffer in_buff_a[4];
    cl::Buffer out_buff_a[4];
    // pong buffer
    cl::Buffer in_buff_b[4];
    cl::Buffer out_buff_b[4];

    // Map buffers
    for (int i = 0; i < 4; i++) {
        in_buff_a[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  (size_t)(sizeof(ap_uint<512>) * (n_msg * n_task * CH_NM / 64 + CH_NM)), &mext_in[i]);
        out_buff_a[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                   (size_t)(sizeof(ap_uint<512>) * (n_task * CH_NM)), &mext_out_a[i]);
        in_buff_b[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  (size_t)(sizeof(ap_uint<512>) * (n_msg * n_task * CH_NM / 64 + CH_NM)), &mext_in[i]);
        out_buff_b[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                   (size_t)(sizeof(ap_uint<512>) * (n_task * CH_NM)), &mext_out_b[i]);
    }

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
        kernel_events[i].resize(4);
        read_events[i].resize(1);
    }
    std::cout << "num_rep " << num_rep << std::endl;
    /*
     * W0-. W1----.     W2-.     W3-.
     *    '-K0--. '-K1-/-. '-K2-/-. '-K3---.
     *          '---R0-  '---R1-  '---R2   '--R3
     */

    for (int i = 0; i < num_rep; i++) {
        int use_a = i & 1;

        // write data to DDR
        std::vector<cl::Memory> ib;
        if (use_a) {
            ib.push_back(in_buff_a[0]);
            ib.push_back(in_buff_a[1]);
            ib.push_back(in_buff_a[2]);
            ib.push_back(in_buff_a[3]);
        } else {
            ib.push_back(in_buff_b[0]);
            ib.push_back(in_buff_b[1]);
            ib.push_back(in_buff_b[2]);
            ib.push_back(in_buff_b[3]);
        }

        if (i > 1) {
            q.enqueueMigrateMemObjects(ib, 0, &read_events[i - 2], &write_events[i][0]);
        } else {
            q.enqueueMigrateMemObjects(ib, 0, nullptr, &write_events[i][0]);
        }
        // set args and enqueue kernel

        if (use_a) {
            int j = 0;
            kernel0.setArg(j++, in_buff_a[0]);
            kernel0.setArg(j++, out_buff_a[0]);
            j = 0;
            kernel1.setArg(j++, in_buff_a[1]);
            kernel1.setArg(j++, out_buff_a[1]);
            j = 0;
            kernel2.setArg(j++, in_buff_a[2]);
            kernel2.setArg(j++, out_buff_a[2]);
            j = 0;
            kernel3.setArg(j++, in_buff_a[3]);
            kernel3.setArg(j++, out_buff_a[3]);
        } else {
            int j = 0;
            kernel0.setArg(j++, in_buff_b[0]);
            kernel0.setArg(j++, out_buff_b[0]);
            j = 0;
            kernel1.setArg(j++, in_buff_b[1]);
            kernel1.setArg(j++, out_buff_b[1]);
            j = 0;
            kernel2.setArg(j++, in_buff_b[2]);
            kernel2.setArg(j++, out_buff_b[2]);
            j = 0;
            kernel3.setArg(j++, in_buff_b[3]);
            kernel3.setArg(j++, out_buff_b[3]);
        }

        q.enqueueTask(kernel0, &write_events[i], &kernel_events[i][0]);
        q.enqueueTask(kernel1, &write_events[i], &kernel_events[i][1]);
        q.enqueueTask(kernel2, &write_events[i], &kernel_events[i][2]);
        q.enqueueTask(kernel3, &write_events[i], &kernel_events[i][3]);
        // read data from DDR
        std::vector<cl::Memory> ob;
        if (use_a) {
            ob.push_back(out_buff_a[0]);
            ob.push_back(out_buff_a[1]);
            ob.push_back(out_buff_a[2]);
            ob.push_back(out_buff_a[3]);
        } else {
            ob.push_back(out_buff_b[0]);
            ob.push_back(out_buff_b[1]);
            ob.push_back(out_buff_b[2]);
            ob.push_back(out_buff_b[3]);
        }

        q.enqueueMigrateMemObjects(ob, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[i], &read_events[i][0]);
        // q.enqueueMigrateMemObjects(ob, CL_MIGRATE_MEM_OBJECT_HOST, &write_events[i], &read_events[i][0]);
    }
#endif
    // wait all to finish
    q.flush();
    q.finish();

#ifdef HLS_TEST
    hmacSha1Kernel_1(hb_in1, hb_out_a[0]);
#endif
    gettimeofday(&end_time, 0);

    // check result
    bool checked = true;

    // check ping buffer
    std::cout << "check ping buffer" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < n_task; j++) {
            for (int k = 0; k < CH_NM; k++) {
                if (hb_out_a[i][j * CH_NM + k] != golden) {
                    checked = false;
                    std::cout << std::dec << i << "th kernel " << j << "th message " << k
                              << "th channel's result is wrong" << std::endl;
                    std::cout << std::hex << "golden: " << golden << std::endl;
                    std::cout << std::hex << "result: " << hb_out_a[i][j * CH_NM + k] << std::endl;
                }
            }
        }
    }
#ifndef HLS_TEST
    // check pong buffer
    std::cout << "check pong buffer" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < n_task; j++) {
            for (int k = 0; k < CH_NM; k++) {
                if (hb_out_b[i][j * CH_NM + k] != golden) {
                    checked = false;
                    std::cout << std::dec << i << "th kernel " << j << "th message " << k
                              << "th channel's result is wrong" << std::endl;
                    std::cout << std::hex << "golden: " << golden << std::endl;
                    std::cout << std::hex << "result: " << hb_out_b[i][j * CH_NM + k] << std::endl;
                }
            }
        }
    }
#endif
    // final output
    std::cout << std::dec << std::endl;
    if (checked) {
        std::cout << std::dec << CH_NM << " channels, " << n_task << " tasks, " << n_msg
                  << " bytes message each verified. No error found!" << std::endl;
    }

    std::cout << "Kernel has been run for " << std::dec << num_rep << " times." << std::endl;
    std::cout << "Total execution time " << tvdiff(&start_time, &end_time) << "us" << std::endl;

#ifndef HLS_TEST
    free(hb_in1);
    std::cout << "hb_in1 free" << std::endl;
    free(hb_in2);
    std::cout << "hb_in1 free" << std::endl;
    free(hb_in3);
    std::cout << "hb_in1 free" << std::endl;
    free(hb_in4);
    std::cout << "hb_in1 free" << std::endl;
    for (int i = 0; i < 4; i++) {
        free(hb_out_a[i]);
        std::cout << "hb_out_a[" << i << "] free" << std::endl;
        free(hb_out_b[i]);
        std::cout << "hb_out_b[" << i << "] free" << std::endl;
    }
#endif
    if (checked) {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
        return 0;
    } else {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
        return 1;
    }
}
