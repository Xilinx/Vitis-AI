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

#include <openssl/rc4.h>
#include <openssl/evp.h>

#include <sys/time.h>
#include <new>
#include <cstdlib>

#include <xcl2.hpp>
#include "xf_utils_sw/logger.hpp"

// text length for each task in byte
#define N_ROW 2048
//#define N_ROW 2097152
// number of tasks for a single PCIe block
#define N_TASK 2
// number of PUs
// XXX: should be a multiple of 2
#define CH_NM 12
// cipher key size in byte
#define KEY_SIZE 32

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
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR:xclbin path is not set!\n";
        return 1;
    }

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

    std::cout << "Starting test.\n";

    // input data
    const char datain[] = {0x01};
    //    const char datain2[] = {0x01};
    const char datain2[] = {0x7e};

    // cipher key
    const unsigned char key[] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
                                 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29,
                                 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36,
                                 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43,
                                 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f};

    // generate golden
    // ouput length of the result
    int outlen = 0;
    // output result
    unsigned char golden[4][N_ROW];

    // call OpenSSL API to get the golden
    EVP_CIPHER_CTX* ctx;
    ctx = EVP_CIPHER_CTX_new();
    EVP_CipherInit_ex(ctx, EVP_rc4(), NULL, NULL, NULL, 1);
    EVP_CIPHER_CTX_set_key_length(ctx, KEY_SIZE);
    EVP_CipherInit_ex(ctx, NULL, NULL, key, NULL, 1);
    for (unsigned int i = 0; i < N_ROW; i++) {
        EVP_CipherUpdate(ctx, golden[0] + i, &outlen, (const unsigned char*)datain, 1);
        i++;
        EVP_CipherUpdate(ctx, golden[0] + i, &outlen, (const unsigned char*)datain2, 1);
    }
    EVP_CIPHER_CTX_free(ctx);

    outlen = 0;
    ctx = EVP_CIPHER_CTX_new();
    EVP_CipherInit_ex(ctx, EVP_rc4(), NULL, NULL, NULL, 1);
    EVP_CIPHER_CTX_set_key_length(ctx, KEY_SIZE);
    EVP_CipherInit_ex(ctx, NULL, NULL, key, NULL, 1);
    for (unsigned int i = 0; i < N_ROW; i++) {
        EVP_CipherUpdate(ctx, golden[1] + i, &outlen, (const unsigned char*)datain, 1);
        i++;
        EVP_CipherUpdate(ctx, golden[1] + i, &outlen, (const unsigned char*)datain2, 1);
    }
    EVP_CIPHER_CTX_free(ctx);

    outlen = 0;
    ctx = EVP_CIPHER_CTX_new();
    EVP_CipherInit_ex(ctx, EVP_rc4(), NULL, NULL, NULL, 1);
    EVP_CIPHER_CTX_set_key_length(ctx, KEY_SIZE);
    EVP_CipherInit_ex(ctx, NULL, NULL, key, NULL, 1);
    for (unsigned int i = 0; i < N_ROW; i++) {
        EVP_CipherUpdate(ctx, golden[2] + i, &outlen, (const unsigned char*)datain, 1);
        i++;
        EVP_CipherUpdate(ctx, golden[2] + i, &outlen, (const unsigned char*)datain2, 1);
    }
    EVP_CIPHER_CTX_free(ctx);

    outlen = 0;
    ctx = EVP_CIPHER_CTX_new();
    EVP_CipherInit_ex(ctx, EVP_rc4(), NULL, NULL, NULL, 1);
    EVP_CIPHER_CTX_set_key_length(ctx, KEY_SIZE);
    EVP_CipherInit_ex(ctx, NULL, NULL, key, NULL, 1);
    for (unsigned int i = 0; i < N_ROW; i++) {
        EVP_CipherUpdate(ctx, golden[3] + i, &outlen, (const unsigned char*)datain, 1);
        i++;
        EVP_CipherUpdate(ctx, golden[3] + i, &outlen, (const unsigned char*)datain2, 1);
    }
    EVP_CIPHER_CTX_free(ctx);

    ap_uint<512> keyBlock[4];
    for (unsigned int i = 0; i < KEY_SIZE; i++) {
        keyBlock[i / 64].range((i % 64) * 8 + 7, (i % 64) * 8) = key[i];
    }

    ap_uint<512> dataBlock;
    for (unsigned int i = 0; i < 64; i++) {
        if (i % 2 == 0) {
            dataBlock.range(i * 8 + 7, i * 8) = datain[0];
        } else {
            dataBlock.range(i * 8 + 7, i * 8) = datain2[0];
        }
    }

    std::cout << "Goldens have been created using OpenSSL.\n";

    // Host buffers
    ap_uint<512>* hb_in1 = aligned_alloc<ap_uint<512> >(CH_NM * 4 + N_ROW * N_TASK * CH_NM / 64 + 1);
    ap_uint<512>* hb_in2 = aligned_alloc<ap_uint<512> >(CH_NM * 4 + N_ROW * N_TASK * CH_NM / 64 + 1);
    ap_uint<512>* hb_in3 = aligned_alloc<ap_uint<512> >(CH_NM * 4 + N_ROW * N_TASK * CH_NM / 64 + 1);
    ap_uint<512>* hb_in4 = aligned_alloc<ap_uint<512> >(CH_NM * 4 + N_ROW * N_TASK * CH_NM / 64 + 1);
    ap_uint<512>* hb_out_a[4];
    for (int i = 0; i < 4; i++) {
        hb_out_a[i] = aligned_alloc<ap_uint<512> >(N_ROW * N_TASK * CH_NM / 64);
    }
    ap_uint<512>* hb_out_b[4];
    for (int i = 0; i < 4; i++) {
        hb_out_b[i] = aligned_alloc<ap_uint<512> >(N_ROW * N_TASK * CH_NM / 64);
    }

    // generate configuration block
    hb_in1[0].range(127, 0) = N_ROW;
    hb_in1[0].range(191, 128) = N_TASK;
    hb_in1[0].range(207, 192) = KEY_SIZE;
    hb_in2[0].range(127, 0) = N_ROW;
    hb_in2[0].range(191, 128) = N_TASK;
    hb_in2[0].range(207, 192) = KEY_SIZE;
    hb_in3[0].range(127, 0) = N_ROW;
    hb_in3[0].range(191, 128) = N_TASK;
    hb_in3[0].range(207, 192) = KEY_SIZE;
    hb_in4[0].range(127, 0) = N_ROW;
    hb_in4[0].range(191, 128) = N_TASK;
    hb_in4[0].range(207, 192) = KEY_SIZE;
    // generate key blocks
    for (unsigned int j = 0; j < CH_NM * 4; j++) {
        hb_in1[j + 1] = keyBlock[j % 4];
        hb_in2[j + 1] = keyBlock[j % 4];
        hb_in3[j + 1] = keyBlock[j % 4];
        hb_in4[j + 1] = keyBlock[j % 4];
    }
    // generate texts
    for (unsigned int j = 0; j < N_ROW * N_TASK * CH_NM / 64; j++) {
        hb_in1[j + 1 + 4 * CH_NM] = dataBlock;
        hb_in2[j + 1 + 4 * CH_NM] = dataBlock;
        hb_in3[j + 1 + 4 * CH_NM] = dataBlock;
        hb_in4[j + 1 + 4 * CH_NM] = dataBlock;
    }

    std::cout << "Host map buffer has been allocated and set.\n";

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

    cl::Kernel kernel0(program, "rc4EncryptKernel_1", &err);
    logger.logCreateKernel(err);
    cl::Kernel kernel1(program, "rc4EncryptKernel_2", &err);
    logger.logCreateKernel(err);
    cl::Kernel kernel2(program, "rc4EncryptKernel_3", &err);
    logger.logCreateKernel(err);
    cl::Kernel kernel3(program, "rc4EncryptKernel_4", &err);
    logger.logCreateKernel(err);

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
        in_buff_a[i] =
            cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                       (size_t)(sizeof(ap_uint<512>) * (1 + CH_NM * 4 + N_ROW * N_TASK * CH_NM / 64)), &mext_in[i]);
        out_buff_a[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                   (size_t)(sizeof(ap_uint<512>) * (N_ROW * N_TASK * CH_NM / 64)), &mext_out_a[i]);
        in_buff_b[i] =
            cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                       (size_t)(sizeof(ap_uint<512>) * (1 + CH_NM * 4 + N_ROW * N_TASK * CH_NM / 64)), &mext_in[i]);
        out_buff_b[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                   (size_t)(sizeof(ap_uint<512>) * (N_ROW * N_TASK * CH_NM / 64)), &mext_out_b[i]);
    }

    std::cout << "DDR buffers have been mapped/copy-and-mapped\n";

    q.finish();

    struct timeval start_time, end_time;
    gettimeofday(&start_time, 0);

    std::vector<std::vector<cl::Event> > write_events(num_rep);
    std::vector<std::vector<cl::Event> > kernel_events(num_rep);
    std::vector<std::vector<cl::Event> > read_events(num_rep);
    for (int i = 0; i < num_rep; i++) {
        write_events[i].resize(1);
        kernel_events[i].resize(4);
        read_events[i].resize(1);
    }

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
    }

    // wait all to finish
    q.flush();
    q.finish();
    gettimeofday(&end_time, 0);
    std::cout << "Kernel has been run for " << std::dec << num_rep << " times." << std::endl;
    std::cout << "Execution time " << tvdiff(&start_time, &end_time) << "us" << std::endl;

    // check result
    bool checked = true;
    // check ping buffer
    for (unsigned int n = 0; n < 4; n++) {
        for (unsigned int j = 0; j < N_TASK; j++) {
            for (unsigned int k = 0; k < CH_NM; k++) {
                for (unsigned int i = 0; i < N_ROW; i++) {
                    if (hb_out_a[n][j * ((N_ROW / 32) * (CH_NM / 2)) + (i / 32) * (CH_NM / 2) + k / 2].range(
                            (k % 2) * 256 + (i % 32) * 8 + 7, (k % 2) * 256 + (i % 32) * 8) != golden[n][i]) {
                        checked = false;
                        std::cout << "Error found in kernel_ " << std::dec << n << " " << k << " channel, " << j
                                  << " task, " << i << " message" << std::endl;
                        std::cout << "golden[n] = " << std::hex << (int)golden[n][i] << std::endl;
                        std::cout << "fpga   = " << std::hex
                                  << hb_out_a[n][j * ((N_ROW / 32) * (CH_NM / 2)) + (i / 32) * (CH_NM / 2) + k / 2]
                                         .range((k % 2) * 256 + (i % 32) * 8 + 7, (k % 2) * 256 + (i % 32) * 8)
                                  << std::endl;
                    }
                }
            }
        }
    }

    // check pong buffer
    for (unsigned int n = 0; n < 4; n++) {
        for (unsigned int j = 0; j < N_TASK; j++) {
            for (unsigned int k = 0; k < CH_NM; k++) {
                for (unsigned int i = 0; i < N_ROW; i++) {
                    if (hb_out_b[n][j * ((N_ROW / 32) * (CH_NM / 2)) + (i / 32) * (CH_NM / 2) + k / 2].range(
                            (k % 2) * 256 + (i % 32) * 8 + 7, (k % 2) * 256 + (i % 32) * 8) != golden[n][i]) {
                        checked = false;
                        std::cout << "Error found in kernel_ " << std::dec << n << " " << k << " channel, " << j
                                  << " task, " << i << " message" << std::endl;
                        std::cout << "golden[n] = " << std::hex << (int)golden[n][i] << std::endl;
                        std::cout << "fpga   = " << std::hex
                                  << hb_out_b[n][j * ((N_ROW / 32) * (CH_NM / 2)) + (i / 32) * (CH_NM / 2) + k / 2]
                                         .range((k % 2) * 256 + (i % 32) * 8 + 7, (k % 2) * 256 + (i % 32) * 8)
                                  << std::endl;
                    }
                }
            }
        }
    }

    if (checked) {
        std::cout << std::dec << CH_NM << " channels, " << N_TASK << " tasks, " << N_ROW
                  << " messages verified. No error found!" << std::endl;
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
        return 0;
    } else {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
        return 1;
    }
}
