/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
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
 *
 */
#include "gzip_compress_streaming.hpp"

int validate(std::string& inFile_name, std::string& outFile_name) {
    std::string command = "cmp " + inFile_name + " " + outFile_name;
    int ret = system(command.c_str());
    return ret;
}

void error_message(const std::string& val) {
    std::cout << "\n";
    std::cout << "Please provide " << val << " option" << std::endl;
    std::cout << "Exiting Application" << std::endl;
    std::cout << "\n";
}

int xfGzip::init(const std::string& binaryFile) {
    cl_int err;
    // The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[m_deviceid];

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, m_context = new cl::Context(device, NULL, NULL, NULL, &err));

    OCL_CHECK(err, m_q_cmp = new cl::CommandQueue(
                       *m_context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, m_q_dm = new cl::CommandQueue(
                       *m_context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err));

    // import_binary() command will find the OpenCL binary file created using the
    // v++ compiler load into OpenCL Binary and return as Binaries
    // OpenCL and it can contain many functions which can be executed on the
    // device.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    m_program = new cl::Program(*m_context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device[" << m_deviceid << "] with xclbin file!\n";
    } else {
// Create Compress kernels
#ifdef DYNAMIC_MODE
        OCL_CHECK(err, m_compress_kernel = new cl::Kernel(*m_program, compress_kernel_names[1].c_str(), &err));
#else
        OCL_CHECK(err, m_compress_kernel = new cl::Kernel(*m_program, compress_kernel_names[0].c_str(), &err));
#endif
        OCL_CHECK(err, m_datamover_kernel = new cl::Kernel(*m_program, datamover_kernel_names[0].c_str(), &err));
    }
    return 0;
}
// Constructor
xfGzip::xfGzip(const std::string& binaryFile, uint32_t block_size_kb, uint8_t device_id) {
    // store class variables
    m_BlockSizeInKb = block_size_kb;
    m_deviceid = device_id;

    // create context, command queues and kernel objects
    init(binaryFile);

    // resize class internal buffers to standard size
    h_buf_in.resize(HOST_BUFFER_SIZE);
    h_buf_out.resize(HOST_BUFFER_SIZE);
    h_compressSize.resize(2);
}

// Destructor
xfGzip::~xfGzip() {
    delete (m_compress_kernel);
    delete (m_datamover_kernel);
    delete (m_program);
    delete (m_q_cmp);
    delete (m_q_dm);
    delete (m_context);
}

uint64_t xfGzip::compressFile(std::string& inFile_name, std::string& outFile_name, uint64_t input_size) {
    std::ifstream inFile(inFile_name.c_str(), std::ifstream::binary);
    std::ofstream outFile(outFile_name.c_str(), std::ofstream::binary);

    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1);
    }

    std::vector<uint8_t, aligned_allocator<uint8_t> > in(input_size);
    std::vector<uint8_t, aligned_allocator<uint8_t> > out(input_size * 2);

    inFile.read((char*)in.data(), input_size);

    uint32_t host_buffer_size = HOST_BUFFER_SIZE;

    uint64_t enbytes;

    // Zlib multiple/single cu sequential version
    // std::cout << "Start compress " << inFile_name << std::endl;
    enbytes = compressSequential(in.data(), out.data(), input_size, host_buffer_size);
    // Writing compressed data
    outFile.write((char*)out.data(), enbytes);
    // std::cout << "Done " << std::endl;

    // Close file
    inFile.close();
    outFile.close();
    return enbytes;
}

// Note: Various block sizes supported by LZ4 standard are not applicable to
// this function. It just supports Block Size 64KB
uint64_t xfGzip::compressSequential(uint8_t* in, uint8_t* out, uint64_t input_size, uint32_t host_buffer_size) {
    // uint32_t block_size_in_bytes = m_BlockSizeInKb * 1024;
    // uint32_t max_num_blks = host_buffer_size / block_size_in_bytes;
    cl_int err;
    uint32_t outSize = input_size * 2;
    h_buf_in.resize(input_size);
    h_buf_out.resize(outSize);
    h_compressSize.resize(2);

    std::chrono::duration<double, std::nano> kernel_time_ns_1(0);

    cl::Buffer* buffer_input;
    cl::Buffer* buffer_output;
    cl::Buffer* buffer_size;
    // Allocate Buffer in Global Memory
    OCL_CHECK(err, buffer_input = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR, input_size, h_buf_in.data(), &err));
    OCL_CHECK(err, buffer_output = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR, outSize, h_buf_out.data(), &err));
    OCL_CHECK(err,
              buffer_size = new cl::Buffer(*m_context, CL_MEM_USE_HOST_PTR, sizeof(int), h_compressSize.data(), &err));

    // transfer input data
    std::memcpy(h_buf_in.data(), in, input_size);

    // Set the Kernel Arguments
    OCL_CHECK(err, err = m_datamover_kernel->setArg(0, *buffer_input));
    OCL_CHECK(err, err = m_datamover_kernel->setArg(2, (uint32_t)input_size));
    OCL_CHECK(err, err = m_datamover_kernel->setArg(1, *buffer_output));
    OCL_CHECK(err, err = m_datamover_kernel->setArg(3, *buffer_size));

    OCL_CHECK(err, err = m_q_dm->enqueueMigrateMemObjects({*buffer_input}, 0));
    OCL_CHECK(err, err = m_q_dm->finish());

#ifndef DISABLE_FREE_RUNNING_KERNEL // free running flow
    // enqueue kernels
    auto decompress_API_start = std::chrono::high_resolution_clock::now();
    m_q_dm->enqueueTask(*m_datamover_kernel);
    // wait for kernels to finish
    OCL_CHECK(err, err = m_q_dm->finish());
    auto decompress_API_end = std::chrono::high_resolution_clock::now();

#else // regular running flow
    // enqueue kernels
    m_q_dm->enqueueTask(*m_datamover_kernel);
    // sleep(1);
    auto decompress_API_start = std::chrono::high_resolution_clock::now();
    m_q_cmp->enqueueTask(*m_compress_kernel);
    // wait for kernels to finish
    OCL_CHECK(err, err = m_q_cmp->finish());
    auto decompress_API_end = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = m_q_dm->finish());
#endif

    // fetch output data
    OCL_CHECK(err, err = m_q_dm->enqueueMigrateMemObjects({*buffer_output, *buffer_size}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = m_q_dm->finish());

    // calculate duration for kernel execution
    auto duration = std::chrono::duration<double, std::nano>(decompress_API_end - decompress_API_start);
    kernel_time_ns_1 += duration;

    // copy output compressed data
    uint32_t cmpSize = h_compressSize[0];
    std::memcpy(out, h_buf_out.data(), cmpSize);

    // free CL buffers
    delete (buffer_input);
    delete (buffer_output);
    delete (buffer_size);

    float throughput_in_mbps_1 = (float)input_size * 1000 / kernel_time_ns_1.count();
    std::cout << std::fixed << std::setprecision(2) << throughput_in_mbps_1;
    return cmpSize;
}
