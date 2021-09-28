/*
 * Copyright 2019-2021 Xilinx, Inc.
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
#include <cassert>
#include <vector>
#include "lz4_p2p.hpp"
#include "lz4_p2p_dec.hpp"
#include <cstdio>
#include <fstream>
#include <iosfwd>
#include "CL/cl.h"

using std::ifstream;
using std::ios;
using std::streamsize;
int fd_p2p_c_out = 0;
int fd_p2p_c_in = 0;

std::vector<unsigned char> readBinary(const std::string& fileName) {
    ifstream file(fileName, ios::binary | ios::ate);
    if (file) {
        file.seekg(0, ios::end);
        streamsize size = file.tellg();
        file.seekg(0, ios::beg);
        std::vector<unsigned char> buffer(size);
        file.read((char*)buffer.data(), size);
        return buffer;
    } else {
        return std::vector<unsigned char>(0);
    }
}

// Constructor
xfLz4::xfLz4(const std::string& binaryFile) {
    int err;
    cl_int error;
    m_device = 0;
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id* platform_ids = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platform_ids, NULL);
    size_t i;

    for (i = 0; i < num_platforms; i++) {
        size_t platform_name_size;
        err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 0, NULL, &platform_name_size);
        if (err != CL_SUCCESS) {
            printf("Error: Could not determine platform name!\n");
            exit(EXIT_FAILURE);
        }

        char* platform_name = (char*)malloc(sizeof(char) * platform_name_size);
        if (platform_name == NULL) {
            printf("Error: out of memory!\n");
            exit(EXIT_FAILURE);
        }

        err = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, platform_name_size, platform_name, NULL);
        if (err != CL_SUCCESS) {
            printf("Error: could not determine platform name!\n");
            exit(EXIT_FAILURE);
        }

        if (!strcmp(platform_name, "Xilinx")) {
            free(platform_name);
            platform = platform_ids[i];
            break;
        }

        free(platform_name);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &m_device, NULL);
    cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

    m_context = clCreateContext(properties, 1, &m_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) std::cout << "clCreateContext call: Failed to create a compute context" << err << std::endl;

    std::vector<unsigned char> binary = readBinary(binaryFile);
    size_t binary_size = binary.size();
    const unsigned char* binary_data = binary.data();

    m_program = clCreateProgramWithBinary(m_context, 1, &m_device, &binary_size, &binary_data, NULL, &err);

    ooo_q = clCreateCommandQueue(m_context, m_device,
                                 CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &error);
}

// Destructor
xfLz4::~xfLz4() {
    clReleaseCommandQueue(ooo_q);
    clReleaseContext(m_context);
    clReleaseProgram(m_program);
}

void xfLz4::decompress_in_line_multiple_files(const std::vector<std::string>& inFileVec,
                                              std::vector<int>& fd_p2p_vec,
                                              std::vector<char*>& outVec,
                                              std::vector<uint64_t>& orgSizeVec,
                                              std::vector<uint32_t>& inSizeVec,
                                              bool enable_p2p,
                                              uint8_t maxCR) {
    std::vector<cl_kernel> dmKernelVec;
    std::vector<cl_kernel> decompressKernelVec;
    std::vector<cl_mem> bufInputVec;
    std::vector<cl_mem> bufOutVec;
    std::vector<cl_mem> bufOutSizeVec;
    std::vector<uint64_t> inSizeVec4k;
    std::vector<char*> p2pPtrVec;
    std::vector<uint8_t, aligned_allocator<uint8_t> > out;
    std::vector<uint32_t, aligned_allocator<uint32_t> > decSize;

    std::vector<std::vector<uint8_t, aligned_allocator<uint8_t> > > inVec;
    std::vector<std::vector<uint8_t, aligned_allocator<uint8_t> > > outputVec;
    std::vector<std::vector<uint32_t, aligned_allocator<uint32_t> > > outSizeVec;

    uint64_t total_size = 0;
    uint64_t total_in_size = 0;
    std::chrono::duration<double, std::nano> total_ssd_time_ns(0);

    int ret = 0;
    cl_int error;

    cl_mem buffer_input;
    for (uint32_t fid = 0; fid < inFileVec.size(); fid++) {
        std::string dm_kname = datamover_kernel_names[0];
        std::string dec_kname = decompress_kernel_names[0];

        cl_mem_ext_ptr_t p2pBoExt = {0};
        if (enable_p2p) p2pBoExt = {XCL_MEM_EXT_P2P_BUFFER, NULL, 0};

        if ((fid % 2) == 0) {
            dm_kname += ":{xilDecompDatamover_1}";
            dec_kname += ":{xilLz4DecompressStream_1}";
        } else {
            dm_kname += ":{xilDecompDatamover_2}";
            dec_kname += ":{xilLz4DecompressStream_2}";
        }
        uint32_t input_size = inSizeVec[fid];
        uint64_t outputSize = maxCR * input_size;

        out.resize(outputSize);
        decSize.resize(sizeof(uint32_t));

        outputVec.push_back(out);
        outSizeVec.push_back(decSize);

        uint64_t input_size_4k_multiple = ((input_size - 1) / (4096) + 1) * 4096;
        inSizeVec4k.push_back(input_size_4k_multiple);
        total_in_size += input_size;

        if (!enable_p2p) {
            std::vector<uint8_t, aligned_allocator<uint8_t> > in(input_size_4k_multiple);
            inVec.push_back(in);
            read(fd_p2p_vec[fid], inVec[fid].data(), inSizeVec4k[fid]);
        }
        // Allocate BOs.
        if (enable_p2p) {
            buffer_input = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, input_size_4k_multiple,
                                          &p2pBoExt, &error);
        } else {
            buffer_input = clCreateBuffer(m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, input_size_4k_multiple,
                                          inVec[fid].data(), &error);
        }
        if (error)
            std::cout << "P2P: buffer_input creation failed, error: " << error << ", line: " << __LINE__ << std::endl;

        bufInputVec.push_back(buffer_input);
        if (enable_p2p) {
            char* p2pPtr = (char*)clEnqueueMapBuffer(ooo_q, buffer_input, CL_TRUE, CL_MAP_READ, 0,
                                                     input_size_4k_multiple, 0, NULL, NULL, NULL);
            p2pPtrVec.push_back(p2pPtr);
        }

        cl_mem buffer_output = clCreateBuffer(m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, outputSize,
                                              outputVec[fid].data(), &error);

        bufOutVec.push_back(buffer_output);

        cl_mem buffer_outputsize = clCreateBuffer(m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(uint32_t),
                                                  outSizeVec[fid].data(), &error);
        bufOutSizeVec.push_back(buffer_outputsize);

        cl_kernel datamover_kernel = clCreateKernel(m_program, dm_kname.c_str(), &error);
        uint32_t narg = 0;
        clSetKernelArg(datamover_kernel, narg++, sizeof(cl_mem), &bufInputVec[fid]);
        clSetKernelArg(datamover_kernel, narg++, sizeof(cl_mem), &bufOutVec[fid]);
        clSetKernelArg(datamover_kernel, narg++, sizeof(uint32_t), &inSizeVec[fid]);
        clSetKernelArg(datamover_kernel, narg++, sizeof(cl_mem), &bufOutSizeVec[fid]);
        dmKernelVec.push_back(datamover_kernel);

        cl_kernel decompress_kernel = clCreateKernel(m_program, dec_kname.c_str(), &error);
        clSetKernelArg(decompress_kernel, 3, sizeof(uint32_t), &inSizeVec[fid]);
        decompressKernelVec.push_back(decompress_kernel);
    }
    error = clFinish(ooo_q);

    auto total_start = std::chrono::high_resolution_clock::now();
    for (uint32_t fid = 0; fid < inFileVec.size(); fid++) {
        cl_event e_wr;
        if (!enable_p2p) clEnqueueMigrateMemObjects(ooo_q, 1, &bufInputVec[fid], 0, 0, NULL, &e_wr);

        auto ssd_start = std::chrono::high_resolution_clock::now();
        if (enable_p2p) {
            ret = read(fd_p2p_vec[fid], p2pPtrVec[fid], inSizeVec4k[fid]);
            if (ret == -1) std::cout << "P2P: read() failed, err: " << ret << ", line: " << __LINE__ << std::endl;
        }
        auto ssd_end = std::chrono::high_resolution_clock::now();
        auto ssd_time_ns = std::chrono::duration<double, std::nano>(ssd_end - ssd_start);
        total_ssd_time_ns += ssd_time_ns;

        cl_event e_dm;
        if (enable_p2p) {
            error = clEnqueueTask(ooo_q, dmKernelVec[fid], 0, NULL, &e_dm);
        } else {
            error = clEnqueueTask(ooo_q, dmKernelVec[fid], 1, &e_wr, &e_dm);
        }
        error = clEnqueueTask(ooo_q, decompressKernelVec[fid], 0, NULL, NULL);
        error = clEnqueueMigrateMemObjects(ooo_q, 1, &bufOutVec[fid], CL_MIGRATE_MEM_OBJECT_HOST, 1, &e_dm, NULL);
        error = clEnqueueMigrateMemObjects(ooo_q, 1, &bufOutSizeVec[fid], CL_MIGRATE_MEM_OBJECT_HOST, 1, &e_dm, NULL);
        if (!enable_p2p) clReleaseEvent(e_wr);
        clReleaseEvent(e_dm);
    }

    error = clFinish(ooo_q);
    auto total_end = std::chrono::high_resolution_clock::now();

    for (uint32_t fid = 0; fid < inFileVec.size(); fid++) {
        orgSizeVec[fid] = outSizeVec[fid].data()[0];
        total_size += orgSizeVec[fid];
        std::memcpy(outVec[fid], outputVec[fid].data(), orgSizeVec[fid]);
        if (enable_p2p) clEnqueueUnmapMemObject(ooo_q, bufInputVec[fid], p2pPtrVec[fid], 0, NULL, NULL);
        clReleaseKernel(dmKernelVec[fid]);
        clReleaseKernel(decompressKernelVec[fid]);
        clReleaseMemObject(bufOutVec[fid]);
        clReleaseMemObject(bufOutSizeVec[fid]);
    }
    auto total_time_ns = std::chrono::duration<double, std::nano>(total_end - total_start);
    float throughput_in_mbps_1 = (float)total_size * 1000 / total_time_ns.count();
    float ssd_throughput_in_mbps_1 = (float)total_in_size * 1000 / total_ssd_time_ns.count();
    std::cout << std::fixed << std::setprecision(2) << "Throughput\t\t:" << throughput_in_mbps_1 << std::endl;
    if (enable_p2p) std::cout << "SSD Throughput\t\t:" << ssd_throughput_in_mbps_1 << std::endl;
    std::cout << "InputSize(inMB)\t\t:" << total_in_size / 1000000 << std::endl
              << "outputSize(inMB)\t:" << total_size / 1000000 << std::endl
              << "CR\t\t\t:" << ((float)total_size / total_in_size) << std::endl;
}
