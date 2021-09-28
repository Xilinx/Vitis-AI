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
#ifndef XF_HPC_FPGA_HPP
#define XF_HPC_FPGA_HPP
#include <iostream>
#include <vector>
#include <regex>
#include <unordered_map>

// This extension file is required for stream APIs
#include "CL/cl_ext_xilinx.h"
// This file is required for OpenCL C++ wrapper APIs
#include "xcl2.hpp"
using namespace std;

template <typename T>
using host_buffer_t = std::vector<T, aligned_allocator<T> >;

class FPGA {
   public:
    FPGA(string deviceName) {
        getDevices(deviceName);
        m_device = m_Devices[m_id];
        m_id = -1;
    }
    FPGA(unsigned int p_id = 0, string deviceName = "") {
        getDevices(deviceName);
        setID(p_id);
    }

    FPGA* next() const {
        if (m_id == m_Devices.size() - 1) {
            return nullptr;
        }
        FPGA* ptr = new FPGA(m_id + 1, m_Devices);
        return ptr;
    }

    void setID(uint32_t id) {
        m_id = id;
        if (m_id >= m_Devices.size()) {
            cout << "Device specified by id = " << m_id << " is not found." << endl;
            throw;
        }
        m_device = m_Devices[m_id];
    }

    bool xclbin(string binaryFile) {
        cl_int err;
        // get_xil_devices() is a utility API which will find the xilinx
        // platforms and will return list of devices connected to Xilinx platform

        // Creating Context
        OCL_CHECK(err, m_context = cl::Context(m_device, NULL, NULL, NULL, &err));

        // Creating Command Queue
        OCL_CHECK(err,
                  m_queue = cl::CommandQueue(m_context, m_device,
                                             CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));
        // read_binary_file() is a utility API which will load the binaryFile
        // and will return the pointer to file buffer.
        cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

        // Creating Program
        OCL_CHECK(err, m_program = cl::Program(m_context, {m_device}, bins, NULL, &err));
        return true;
    }
    const cl::Context& getContext() const { return m_context; }
    const cl::CommandQueue& getCommandQueue() const { return m_queue; }
    cl::CommandQueue& getCommandQueue() { return m_queue; }

    const cl::Program& getProgram() const { return m_program; }

    void finish() const { m_queue.finish(); }

    template <typename T>
    vector<cl::Buffer> createDeviceBuffer(cl_mem_flags p_flags, const vector<host_buffer_t<T> >& p_buffer) {
        size_t p_hbm_pc = p_buffer.size();
        vector<cl::Buffer> l_buffer(p_hbm_pc);
        for (int i = 0; i < p_hbm_pc; i++) {
            l_buffer[i] = createDeviceBuffer(p_flags, p_buffer[i]);
        }
        return l_buffer;
    }

    template <typename T>
    cl::Buffer createDeviceBuffer(cl_mem_flags p_flags, const host_buffer_t<T>& p_buffer) {
        const void* l_ptr = (const void*)p_buffer.data();
        if (exists(l_ptr)) return m_bufferMaps[l_ptr];

        size_t l_bufferSize = sizeof(T) * p_buffer.size();
        cl_int err;
        m_bufferMaps.insert(
            {l_ptr, cl::Buffer(m_context, p_flags | CL_MEM_USE_HOST_PTR, l_bufferSize, (void*)p_buffer.data(), &err)});
        if (err != CL_SUCCESS) {
            printf("Failed to allocate device buffer!\n");
            throw std::bad_alloc();
        }
        return m_bufferMaps[l_ptr];
    }

   protected:
    bool exists(const void* p_ptr) const {
        auto it = m_bufferMaps.find(p_ptr);
        return it != m_bufferMaps.end();
    }
    void getDevices(string deviceName) {
        cl_int err;
        auto devices = xcl::get_xil_devices();
        auto regexStr = regex(".*" + deviceName + ".*");
        for (auto device : devices) {
            string cl_device_name;
            OCL_CHECK(err, err = device.getInfo(CL_DEVICE_NAME, &cl_device_name));
            if (regex_match(cl_device_name, regexStr)) m_Devices.push_back(device);
        }
        if (0 == m_Devices.size()) {
            cout << "Device specified by name == " << deviceName << " is not found." << endl;
            throw;
        }
    }

    FPGA(unsigned int p_id, const vector<cl::Device>& devices) {
        m_id = p_id;
        m_Devices = devices;
        m_device = m_Devices[m_id];
    }

   private:
    unsigned int m_id;
    cl::Device m_device;
    vector<cl::Device> m_Devices;
    cl::Context m_context;
    cl::CommandQueue m_queue;
    cl::Program m_program;
    unordered_map<const void*, cl::Buffer> m_bufferMaps;
};

class Kernel {
   public:
    Kernel(FPGA* p_fpga = nullptr) : m_fpga(p_fpga) {}

    void fpga(FPGA* p_fpga) { m_fpga = p_fpga; }

    void getCU(const string& p_name) {
        cl_int err;
        OCL_CHECK(err, m_kernel = cl::Kernel(m_fpga->getProgram(), p_name.c_str(), &err));
    }

    const cl::Kernel& operator()() const { return m_kernel; }
    cl::Kernel& operator()() { return m_kernel; }

    void enqueueTask() const {
        cl_int err;
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueTask(m_kernel));
    }

    void finish() const { m_fpga->finish(); }

    static double run(const vector<Kernel>& p_kernels) {
        auto start = chrono::high_resolution_clock::now();
        for (auto ker : p_kernels) ker.enqueueTask();
        for (auto ker : p_kernels) ker.finish();
        auto finish = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = finish - start;
        double t_sec = elapsed.count();
        return t_sec;
    }

    void getBuffer(vector<cl::Memory>& h_m) {
        cl_int err;
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(h_m, CL_MIGRATE_MEM_OBJECT_HOST));
        finish();
    }

    void sendBuffer(vector<cl::Memory>& h_m) {
        cl_int err;
        OCL_CHECK(err, err = m_fpga->getCommandQueue().enqueueMigrateMemObjects(h_m, 0)); /* 0 means from host*/
        finish();
    }

    template <typename T>
    cl::Buffer createDeviceBuffer(cl_mem_flags p_flags, const host_buffer_t<T>& p_buffer) const {
        return m_fpga->createDeviceBuffer(p_flags, p_buffer);
    }

    template <typename T>
    vector<cl::Buffer> createDeviceBuffer(cl_mem_flags p_flags, const vector<host_buffer_t<T> >& p_buffer) const {
        return m_fpga->createDeviceBuffer(p_flags, p_buffer);
    }

    FPGA* m_fpga;
    cl::Kernel m_kernel;
};

#endif
