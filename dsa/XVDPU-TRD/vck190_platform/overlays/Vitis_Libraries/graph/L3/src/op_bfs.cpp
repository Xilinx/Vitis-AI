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

#pragma once

#ifndef _XF_GRAPH_L3_OP_BFS_CPP_
#define _XF_GRAPH_L3_OP_BFS_CPP_

#include "op_bfs.hpp"
#include "xf_utils_sw/logger.hpp"

namespace xf {
namespace graph {
namespace L3 {

void createHandleBFS(clHandle& handle, const char* kernelName, const char* pXclbin, int32_t IDDevice) {
    // Platform related operations
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int err;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    handle.device = devices[IDDevice];
    handle.context = cl::Context(handle.device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    handle.q = cl::CommandQueue(handle.context, handle.device,
                                CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    logger.logCreateCommandQueue(err);
    std::string devName = handle.device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    handle.xclBins = xcl::import_binary_file(pXclbin);
    std::vector<cl::Device> devices2;
    devices2.push_back(handle.device);
    handle.program = cl::Program(handle.context, devices2, handle.xclBins, NULL, &err);
    logger.logCreateProgram(err);
}

uint32_t opBFS::cuPerBoardBFS;

uint32_t opBFS::dupNmBFS;

void opBFS::setHWInfo(uint32_t numDev, uint32_t CUmax) {
    maxCU = CUmax;
    deviceNm = numDev;
    cuPerBoardBFS = maxCU / deviceNm;
    handles = new clHandle[CUmax];
};

void opBFS::freeBFS() {
    for (int i = 0; i < maxCU; ++i) {
        delete[] handles[i].buffer;
    }
    delete[] handles;
};

void opBFS::cuRelease(xrmContext* ctx, xrmCuResource* resR) {
    while (!xrmCuRelease(ctx, resR)) {
    };
    free(resR);
};

void opBFS::init(char* kernelName, char* xclbinFile, uint32_t* deviceIDs, uint32_t* cuIDs, unsigned int requestLoad) {
    dupNmBFS = 100 / requestLoad;
    cuPerBoardBFS /= dupNmBFS;
    uint32_t bufferNm = 7;
    unsigned int cnt = 0;
    unsigned int cntCU = 0;
    unsigned int* handleID = new unsigned int[maxCU];
    handleID[0] = cnt;
    handles[0].deviceID = deviceIDs[0];
    handles[0].cuID = cuIDs[0];
    handles[0].dupID = 0;
    std::thread th[maxCU];
    createHandleBFS(handles[cnt], kernelName, xclbinFile, deviceIDs[cnt]);
    handles[cnt].buffer = new cl::Buffer[bufferNm];
    unsigned int prev = deviceIDs[0];
    unsigned int prevCU = cuIDs[0];
    deviceOffset.push_back(0);
    for (int i = 1; i < maxCU; ++i) {
        handles[i].deviceID = deviceIDs[i];
        handles[i].cuID = cuIDs[i];
        handles[i].dupID = i % dupNmBFS;
        // th[i] = std::thread(&createHandleBFS, std::ref(handles[i]), kernelName, xclbinFile, deviceIDs[i]);
        createHandleBFS(handles[i], kernelName, xclbinFile, deviceIDs[i]);
        handles[i].buffer = new cl::Buffer[bufferNm];
        if (deviceIDs[i] != prev) {
            prev = deviceIDs[i];
            deviceOffset.push_back(i);
        }
    }
    delete[] handleID;
}

void opBFS::migrateMemObj(clHandle* hds,
                          bool type,
                          unsigned int num_runs,
                          std::vector<cl::Memory>& ob,
                          std::vector<cl::Event>* evIn,
                          cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueMigrateMemObjects(ob, type, evIn, evOut); // 0 : migrate from host to dev
    }
};

void opBFS::bufferInit(clHandle* hds,
                       std::string instanceName0,
                       uint32_t sourceID,
                       xf::graph::Graph<uint32_t, uint32_t> g,
                       uint32_t* queue,
                       uint32_t* discovery,
                       uint32_t* finish,
                       uint32_t* predecent,
                       uint32_t* distance,
                       cl::Kernel& kernel0,
                       std::vector<cl::Memory>& ob_in,
                       std::vector<cl::Memory>& ob_out) {
    cl::Device device = hds[0].device;
    const char* instanceName = instanceName0.c_str();
    // Creating Context and Command Queue for selected Device
    cl::Context context = hds[0].context;
    cl::CommandQueue q = hds[0].q;
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    std::vector<cl::Device> devices;
    devices.push_back(hds[0].device);
    cl::Program program = hds[0].program;
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int err;
    kernel0 = cl::Kernel(program, instanceName, &err);
    logger.logCreateKernel(err);
    std::cout << "INFO: Kernel has been created" << std::endl;

    std::vector<cl_mem_ext_ptr_t> mext_in = std::vector<cl_mem_ext_ptr_t>(7);
    mext_in[0] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, g.offsetsCSR, kernel0()};
    mext_in[1] = {(unsigned int)(2) | XCL_MEM_TOPOLOGY, g.indicesCSR, kernel0()};
    mext_in[2] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, queue, kernel0()};
    mext_in[3] = {(unsigned int)(6) | XCL_MEM_TOPOLOGY, discovery, kernel0()};
    mext_in[4] = {(unsigned int)(8) | XCL_MEM_TOPOLOGY, finish, kernel0()};
    mext_in[5] = {(unsigned int)(9) | XCL_MEM_TOPOLOGY, predecent, kernel0()};
    mext_in[6] = {(unsigned int)(10) | XCL_MEM_TOPOLOGY, distance, kernel0()};

    uint32_t numVertices = g.nodeNum;
    uint32_t numEdges = g.edgeNum;
    // create device buffer and map dev buf to host buf
    hds[0].buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * (numVertices + 1), &mext_in[0]);

    hds[0].buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * numEdges, &mext_in[1]);

    hds[0].buffer[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * numVertices, &mext_in[2]);

    hds[0].buffer[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * ((numVertices + 15) / 16) * 16, &mext_in[3]);

    hds[0].buffer[4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * ((numVertices + 15) / 16) * 16, &mext_in[4]);

    hds[0].buffer[5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * ((numVertices + 15) / 16) * 16, &mext_in[5]);

    hds[0].buffer[6] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * ((numVertices + 15) / 16) * 16, &mext_in[6]);

    // add buffers to migrate
    ob_in.push_back(hds[0].buffer[0]);
    ob_in.push_back(hds[0].buffer[1]);
    ob_out.push_back(hds[0].buffer[5]);
    ob_out.push_back(hds[0].buffer[6]);

    kernel0.setArg(0, sourceID);          // source ID
    kernel0.setArg(1, g.nodeNum);         // node number
    kernel0.setArg(2, hds[0].buffer[1]);  // indices
    kernel0.setArg(3, hds[0].buffer[0]);  // offsets
    kernel0.setArg(4, hds[0].buffer[2]);  // queue
    kernel0.setArg(5, hds[0].buffer[2]);  // queue
    kernel0.setArg(6, hds[0].buffer[3]);  // discovery
    kernel0.setArg(7, hds[0].buffer[3]);  // discovery
    kernel0.setArg(8, hds[0].buffer[4]);  // finish
    kernel0.setArg(9, hds[0].buffer[5]);  // predecent
    kernel0.setArg(10, hds[0].buffer[6]); // distance
};

int opBFS::cuExecute(
    clHandle* hds, cl::Kernel& kernel0, unsigned int num_runs, std::vector<cl::Event>* evIn, cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueTask(kernel0, evIn, evOut);
    }
    return 0;
}

int opBFS::compute(unsigned int deviceID,
                   unsigned int cuID,
                   unsigned int channelID,
                   xrmContext* ctx,
                   xrmCuResource* resR,
                   std::string instanceName,
                   clHandle* handles,
                   uint32_t sourceID,
                   xf::graph::Graph<uint32_t, uint32_t> g,
                   uint32_t* predecent,
                   uint32_t* distance) {
    clHandle* hds = &handles[channelID + cuID * dupNmBFS + deviceID * dupNmBFS * cuPerBoardBFS];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    unsigned int num_runs = 1;
    uint32_t numVertices = g.nodeNum;

    uint32_t* queue = aligned_alloc<uint32_t>(numVertices);
    uint32_t* discovery = aligned_alloc<uint32_t>(((numVertices + 15) / 16) * 16);
    uint32_t* finish = aligned_alloc<uint32_t>(((numVertices + 15) / 16) * 16);

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, sourceID, g, queue, discovery, finish, predecent, distance, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    cuRelease(ctx, resR);

    free(queue);
    free(discovery);
    free(finish);

    return ret;
};

event<int> opBFS::addwork(uint32_t sourceID,
                          xf::graph::Graph<uint32_t, uint32_t> g,
                          uint32_t* predecent,
                          uint32_t* distance) {
    return createL3(task_queue[0], &(compute), handles, sourceID, g, predecent, distance);
};

} // L3
} // graph
} // xf
#endif
