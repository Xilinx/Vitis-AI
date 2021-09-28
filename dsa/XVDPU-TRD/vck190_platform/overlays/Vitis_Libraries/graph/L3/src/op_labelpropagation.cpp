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

#ifndef _XF_GRAPH_L3_OP_LABELPROPAGATION_CPP_
#define _XF_GRAPH_L3_OP_LABELPROPAGATION_CPP_

#include "op_labelpropagation.hpp"
#include "xf_utils_sw/logger.hpp"

namespace xf {
namespace graph {
namespace L3 {

void createHandleLabelPropagation(clHandle& handle, const char* kernelName, const char* pXclbin, int32_t IDDevice) {
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

uint32_t opLabelPropagation::cuPerBoardLabelPropagation;

uint32_t opLabelPropagation::dupNmLabelPropagation;

void opLabelPropagation::setHWInfo(uint32_t numDev, uint32_t CUmax) {
    maxCU = CUmax;
    deviceNm = numDev;
    cuPerBoardLabelPropagation = maxCU / deviceNm;
    handles = new clHandle[CUmax];
};

void opLabelPropagation::freeLabelPropagation() {
    for (int i = 0; i < maxCU; ++i) {
        delete[] handles[i].buffer;
    }
    delete[] handles;
};

void opLabelPropagation::cuRelease(xrmContext* ctx, xrmCuResource* resR) {
    while (!xrmCuRelease(ctx, resR)) {
    };
    free(resR);
};

void opLabelPropagation::init(
    char* kernelName, char* xclbinFile, uint32_t* deviceIDs, uint32_t* cuIDs, unsigned int requestLoad) {
    dupNmLabelPropagation = 100 / requestLoad;
    cuPerBoardLabelPropagation /= dupNmLabelPropagation;
    uint32_t bufferNm = 8;
    unsigned int cnt = 0;
    unsigned int cntCU = 0;
    unsigned int* handleID = new unsigned int[maxCU];
    handleID[0] = cnt;
    handles[0].deviceID = deviceIDs[0];
    handles[0].cuID = cuIDs[0];
    handles[0].dupID = 0;
    std::thread th[maxCU];
    // th[0] = std::thread(&createHandleLabelPropagation, std::ref(handles[cnt]), kernelName, xclbinFile,
    // deviceIDs[cnt]);
    createHandleLabelPropagation(handles[cnt], kernelName, xclbinFile, deviceIDs[cnt]);
    handles[cnt].buffer = new cl::Buffer[bufferNm];
    unsigned int prev = deviceIDs[0];
    unsigned int prevCU = cuIDs[0];
    deviceOffset.push_back(0);
    for (int i = 1; i < maxCU; ++i) {
        handles[i].deviceID = deviceIDs[i];
        handles[i].cuID = cuIDs[i];
        handles[i].dupID = i % dupNmLabelPropagation;
        // th[i] = std::thread(&createHandleLabelPropagation, std::ref(handles[i]), kernelName, xclbinFile,
        // deviceIDs[i]);
        createHandleLabelPropagation(handles[i], kernelName, xclbinFile, deviceIDs[i]);
        handles[i].buffer = new cl::Buffer[bufferNm];
        if (deviceIDs[i] != prev) {
            prev = deviceIDs[i];
            deviceOffset.push_back(i);
        }
    }
    delete[] handleID;
}

void opLabelPropagation::migrateMemObj(clHandle* hds,
                                       bool type,
                                       unsigned int num_runs,
                                       std::vector<cl::Memory>& ob,
                                       std::vector<cl::Event>* evIn,
                                       cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueMigrateMemObjects(ob, type, evIn, evOut); // 0 : migrate from host to dev
    }
};

void opLabelPropagation::bufferInit(clHandle* hds,
                                    std::string instanceName0,
                                    uint32_t maxIter,
                                    uint32_t* labelPing,
                                    uint32_t* labelPong,
                                    uint32_t* buffPing,
                                    uint32_t* buffPong,
                                    xf::graph::Graph<uint32_t, uint32_t> g,
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

    uint32_t V = 800000 * 16;
    uint32_t E = 800000 * 16;

    std::vector<cl_mem_ext_ptr_t> mext_in = std::vector<cl_mem_ext_ptr_t>(8);
    mext_in[0] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, g.offsetsCSR, kernel0()};
    mext_in[1] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, g.indicesCSR, kernel0()};
    mext_in[2] = {(unsigned int)(5) | XCL_MEM_TOPOLOGY, g.offsetsCSC, kernel0()};
    mext_in[3] = {(unsigned int)(6) | XCL_MEM_TOPOLOGY, g.indicesCSC, kernel0()};
    mext_in[4] = {(unsigned int)(8) | XCL_MEM_TOPOLOGY, buffPing, kernel0()};
    mext_in[5] = {(unsigned int)(9) | XCL_MEM_TOPOLOGY, buffPong, kernel0()};
    mext_in[6] = {(unsigned int)(10) | XCL_MEM_TOPOLOGY, labelPing, kernel0()};
    mext_in[7] = {(unsigned int)(11) | XCL_MEM_TOPOLOGY, labelPong, kernel0()};

    // Create device buffer and map dev buf to host buf
    hds[0].buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * V, &mext_in[0]);
    hds[0].buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * E, &mext_in[1]);
    hds[0].buffer[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * V, &mext_in[2]);
    hds[0].buffer[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * E, &mext_in[3]);
    hds[0].buffer[4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * V, &mext_in[4]);
    hds[0].buffer[5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * V, &mext_in[5]);
    hds[0].buffer[6] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * V, &mext_in[6]);
    hds[0].buffer[7] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * V, &mext_in[7]);

    // add buffers to migrate
    std::vector<cl::Memory> init;
    init.push_back(hds[0].buffer[2]);
    init.push_back(hds[0].buffer[3]);
    init.push_back(hds[0].buffer[4]);
    init.push_back(hds[0].buffer[5]);
    if (maxIter % 2) {
        init.push_back(hds[0].buffer[6]);
    } else {
        init.push_back(hds[0].buffer[7]);
    }
    std::vector<cl::Event> event(1);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &event[0]);
    event[0].wait();

    ob_in.push_back(hds[0].buffer[0]);
    ob_in.push_back(hds[0].buffer[1]);
    if (maxIter % 2) {
        ob_out.push_back(hds[0].buffer[7]);
    } else {
        ob_out.push_back(hds[0].buffer[6]);
    }

    kernel0.setArg(0, g.nodeNum);         // node number
    kernel0.setArg(1, g.edgeNum);         // edge number
    kernel0.setArg(2, maxIter);           // max Iteration
    kernel0.setArg(3, hds[0].buffer[0]);  // offsetsCSR
    kernel0.setArg(4, hds[0].buffer[1]);  // indicesCSR
    kernel0.setArg(5, hds[0].buffer[2]);  // offsetsCSC
    kernel0.setArg(6, hds[0].buffer[3]);  // indicesCSC
    kernel0.setArg(7, hds[0].buffer[3]);  // indicesCSC
    kernel0.setArg(8, hds[0].buffer[4]);  // buffPing
    kernel0.setArg(9, hds[0].buffer[5]);  // buffPong
    kernel0.setArg(10, hds[0].buffer[6]); // labelPing
    kernel0.setArg(11, hds[0].buffer[7]); // labelPong
};

int opLabelPropagation::cuExecute(
    clHandle* hds, cl::Kernel& kernel0, unsigned int num_runs, std::vector<cl::Event>* evIn, cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueTask(kernel0, evIn, evOut);
    }
    return 0;
}

int opLabelPropagation::compute(unsigned int deviceID,
                                unsigned int cuID,
                                unsigned int channelID,
                                xrmContext* ctx,
                                xrmCuResource* resR,
                                std::string instanceName,
                                clHandle* handles,
                                uint32_t maxIter,
                                xf::graph::Graph<uint32_t, uint32_t> g,
                                uint32_t* labels) {
    clHandle* hds = &handles[channelID + cuID * dupNmLabelPropagation +
                             deviceID * dupNmLabelPropagation * cuPerBoardLabelPropagation];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    unsigned int num_runs = 1;

    uint32_t maxVertices = 800000 * 16;
    uint32_t* labelP = aligned_alloc<uint32_t>(maxVertices);
    uint32_t* buffPing = aligned_alloc<uint32_t>(maxVertices);
    uint32_t* buffPong = aligned_alloc<uint32_t>(maxVertices);

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    if (maxIter % 2) {
        bufferInit(hds, instanceName, maxIter, labelP, labels, buffPing, buffPong, g, kernel0, ob_in, ob_out);
    } else {
        bufferInit(hds, instanceName, maxIter, labels, labelP, buffPing, buffPong, g, kernel0, ob_in, ob_out);
    }

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    cuRelease(ctx, resR);

    free(buffPing);
    free(buffPong);

    return ret;
};

event<int> opLabelPropagation::addwork(uint32_t maxIter, xf::graph::Graph<uint32_t, uint32_t> g, uint32_t* labels) {
    return createL3(task_queue[0], &(compute), handles, maxIter, g, labels);
};

} // L3
} // graph
} // xf
#endif
