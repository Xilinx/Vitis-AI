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

#ifndef _XF_GRAPH_L3_OP_TWOHOP_CPP_
#define _XF_GRAPH_L3_OP_TWOHOP_CPP_

#include "op_twohop.hpp"
#include "xf_utils_sw/logger.hpp"

namespace xf {
namespace graph {
namespace L3 {

void createHandleTwoHop(clHandle& handle, const char* kernelName, const char* pXclbin, int32_t IDDevice) {
    // Platform related operations
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int fail;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    handle.device = devices[IDDevice];
    handle.context = cl::Context(handle.device, NULL, NULL, NULL, &fail);
    logger.logCreateContext(fail);
    handle.q = cl::CommandQueue(handle.context, handle.device,
                                CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &fail);
    logger.logCreateCommandQueue(fail);

    std::string devName = handle.device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    handle.xclBins = xcl::import_binary_file(pXclbin);
    std::vector<cl::Device> devices2;
    devices2.push_back(handle.device);
    handle.program = cl::Program(handle.context, devices2, handle.xclBins, NULL, &fail);
    logger.logCreateProgram(fail);
}

uint32_t opTwoHop::cuPerBoardTwoHop;

uint32_t opTwoHop::dupNmTwoHop;

void opTwoHop::setHWInfo(uint32_t numDev, uint32_t CUmax) {
    maxCU = CUmax;
    deviceNm = numDev;
    cuPerBoardTwoHop = maxCU / deviceNm;
    handles = new clHandle[CUmax];
};

void opTwoHop::freeTwoHop() {
    twoHopThread.join();
    for (int i = 0; i < maxCU; ++i) {
        delete[] handles[i].buffer;
    }
    delete[] handles;
};

void opTwoHop::cuRelease(xrmContext* ctx, xrmCuResource* resR) {
    while (!xrmCuRelease(ctx, resR)) {
    };
    free(resR);
};

void opTwoHop::init(
    char* kernelName, char* xclbinFile, uint32_t* deviceIDs, uint32_t* cuIDs, unsigned int requestLoad) {
    dupNmTwoHop = 100 / requestLoad;
    cuPerBoardTwoHop /= dupNmTwoHop;
    uint32_t bufferNm = 6;
    unsigned int cnt = 0;
    unsigned int cntCU = 0;
    unsigned int* handleID = new unsigned int[maxCU];
    handleID[0] = cnt;
    handles[0].deviceID = deviceIDs[0];
    handles[0].cuID = cuIDs[0];
    handles[0].dupID = 0;
    std::thread th[maxCU];
    createHandleTwoHop(handles[cnt], kernelName, xclbinFile, deviceIDs[cnt]);
    handles[cnt].buffer = new cl::Buffer[bufferNm];
    unsigned int prev = deviceIDs[0];
    unsigned int prevCU = cuIDs[0];
    deviceOffset.push_back(0);
    for (int i = 1; i < maxCU; ++i) {
        handles[i].deviceID = deviceIDs[i];
        handles[i].cuID = cuIDs[i];
        handles[i].dupID = i % dupNmTwoHop;
        createHandleTwoHop(handles[i], kernelName, xclbinFile, deviceIDs[i]);
        handles[i].buffer = new cl::Buffer[bufferNm];
        if (deviceIDs[i] != prev) {
            prev = deviceIDs[i];
            deviceOffset.push_back(i);
        }
    }
    delete[] handleID;
}

void opTwoHop::migrateMemObj(clHandle* hds,
                             bool type,
                             unsigned int num_runs,
                             std::vector<cl::Memory>& ob,
                             std::vector<cl::Event>* evIn,
                             cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueMigrateMemObjects(ob, type, evIn, evOut); // 0 : migrate from host to dev
    }
};

void loadGraphCoreTwoHop(clHandle* hds, int nrows, int nnz, int cuID, xf::graph::Graph<uint32_t, float> g) {
    cl::Device device = hds[0].device;
    cl::Context context = hds[0].context;
    cl::CommandQueue q = hds[0].q;

    std::vector<cl_mem_ext_ptr_t> mext_in = std::vector<cl_mem_ext_ptr_t>(4);

    if (cuID == 0) {
        mext_in[0] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[1] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
        mext_in[2] = {(unsigned int)(5) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[3] = {(unsigned int)(5) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
    } else if (cuID == 1) {
        mext_in[0] = {(unsigned int)(9) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[1] = {(unsigned int)(9) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
        mext_in[2] = {(unsigned int)(10) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[3] = {(unsigned int)(10) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
    } else if (cuID == 2) {
        mext_in[0] = {(unsigned int)(14) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[1] = {(unsigned int)(14) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
        mext_in[2] = {(unsigned int)(16) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[3] = {(unsigned int)(16) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
    } else if (cuID == 3) {
        mext_in[0] = {(unsigned int)(20) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[1] = {(unsigned int)(20) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
        mext_in[2] = {(unsigned int)(23) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[3] = {(unsigned int)(23) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
    } else if (cuID == 4) {
        mext_in[0] = {(unsigned int)(27) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[1] = {(unsigned int)(27) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
        mext_in[2] = {(unsigned int)(25) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
        mext_in[3] = {(unsigned int)(25) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
    } else {
        printf("Error: unknown cu detected. Cannot find memory bank\n");
    }

    // Create device buffer and map dev buf to host buf
    hds[0].buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * (nrows + 1), &mext_in[0]); // one hop offset
    hds[0].buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  sizeof(uint32_t) * nnz, &mext_in[1]); // one hop index
    hds[0].buffer[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  sizeof(uint32_t) * (nrows + 1), &mext_in[2]); // two hop offset
    hds[0].buffer[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * nnz, &mext_in[3]); // two hop index

    // add buffers to migrate
    std::vector<cl::Event> eventSecond(1);

    std::vector<cl::Memory> ob_in;
    ob_in.push_back(hds[0].buffer[0]);
    ob_in.push_back(hds[0].buffer[1]);
    ob_in.push_back(hds[0].buffer[2]);
    ob_in.push_back(hds[0].buffer[3]);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &eventSecond[0]); // 0 : migrate from host to dev

    eventSecond[0].wait();
};

void opTwoHop::loadGraph(int deviceID, int cuID, xf::graph::Graph<uint32_t, float> g) {
    int nnz = g.edgeNum;
    int nrows = g.nodeNum;
    bool freed[maxCU];

    std::thread* th = new std::thread[maxCU];
    std::future<void>* fut = new std::future<void>[ maxCU ];
    int cnt = 0;
    for (unsigned int j = 0; j < maxCU; ++j) {
        if ((handles[j].deviceID == (unsigned int)deviceID) && (handles[j].cuID == (unsigned int)cuID) &&
            (handles[j].dupID == 0)) {
            cnt = j;
            std::packaged_task<void(clHandle*, int, int, int, xf::graph::Graph<uint32_t, float>)> t(
                loadGraphCoreTwoHop);
            fut[j] = t.get_future();
            th[j] = std::thread(std::move(t), &handles[j], nrows, nnz, cuID, g);
        }
        freed[j] = 0;
    }
    cnt = 0;
    for (unsigned int j = 0; j < maxCU; ++j) {
        if ((handles[j].deviceID == (unsigned int)deviceID) && (handles[j].cuID == (unsigned int)cuID)) {
            if (handles[j].dupID != 0) {
                if (freed[cnt] == 0) {
                    fut[cnt].get();
                    th[cnt].join();
                    freed[cnt] = 1;
                }
                handles[j].buffer[0] = handles[cnt].buffer[0];
                handles[j].buffer[1] = handles[cnt].buffer[1];
                handles[j].buffer[2] = handles[cnt].buffer[2];
                handles[j].buffer[3] = handles[cnt].buffer[3];
            } else {
                cnt = j;
            }
        }
    }
    for (unsigned int j = 0; j < maxCU; ++j) {
        if ((handles[j].deviceID == (unsigned int)deviceID) && (handles[j].cuID == (unsigned int)cuID) &&
            (handles[j].dupID == 0)) {
            if (freed[j] == 0) {
                fut[j].get();
                th[j].join();
            }
        }
    }
    delete[] th;
    delete[] fut;
};

void opTwoHop::bufferInit(clHandle* hds,
                          std::string instanceName0,
                          xf::graph::Graph<uint32_t, float> g,
                          int cuID,
                          uint32_t numPart,
                          uint64_t* pairPart,
                          uint32_t* resPart,
                          cl::Kernel& kernel0,
                          std::vector<cl::Memory>& ob_in,
                          std::vector<cl::Memory>& ob_out) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int fail;

    int nnz = g.edgeNum;

    cl::Device device = hds[0].device;

    instanceName0 = "twoHop_kernel:{" + instanceName0 + "}";
    const char* instanceName = instanceName0.c_str();

    // Creating Context and Command Queue for selected Device
    cl::Context context = hds[0].context;
    cl::CommandQueue q = hds[0].q;
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    std::vector<cl::Device> devices;
    devices.push_back(hds[0].device);
    cl::Program program = hds[0].program;
    kernel0 = cl::Kernel(program, instanceName, &fail);
    logger.logCreateKernel(fail);

    std::cout << "INFO: Kernel has been created" << std::endl;

    std::vector<cl_mem_ext_ptr_t> mext_in = std::vector<cl_mem_ext_ptr_t>(2);

    if (strcmp(instanceName, "twoHop_kernel:{twoHop_kernel0}") == 0) {
        mext_in[0] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, pairPart, 0};
        mext_in[1] = {(unsigned int)(1) | XCL_MEM_TOPOLOGY, resPart, 0};
    } else if (strcmp(instanceName, "twoHop_kernel:{twoHop_kernel1}") == 0) {
        mext_in[0] = {(unsigned int)(6) | XCL_MEM_TOPOLOGY, pairPart, 0};
        mext_in[1] = {(unsigned int)(7) | XCL_MEM_TOPOLOGY, resPart, 0};
    } else if (strcmp(instanceName, "twoHop_kernel:{twoHop_kernel2}") == 0) {
        mext_in[0] = {(unsigned int)(12) | XCL_MEM_TOPOLOGY, pairPart, 0};
        mext_in[1] = {(unsigned int)(13) | XCL_MEM_TOPOLOGY, resPart, 0};
    } else if (strcmp(instanceName, "twoHop_kernel:{twoHop_kernel3}") == 0) {
        mext_in[0] = {(unsigned int)(18) | XCL_MEM_TOPOLOGY, pairPart, 0};
        mext_in[1] = {(unsigned int)(19) | XCL_MEM_TOPOLOGY, resPart, 0};
    } else if (strcmp(instanceName, "twoHop_kernel:{twoHop_kernel4}") == 0) {
        mext_in[0] = {(unsigned int)(28) | XCL_MEM_TOPOLOGY, pairPart, 0};
        mext_in[1] = {(unsigned int)(29) | XCL_MEM_TOPOLOGY, resPart, 0};
    } else {
        printf("ERROR: unknow cu detected. cannot find memory bank");
    }

    // Create device buffer and map dev buf to host buf
    hds[0].buffer[4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint64_t) * numPart, &mext_in[0]); // pairPart
    hds[0].buffer[5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * numPart, &mext_in[1]); // resPart

    // add buffers to migrate
    std::vector<cl::Memory> init;
    init.push_back(hds[0].buffer[4]);
    init.push_back(hds[0].buffer[5]);

    std::vector<cl::Event> event(1);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &event[0]);

    event[0].wait();

    ob_in.push_back(hds[0].buffer[4]);
    ob_out.push_back(hds[0].buffer[5]);

    kernel0.setArg(0, numPart);          // config
    kernel0.setArg(1, hds[0].buffer[4]); // pair
    kernel0.setArg(2, hds[0].buffer[0]); // one hop offset
    kernel0.setArg(3, hds[0].buffer[1]); // one hop index
    kernel0.setArg(4, hds[0].buffer[2]); // two hop offset
    kernel0.setArg(5, hds[0].buffer[3]); // two hop index
    kernel0.setArg(6, hds[0].buffer[5]); // result
};

int opTwoHop::cuExecute(
    clHandle* hds, cl::Kernel& kernel0, unsigned int num_runs, std::vector<cl::Event>* evIn, cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueTask(kernel0, evIn, evOut);
    }
    return 0;
}

int opTwoHop::compute(unsigned int deviceID,
                      unsigned int cuID,
                      unsigned int channelID,
                      xrmContext* ctx,
                      xrmCuResource* resR,
                      std::string instanceName,
                      clHandle* handles,
                      uint32_t numPart,
                      uint64_t* pairPart,
                      uint32_t* resPart,
                      xf::graph::Graph<uint32_t, float> g) {
    uint32_t local_cuID;
    if (strcmp(instanceName.c_str(), "twoHop_kernel0") == 0) {
        local_cuID = 0;
    } else if (strcmp(instanceName.c_str(), "twoHop_kernel1") == 0) {
        local_cuID = 1;
    } else if (strcmp(instanceName.c_str(), "twoHop_kernel2") == 0) {
        local_cuID = 2;
    } else if (strcmp(instanceName.c_str(), "twoHop_kernel3") == 0) {
        local_cuID = 3;
    } else if (strcmp(instanceName.c_str(), "twoHop_kernel4") == 0) {
        local_cuID = 4;
    } else {
        std::cout << "unknown cu instance name" << std::endl;
    }

    clHandle* hds = &handles[local_cuID];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    unsigned int num_runs = 1;
    int nrows = g.nodeNum;
    int nnz = g.edgeNum;

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, local_cuID, numPart, pairPart, resPart, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    cuRelease(ctx, resR);

    return ret;
};

event<int> opTwoHop::addwork(uint32_t numPart,
                             uint64_t* pairPart,
                             uint32_t* resPart,
                             xf::graph::Graph<uint32_t, float> g) {
    return createL3(task_queue[0], &(compute), handles, numPart, pairPart, resPart, g);
};

} // L3
} // graph
} // xf
#endif
