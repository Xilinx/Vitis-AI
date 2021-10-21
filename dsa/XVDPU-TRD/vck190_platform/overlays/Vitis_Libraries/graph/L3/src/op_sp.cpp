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

#ifndef _XF_GRAPH_L3_OP_SP_CPP_
#define _XF_GRAPH_L3_OP_SP_CPP_

#include "op_sp.hpp"
#include "xf_utils_sw/logger.hpp"

namespace xf {
namespace graph {
namespace L3 {

void createHandleSP(clHandle& handle, const char* kernelName, const char* pXclbin, int32_t IDDevice) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int fail;

    // Platform related operations
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

uint32_t opSP::cuPerBoardSP;

uint32_t opSP::dupNmSP;

void opSP::setHWInfo(uint32_t numDev, uint32_t CUmax) {
    maxCU = CUmax;
    deviceNm = numDev;
    cuPerBoardSP = maxCU / deviceNm;
    handles = new clHandle[CUmax];
};

void opSP::freeSP() {
    msspThread.join();
    for (int i = 0; i < maxCU; ++i) {
        delete[] handles[i].buffer;
    }
    delete[] handles;
};

void opSP::cuRelease(xrmContext* ctx, xrmCuResource* resR) {
    while (!xrmCuRelease(ctx, resR)) {
    };
    free(resR);
};

void opSP::init(char* kernelName, char* xclbinFile, uint32_t* deviceIDs, uint32_t* cuIDs, unsigned int requestLoad) {
    dupNmSP = 100 / requestLoad;
    cuPerBoardSP /= dupNmSP;
    uint32_t bufferNm = 8;
    unsigned int cnt = 0;
    unsigned int cntCU = 0;
    unsigned int* handleID = new unsigned int[maxCU];
    handleID[0] = cnt;
    handles[0].deviceID = deviceIDs[0];
    handles[0].cuID = cuIDs[0];
    handles[0].dupID = 0;
    std::thread th[maxCU];
    createHandleSP(handles[cnt], kernelName, xclbinFile, deviceIDs[cnt]);
    handles[cnt].buffer = new cl::Buffer[bufferNm];
    unsigned int prev = deviceIDs[0];
    unsigned int prevCU = cuIDs[0];
    deviceOffset.push_back(0);
    for (int i = 1; i < maxCU; ++i) {
        handles[i].deviceID = deviceIDs[i];
        handles[i].cuID = cuIDs[i];
        handles[i].dupID = i % dupNmSP;
        createHandleSP(handles[i], kernelName, xclbinFile, deviceIDs[i]);
        handles[i].buffer = new cl::Buffer[bufferNm];
        if (deviceIDs[i] != prev) {
            prev = deviceIDs[i];
            deviceOffset.push_back(i);
        }
    }
    delete[] handleID;
}

void opSP::migrateMemObj(clHandle* hds,
                         bool type,
                         unsigned int num_runs,
                         std::vector<cl::Memory>& ob,
                         std::vector<cl::Event>* evIn,
                         cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueMigrateMemObjects(ob, type, evIn, evOut); // 0 : migrate from host to dev
    }
};

void runF(std::future<void> fut) {
    fut.get();
}

void loadGraphCoreSP(clHandle* hds, int nrows, int nnz, xf::graph::Graph<uint32_t, float> g) {
    cl::Device device = hds[0].device;
    cl::Context context = hds[0].context;
    cl::CommandQueue q = hds[0].q;
    uint32_t* ddrQue;
    ddrQue = aligned_alloc<uint32_t>(10 * 300 * 4096);

    std::vector<cl_mem_ext_ptr_t> mext_in = std::vector<cl_mem_ext_ptr_t>(4);
#ifndef USE_HBM
    // DDR Settings
    mext_in[0] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
    mext_in[1] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
    mext_in[2] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, g.weightsCSR, 0};
    mext_in[3] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, ddrQue, 0};
#else
    mext_in[0] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
    mext_in[1] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
    mext_in[2] = {(unsigned int)(5) | XCL_MEM_TOPOLOGY, g.weightsCSR, 0};
    mext_in[3] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, ddrQue, 0};
#endif

    // Create device buffer and map dev buf to host buf
    hds[0].buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * (nrows + 1),
                                  &mext_in[0]); // offset// for band to one axi
    hds[0].buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  sizeof(uint32_t) * nnz, &mext_in[1]); // indice
    hds[0].buffer[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  sizeof(float) * nnz, &mext_in[2]); // weight
    hds[0].buffer[5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * 10 * 300 * 4096, &mext_in[3]); // ddrQue

    // add buffers to migrate
    std::vector<cl::Event> eventSecond(1);

    std::vector<cl::Memory> ob_in;
    ob_in.push_back(hds[0].buffer[0]);
    ob_in.push_back(hds[0].buffer[1]);
    ob_in.push_back(hds[0].buffer[2]);
    ob_in.push_back(hds[0].buffer[5]);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &eventSecond[0]); // 0 : migrate from host to dev

    eventSecond[0].wait();
};

void opSP::loadGraph(xf::graph::Graph<uint32_t, float> g) {
    int nnz = g.edgeNum;
    int nrows = g.nodeNum;

    std::thread* th = new std::thread[maxCU];
    std::future<void>* fut = new std::future<void>[ maxCU ];
    int cnt = 0;
    for (int j = 0; j < maxCU; ++j) {
        if ((handles[j].cuID == 0) && (handles[j].dupID == 0)) {
            cnt = j;
            std::packaged_task<void(clHandle*, int, int, xf::graph::Graph<uint32_t, float>)> t(loadGraphCoreSP);
            fut[j] = t.get_future();
            th[j] = std::thread(std::move(t), &handles[j], nrows, nnz, g);
        }
    }
    for (int j = 0; j < maxCU; ++j) {
        if ((handles[j].cuID == 0) && (handles[j].dupID == 0)) {
            fut[j].get();
            th[j].join();
        }
    }
    cnt = 0;
    for (int j = 0; j < maxCU; ++j) {
        if (!((handles[j].cuID == 0) && (handles[j].dupID == 0))) {
            handles[j].buffer[0] = handles[cnt].buffer[0];
            handles[j].buffer[1] = handles[cnt].buffer[1];
            handles[j].buffer[2] = handles[cnt].buffer[2];
            handles[j].buffer[5] = handles[cnt].buffer[5];
        } else {
            cnt = j;
        }
    }
    delete[] th;
    delete[] fut;
};

void opSP::bufferInit(clHandle* hds,
                      std::string instanceName0,
                      xf::graph::Graph<uint32_t, float> g,
                      int nrows,
                      uint8_t* info,
                      float* result,
                      uint32_t* pred,
                      uint32_t* config,
                      cl::Kernel& kernel0,
                      std::vector<cl::Memory>& ob_in,
                      std::vector<cl::Memory>& ob_out) {
    int nnz = g.edgeNum;

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
    kernel0 = cl::Kernel(program, instanceName);
    std::cout << "INFO: Kernel has been created" << std::endl;

    std::vector<cl_mem_ext_ptr_t> mext_in = std::vector<cl_mem_ext_ptr_t>(4);
#ifdef USE_HBM
    mext_in[0] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, info, 0};
    mext_in[1] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, config, 0};
    mext_in[2] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, result, 0};
    mext_in[3] = {(unsigned int)(5) | XCL_MEM_TOPOLOGY, pred, 0};
#else
    mext_in[0] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, info, 0};
    mext_in[1] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, config, 0};
    mext_in[2] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, result, 0};
    mext_in[3] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, pred, 0};
#endif
    // Create device buffer and map dev buf to host buf

    hds[0].buffer[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint8_t) * 4, &mext_in[0]); // info
    hds[0].buffer[4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * 6, &mext_in[1]); // config

    hds[0].buffer[6] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(float) * ((nrows + 1023) / 1024) * 1024, &mext_in[2]); // result
    hds[0].buffer[7] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * ((nrows + 1023) / 1024) * 1024, &mext_in[3]); // pred

    // add buffers to migrate
    std::vector<cl::Memory> init;
    init.push_back(hds[0].buffer[3]);
    init.push_back(hds[0].buffer[4]);
    init.push_back(hds[0].buffer[6]);
    init.push_back(hds[0].buffer[7]);

    std::vector<cl::Event> event(1);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &event[0]);

    event[0].wait();

    ob_in.push_back(hds[0].buffer[3]);
    ob_in.push_back(hds[0].buffer[4]);
    ob_out.push_back(hds[0].buffer[3]);
    ob_out.push_back(hds[0].buffer[6]);
    ob_out.push_back(hds[0].buffer[7]);

    kernel0.setArg(0, hds[0].buffer[4]);  // config
    kernel0.setArg(1, hds[0].buffer[0]);  // offset
    kernel0.setArg(2, hds[0].buffer[1]);  // indice
    kernel0.setArg(3, hds[0].buffer[2]);  // weight
    kernel0.setArg(4, hds[0].buffer[5]);  // ddrQue
    kernel0.setArg(5, hds[0].buffer[5]);  // ddrQue
    kernel0.setArg(6, hds[0].buffer[6]);  // result
    kernel0.setArg(7, hds[0].buffer[6]);  // result
    kernel0.setArg(8, hds[0].buffer[7]);  // pred
    kernel0.setArg(9, hds[0].buffer[7]);  // pred
    kernel0.setArg(10, hds[0].buffer[3]); // info
};

int opSP::cuExecute(
    clHandle* hds, cl::Kernel& kernel0, unsigned int num_runs, std::vector<cl::Event>* evIn, cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueTask(kernel0, evIn, evOut);
    }
    return 0;
}

void opSP::postProcess(int nrows, uint8_t* info, int& ret) {
    if (info[0] != 0) {
        std::cout << "Error: queue overflow" << std::endl;
        ret = 1;
    }
    if (info[1] != 0) {
        std::cout << "Error: table overflow" << std::endl;
        ret = 1;
    }
};

int opSP::compute(unsigned int deviceID,
                  unsigned int cuID,
                  unsigned int channelID,
                  xrmContext* ctx,
                  xrmCuResource* resR,
                  std::string instanceName,
                  clHandle* handles,
                  uint32_t nSource,
                  uint32_t* sourceID,
                  bool weighted,
                  xf::graph::Graph<uint32_t, float> g,
                  float* result,
                  uint32_t* pred) {
    clHandle* hds = &handles[channelID + cuID * dupNmSP + deviceID * dupNmSP * cuPerBoardSP];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    unsigned int num_runs = 1;
    int nrows = g.nodeNum;
    int nnz = g.edgeNum;
    uint8_t* info = aligned_alloc<uint8_t>(4);
    uint32_t* config = aligned_alloc<uint32_t>(6);

    f_cast<float> tmp;
    tmp.f = std::numeric_limits<float>::infinity();

    uint32_t cmd;
    bool enablePred = 1;
    if (weighted && enablePred) {
        cmd = 3;
    } else if (weighted && !enablePred) {
        cmd = 1;
    } else if (!weighted && !enablePred) {
        cmd = 0;
    } else if (!weighted && enablePred) {
        cmd = 2;
    }

    config[0] = nrows;
    config[1] = tmp.i; //-1;
    config[2] = 0;
    config[3] = 10 * 300 * 4096;
    config[4] = cmd;
    config[5] = sourceID[0];

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, nrows, info, result, pred, config, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    postProcess(nrows, info, ret);

    cuRelease(ctx, resR);

    free(info);
    free(config);

    return ret;
};

event<int> opSP::addwork(uint32_t nSource,
                         uint32_t* sourceID,
                         bool weighted,
                         xf::graph::Graph<uint32_t, float> g,
                         float* result,
                         uint32_t* pred) {
    return createL3(task_queue[0], &(compute), handles, nSource, sourceID, weighted, g, result, pred);
};

} // L3
} // graph
} // xf
#endif
