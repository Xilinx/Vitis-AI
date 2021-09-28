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

#ifndef _XF_GRAPH_L3_OP_PAGERANK_CPP_
#define _XF_GRAPH_L3_OP_PAGERANK_CPP_

#include "op_pagerank.hpp"
#include "xf_utils_sw/logger.hpp"

namespace xf {
namespace graph {
namespace L3 {

void createHandlePG(clHandle& handle, const char* kernelName, const char* pXclbin, int32_t IDDevice) {
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

uint32_t opPageRank::cuPerBoardPG;

uint32_t opPageRank::dupNmPG;

void opPageRank::setHWInfo(uint32_t numDev, uint32_t CUmax) {
    maxCU = CUmax;
    deviceNm = numDev;
    cuPerBoardPG = maxCU / deviceNm;
    handles = new clHandle[CUmax];
};

void opPageRank::freePG() {
    for (int i = 0; i < maxCU; ++i) {
        delete[] handles[i].buffer;
    }
    delete[] handles;
};

void opPageRank::cuRelease(xrmContext* ctx, xrmCuResource* resR) {
    while (!xrmCuRelease(ctx, resR)) {
    };
    free(resR);
};

void opPageRank::init(
    char* kernelName, char* xclbinFile, uint32_t* deviceIDs, uint32_t* cuIDs, unsigned int requestLoad) {
    dupNmPG = 100 / requestLoad;
    cuPerBoardPG /= dupNmPG;
    uint32_t bufferNm = 9;
    unsigned int cnt = 0;
    unsigned int cntCU = 0;
    unsigned int* handleID = new unsigned int[maxCU];
    handleID[0] = cnt;
    std::thread th[maxCU];
    // th[0] = std::thread(&createHandlePG, std::ref(handles[cnt]), kernelName, xclbinFile, deviceIDs[cnt]);
    createHandlePG(handles[cnt], kernelName, xclbinFile, deviceIDs[cnt]);
    handles[cnt].buffer = new cl::Buffer[bufferNm];
    unsigned int prev = deviceIDs[0];
    unsigned int prevCU = cuIDs[0];
    deviceOffset.push_back(0);
    handles[0].deviceID = deviceIDs[0];
    handles[0].cuID = cuIDs[0];
    handles[0].dupID = 0 % dupNmPG;
    for (int i = 1; i < maxCU; ++i) {
        handles[i].deviceID = deviceIDs[i];
        handles[i].cuID = cuIDs[i];
        handles[i].dupID = i % dupNmPG;
        // th[i] = std::thread(&createHandlePG, std::ref(handles[i]), kernelName, xclbinFile, deviceIDs[i]);
        createHandlePG(handles[i], kernelName, xclbinFile, deviceIDs[i]);
        handles[i].buffer = new cl::Buffer[bufferNm];
        if (deviceIDs[i] != prev) {
            prev = deviceIDs[i];
            deviceOffset.push_back(i);
        }
    }
    //  for (int j = 0; j < maxCU; ++j) {
    //      th[j].join();
    //  }
    delete[] handleID;
}

void opPageRank::migrateMemObj(clHandle* hds,
                               bool type,
                               unsigned int num_runs,
                               std::vector<cl::Memory>& ob,
                               std::vector<cl::Event>* evIn,
                               cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueMigrateMemObjects(ob, type, evIn, evOut); // 0 : migrate from host to dev
    }
};

void loadGraphCorePG(clHandle* hds, int nrows, int nnz, xf::graph::Graph<uint32_t, float> g) {
    //// Creating Context and Command Queue for selected Device
    cl::Device device = hds[0].device;
    cl::Context context = hds[0].context;
    cl::CommandQueue q = hds[0].q;

    std::vector<cl_mem_ext_ptr_t> mext_in = std::vector<cl_mem_ext_ptr_t>(3);
#ifndef USE_HBM
    // DDR Settings
    mext_in[0] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
    mext_in[1] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
    mext_in[2] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, g.weightsCSR, 0};
#else

    mext_in[0] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, g.offsetsCSR, 0};
    mext_in[1] = {(unsigned int)(2) | XCL_MEM_TOPOLOGY, g.indicesCSR, 0};
    mext_in[2] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, g.weightsCSR, 0};
#endif

    // Create device buffer and map dev buf to host buf
    hds[0].buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * (nrows + 1),
                                  &mext_in[0]); // offset// for band to one axi
    hds[0].buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  sizeof(uint32_t) * nnz, &mext_in[1]); // indice
    hds[0].buffer[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  sizeof(float) * nnz, &mext_in[2]); // weight

    // add buffers to migrate
    std::vector<cl::Memory> init;
    for (int i = 0; i < 3; i++) {
        init.push_back(hds[0].buffer[i]);
    }

    std::vector<cl::Event> eventFirst(1);
    std::vector<cl::Event> eventSecond(1);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &eventFirst[0]);

    std::vector<cl::Memory> ob_in;
    ob_in.push_back(hds[0].buffer[0]);
    ob_in.push_back(hds[0].buffer[1]);
    ob_in.push_back(hds[0].buffer[2]);

    q.enqueueMigrateMemObjects(ob_in, 0, &eventFirst, &eventSecond[0]); // 0 : migrate from host to dev

    eventSecond[0].wait();
};

void opPageRank::loadGraph(xf::graph::Graph<uint32_t, float> g) {
    int nnz = g.edgeNum;
    int nrows = g.nodeNum;

    bool freed[maxCU];

    std::thread* th = new std::thread[maxCU];
    std::future<void>* fut = new std::future<void>[ maxCU ];
    int cnt = 0;
    for (int j = 0; j < maxCU; ++j) {
        if ((handles[j].cuID == 0) && (handles[j].dupID == 0)) {
            cnt = j;
            std::packaged_task<void(clHandle*, int, int, xf::graph::Graph<uint32_t, float>)> t(loadGraphCorePG);
            fut[j] = t.get_future();
            th[j] = std::thread(std::move(t), &handles[j], nrows, nnz, g);
        }
        freed[j] = 0;
    }
    cnt = 0;
    for (int j = 0; j < maxCU; ++j) {
        if (!((handles[j].cuID == 0) && (handles[j].dupID == 0))) {
            if (freed[cnt] == 0) {
                fut[cnt].get();
                th[cnt].join();
                freed[cnt] = 1;
            }
            handles[j].buffer[0] = handles[cnt].buffer[0];
            handles[j].buffer[1] = handles[cnt].buffer[1];
            handles[j].buffer[2] = handles[cnt].buffer[2];
        } else {
            cnt = j;
        }
    }
    for (int j = 0; j < maxCU; ++j) {
        if ((handles[j].cuID == 0) && (handles[j].dupID == 0)) {
            if (freed[j] == 0) {
                fut[j].get();
                th[j].join();
            }
        }
    }
    delete[] th;
    delete[] fut;
};

void opPageRank::bufferInit(clHandle* hds,
                            std::string instanceName0,
                            xf::graph::Graph<uint32_t, float> g,
                            int nrows,
                            float alpha,
                            float tolerance,
                            int maxIter,
                            int num_runs,
                            uint32_t* degreeCSR,
                            uint32_t* cntValFull,
                            uint32_t* buffPing,
                            uint32_t* buffPong,
                            int* resultInfo,
                            uint32_t* orderUnroll,
                            cl::Kernel& kernel0,
                            std::vector<cl::Memory>& ob_in,
                            std::vector<cl::Memory>& ob_out) {
    int unrollNm2 = (sizeof(float) == 4) ? 16 : 8;
    int iteration2 = ((nrows + unrollNm2 - 1) / unrollNm2) * unrollNm2;
    int nnz = g.edgeNum;

    const char* instanceName = instanceName0.c_str();
    cl::Device device = hds[0].device;

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

    std::vector<cl::Buffer> buffer;
    std::vector<cl_mem_ext_ptr_t> mext_in;

    mext_in = std::vector<cl_mem_ext_ptr_t>(6);
#ifndef USE_HBM
    // DDR Settings
    mext_in[0] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, degreeCSR, 0};
    mext_in[1] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, cntValFull, 0};
    mext_in[2] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, buffPing, 0};
    mext_in[3] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, buffPong, 0};
    mext_in[4] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, resultInfo, 0};
    mext_in[5] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, orderUnroll, 0};
#else

    mext_in[0] = {(unsigned int)(6) | XCL_MEM_TOPOLOGY, degreeCSR, 0};
    mext_in[1] = {(unsigned int)(8) | XCL_MEM_TOPOLOGY, cntValFull, 0};
    mext_in[2] = {(unsigned int)(10) | XCL_MEM_TOPOLOGY, buffPing, 0};
    mext_in[3] = {(unsigned int)(12) | XCL_MEM_TOPOLOGY, buffPong, 0};
    mext_in[4] = {(unsigned int)(12) | XCL_MEM_TOPOLOGY, resultInfo, 0};
    mext_in[5] = {(unsigned int)(1) | XCL_MEM_TOPOLOGY, orderUnroll, 0};
#endif

    // Create device buffer and map dev buf to host buf
    hds[0].buffer[3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * nrows, &mext_in[0]); // degree
    hds[0].buffer[4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * iteration2, &mext_in[1]); // const
    hds[0].buffer[5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * iteration2, &mext_in[2]); // buffp
    hds[0].buffer[6] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * iteration2, &mext_in[3]); // buffq
    hds[0].buffer[7] =
        cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * (2),
                   &mext_in[4]); // resultInfo
    hds[0].buffer[8] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * (nrows + 16), &mext_in[5]); // order

    // add buffers to migrate
    std::vector<cl::Memory> init;
    for (int i = 0; i < 6; i++) {
        init.push_back(hds[0].buffer[i + 3]);
    }

    std::vector<cl::Event> event(1);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &event[0]);

    event[0].wait();

    ob_in.push_back(hds[0].buffer[3]);
    ob_in.push_back(hds[0].buffer[4]);
    ob_out.push_back(hds[0].buffer[5]);
    ob_out.push_back(hds[0].buffer[6]);
    ob_out.push_back(hds[0].buffer[7]);

    ob_in.push_back(hds[0].buffer[8]);

    kernel0.setArg(0, nrows);
    kernel0.setArg(1, nnz);
    kernel0.setArg(2, alpha);
    kernel0.setArg(3, tolerance);
    kernel0.setArg(4, maxIter);
    kernel0.setArg(5, hds[0].buffer[0]);
    kernel0.setArg(6, hds[0].buffer[1]);
    kernel0.setArg(7, hds[0].buffer[2]);
    kernel0.setArg(8, hds[0].buffer[3]);
    kernel0.setArg(9, hds[0].buffer[4]);

    kernel0.setArg(10, hds[0].buffer[5]);
    kernel0.setArg(11, hds[0].buffer[6]);
    kernel0.setArg(12, hds[0].buffer[7]);
    kernel0.setArg(13, hds[0].buffer[8]);
};

int opPageRank::cuExecute(
    clHandle* hds, cl::Kernel& kernel0, unsigned int num_runs, std::vector<cl::Event>* evIn, cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueTask(kernel0, evIn, evOut);
    }
    return 0;
}

void opPageRank::postProcess(int nrows, int* resultInfo, uint32_t* buffPing, uint32_t* buffPong, float* pagerank) {
    bool resultinPong = (bool)(*resultInfo);
    int iterations = (int)(*(resultInfo + 1));
    int unrollNm2 = (sizeof(float) == 4) ? 16 : 8;
    int iteration2 = ((nrows + unrollNm2 - 1) / unrollNm2) * unrollNm2;

    int cnt = 0;
    const int sizeT = sizeof(float);
    const int widthT = sizeof(float) * 8;
    for (int i = 0; i < iteration2; ++i) {
        f_cast<float> tt;
        uint32_t tmp11 = resultinPong ? buffPong[i] : buffPing[i]; // pagerank1[i];
        if (cnt < nrows) {
            tt.i = tmp11;
            if (sizeT == 8) {
                pagerank[cnt] = (float)(tt.f);
            } else {
                pagerank[cnt] = (float)(tt.f);
            }
            cnt++;
        }
    }
};

int opPageRank::compute(unsigned int deviceID,
                        unsigned int cuID,
                        unsigned int channelID,
                        xrmContext* ctx,
                        xrmCuResource* resR,
                        std::string instanceName,
                        clHandle* handles,
                        float alpha,
                        float tolerance,
                        int maxIter,
                        xf::graph::Graph<uint32_t, float> g,
                        float* pagerank) {
    clHandle* hds = &handles[channelID + cuID * dupNmPG + deviceID * dupNmPG * cuPerBoardPG];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    int nrows = g.nodeNum;
    unsigned int num_runs = 1;
    uint32_t* cntValFull;
    uint32_t* buffPing;
    uint32_t* buffPong;
    int* resultInfo;
    uint32_t* degreeCSR;
    uint32_t* orderUnroll;

    int nnz = g.edgeNum;

    int unrollNm2 = (sizeof(float) == 4) ? 16 : 8;
    int iteration2 = ((nrows + unrollNm2 - 1) / unrollNm2) * unrollNm2;
    cntValFull = aligned_alloc<uint32_t>(iteration2);
    buffPing = aligned_alloc<uint32_t>(iteration2);
    buffPong = aligned_alloc<uint32_t>(iteration2);
    resultInfo = aligned_alloc<int>(2);
    int depthDegree = (nrows + 16 + 15) / 16;
    int sizeDegree = depthDegree * 16;
    int depthOrder = (nrows + 16 + 7) / 8;
    int sizeOrder = depthOrder * 8;
    degreeCSR = aligned_alloc<uint32_t>(sizeDegree);
    orderUnroll = aligned_alloc<uint32_t>(sizeOrder);

    for (int i = 0; i < nrows; ++i) {
        degreeCSR[i] = 0;
    }

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, nrows, alpha, tolerance, maxIter, num_runs, degreeCSR, cntValFull, buffPing,
               buffPong, resultInfo, orderUnroll, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    postProcess(nrows, resultInfo, buffPing, buffPong, pagerank);

    free(orderUnroll);
    free(cntValFull);
    free(degreeCSR);
    free(buffPing);
    free(buffPong);
    free(resultInfo);

    cuRelease(ctx, resR);

    return ret;
};

event<int> opPageRank::addwork(
    float alpha, float tolerance, int maxIter, xf::graph::Graph<uint32_t, float> g, float* pagerank) {
    return createL3(task_queue[0], &(compute), handles, alpha, tolerance, maxIter, g, pagerank);
};

} // L3
} // graph
} // xf
#endif
