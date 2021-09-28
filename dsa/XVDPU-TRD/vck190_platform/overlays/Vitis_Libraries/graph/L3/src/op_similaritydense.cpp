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

#ifndef _XF_GRAPH_L3_OP_SIMILARITYDENSE_CPP_
#define _XF_GRAPH_L3_OP_SIMILARITYDENSE_CPP_

#include "op_similaritydense.hpp"
#include "xf_utils_sw/logger.hpp"
#include <unordered_map>

namespace xf {
namespace graph {
namespace L3 {

void createHandleSimDense(clHandle& handle, const char* kernelName, const char* pXclbin, int32_t IDDevice) {
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

uint32_t opSimilarityDense::cuPerBoardSimDense;

uint32_t opSimilarityDense::dupNmSimDense;

void opSimilarityDense::setHWInfo(uint32_t numDev, uint32_t CUmax) {
    maxCU = CUmax;
    deviceNm = numDev;
    cuPerBoardSimDense = maxCU / deviceNm;
    handles = new clHandle[CUmax];
};

void opSimilarityDense::freeSimDense() {
    // simDenseThread.join();
    for (unsigned int i = 0; i < maxCU; ++i) {
        delete[] handles[i].buffer;
    }
    delete[] handles;
};

void opSimilarityDense::cuRelease(xrmContext* ctx, xrmCuResource* resR) {
    // while (!xrmCuRelease(ctx, &resR)) {
    while (!xrmCuRelease(ctx, resR)) {
    };
    // std::cout<<"before free cuResource"<<std::endl;
    // free(resR);
    // std::cout<<"after free cuResource"<<std::endl;
};

void opSimilarityDense::init(
    char* kernelName, char* xclbinFile, uint32_t* deviceIDs, uint32_t* cuIDs, unsigned int requestLoad) {
    dupNmSimDense = 100 / requestLoad;
    cuPerBoardSimDense /= dupNmSimDense;
    uint32_t bufferNm = 20;
    unsigned int cnt = 0;
    unsigned int* handleID = new unsigned int[maxCU];
    handleID[0] = cnt;
    handles[0].deviceID = deviceIDs[0];
    handles[0].cuID = cuIDs[0];
    handles[0].dupID = 0;
    std::thread th[maxCU];
    createHandleSimDense(handles[cnt], kernelName, xclbinFile, deviceIDs[cnt]);
    handles[cnt].buffer = new cl::Buffer[bufferNm];
    unsigned int prev = deviceIDs[0];
    deviceOffset.push_back(0);
    for (unsigned int i = 1; i < maxCU; ++i) {
        handles[i].deviceID = deviceIDs[i];
        handles[i].cuID = cuIDs[i];
        handles[i].dupID = i % dupNmSimDense;
        createHandleSimDense(handles[i], kernelName, xclbinFile, deviceIDs[i]);
        handles[i].buffer = new cl::Buffer[bufferNm];
        if (deviceIDs[i] != prev) {
            prev = deviceIDs[i];
            deviceOffset.push_back(i);
        }
    }
    delete[] handleID;
}

void opSimilarityDense::initInt(char* kernelName,
                                char* xclbinFile,
                                char* xclbinFile2,
                                uint32_t* deviceIDs,
                                uint32_t* cuIDs,
                                unsigned int requestLoad) {
    dupNmSimDense = 100 / requestLoad;
    cuPerBoardSimDense /= dupNmSimDense;
    uint32_t bufferNm = 20;
    unsigned int cnt = 0;
    unsigned int* handleID = new unsigned int[maxCU];
    handleID[0] = cnt;
    handles[0].deviceID = deviceIDs[0];
    handles[0].cuID = cuIDs[0];
    handles[0].dupID = 0;
    std::thread th[maxCU];
    createHandleSimDense(handles[cnt], kernelName, xclbinFile, deviceIDs[cnt]);
    handles[cnt].buffer = new cl::Buffer[bufferNm];
    unsigned int prev = deviceIDs[0];
    deviceOffset.push_back(0);
    for (unsigned int i = 1; i < maxCU; ++i) {
        handles[i].deviceID = deviceIDs[i];
        handles[i].cuID = cuIDs[i];
        handles[i].dupID = i % dupNmSimDense;
        if (deviceIDs[i] == 1) {
            createHandleSimDense(handles[i], kernelName, xclbinFile2, deviceIDs[i]);
        } else {
            createHandleSimDense(handles[i], kernelName, xclbinFile, deviceIDs[i]);
        }
        handles[i].buffer = new cl::Buffer[bufferNm];
        if (deviceIDs[i] != prev) {
            prev = deviceIDs[i];
            deviceOffset.push_back(i);
        }
    }
    delete[] handleID;
}

void opSimilarityDense::migrateMemObj(clHandle* hds,
                                      bool type,
                                      unsigned int num_runs,
                                      std::vector<cl::Memory>& ob,
                                      std::vector<cl::Event>* evIn,
                                      cl::Event* evOut) {
    for (unsigned int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueMigrateMemObjects(ob, type, evIn, evOut); // 0 : migrate from host to dev
    }
};

void loadGraphCoreSimDense(clHandle* hds, int nrows, int nnz, xf::graph::Graph<uint32_t, float> g) {
    cl::Device device = hds[0].device;
    cl::Context context = hds[0].context;
    cl::CommandQueue q = hds[0].q;
    uint32_t splitNm = g.splitNum;
    uint32_t CHANNEL_NUMBER = 8;
    // declare map of host buffers
    std::vector<cl_mem_ext_ptr_t> mext_o(4 * splitNm);
    for (int i = 0; i < splitNm; i++) {
        mext_o[4 * i + 0] = {(uint32_t)(8 * i) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i], 0};
        mext_o[4 * i + 1] = {(uint32_t)(8 * i + 1) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i + 1], 0};
        mext_o[4 * i + 2] = {(uint32_t)(8 * i + 2) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i + 2], 0};
        mext_o[4 * i + 3] = {(uint32_t)(8 * i + 3) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i + 3], 0};
    }

    // declare cl::buffers
    for (int i = 0; i < 4 * splitNm; i++) {
        int sizeW = g.numVerticesPU[i / 4] * g.edgeNum;
        hds[0].buffer[2 + i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                          sizeof(uint32_t) * (sizeW + CHANNEL_NUMBER), &mext_o[i]);
    }

    // add buffers to migrate
    std::vector<cl::Memory> init;
    std::vector<cl::Memory> ob_in;
    for (int i = 0; i < 4 * splitNm; i++) {
        init.push_back(hds[0].buffer[2 + i]);
        ob_in.push_back(hds[0].buffer[2 + i]);
    }

    std::vector<cl::Event> eventFirst(1);
    std::vector<cl::Event> eventSecond(1);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &eventFirst[0]);

    q.enqueueMigrateMemObjects(ob_in, 0, &eventFirst, &eventSecond[0]); // 0 : migrate from host to dev

    eventSecond[0].wait();
};

void loadGraphCoreSimDenseInt(clHandle* hds, int nrows, int nnz, int cuID, xf::graph::Graph<int32_t, int32_t> g) {
    cl::Device device = hds[0].device;
    cl::Context context = hds[0].context;
    cl::CommandQueue q = hds[0].q;
    uint32_t splitNm = g.splitNum;
    uint32_t CHANNEL_NUMBER = 16;

    // declare map of host buffers
    std::vector<cl_mem_ext_ptr_t> mext_o(4 * splitNm);
    for (unsigned int i = 0; i < splitNm; i++) {
        if (cuID == 0) {
            mext_o[4 * i + 0] = {(uint32_t)(8 * i) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i], 0};
            mext_o[4 * i + 1] = {(uint32_t)(8 * i + 1) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i + 1], 0};
            mext_o[4 * i + 2] = {(uint32_t)(8 * i + 2) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i + 2], 0};
            mext_o[4 * i + 3] = {(uint32_t)(8 * i + 3) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i + 3], 0};
        } else {
            mext_o[4 * i + 0] = {(uint32_t)(8 * i + 4) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i], 0};
            mext_o[4 * i + 1] = {(uint32_t)(8 * i + 5) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i + 1], 0};
            mext_o[4 * i + 2] = {(uint32_t)(8 * i + 6) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i + 2], 0};
            mext_o[4 * i + 3] = {(uint32_t)(8 * i + 7) | XCL_MEM_TOPOLOGY, g.weightsDense[4 * i + 3], 0};
        }
    }

    // declare cl::buffers
    int edgeAlign8 = ((g.edgeNum + CHANNEL_NUMBER - 1) / CHANNEL_NUMBER) * CHANNEL_NUMBER;
    for (unsigned int i = 0; i < splitNm; i++) {
        int sizeW = (g.numVerticesPU[i] + 3) / 4 * edgeAlign8;

        hds[0].buffer[2 + 4 * i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                              sizeof(uint32_t) * (sizeW + CHANNEL_NUMBER), &mext_o[4 * i]);
        hds[0].buffer[2 + 4 * i + 1] =
            cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       sizeof(uint32_t) * (sizeW + CHANNEL_NUMBER), &mext_o[4 * i + 1]);
        hds[0].buffer[2 + 4 * i + 2] =
            cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       sizeof(uint32_t) * (sizeW + CHANNEL_NUMBER), &mext_o[4 * i + 2]);
        hds[0].buffer[2 + 4 * i + 3] =
            cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       sizeof(uint32_t) * (sizeW + CHANNEL_NUMBER), &mext_o[4 * i + 3]);
    }

    // add buffers to migrate
    std::vector<cl::Memory> init;
    std::vector<cl::Memory> ob_in;
    for (unsigned int i = 0; i < 4 * splitNm; i++) {
        init.push_back(hds[0].buffer[2 + i]);
        ob_in.push_back(hds[0].buffer[2 + i]);
    }

    std::vector<cl::Event> eventFirst(1);
    std::vector<cl::Event> eventSecond(1);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &eventFirst[0]);

    q.enqueueMigrateMemObjects(ob_in, 0, &eventFirst, &eventSecond[0]); // 0 : migrate from host to dev

    eventSecond[0].wait();
};

void opSimilarityDense::loadGraph(xf::graph::Graph<uint32_t, float> g) {
    int nnz = g.edgeNum;
    int nrows = g.nodeNum;
    bool freed[maxCU];

    std::thread* th = new std::thread[maxCU];
    std::future<void>* fut = new std::future<void>[ maxCU ];
    int cnt = 0;
    for (unsigned int j = 0; j < maxCU; ++j) {
        if ((handles[j].cuID == 0) && (handles[j].dupID == 0)) {
            cnt = j;
            std::packaged_task<void(clHandle*, int, int, xf::graph::Graph<uint32_t, float>)> t(loadGraphCoreSimDense);
            fut[j] = t.get_future();
            th[j] = std::thread(std::move(t), &handles[j], nrows, nnz, g);
        }
        freed[j] = 0;
    }
    cnt = 0;
    for (unsigned int j = 0; j < maxCU; ++j) {
        if (!((handles[j].cuID == 0) && (handles[j].dupID == 0))) {
            if (freed[cnt] == 0) {
                fut[cnt].get();
                th[cnt].join();
                freed[cnt] = 1;
            }
            for (unsigned int i = 0; i < (unsigned int)(g.splitNum * 4); i++) {
                handles[j].buffer[2 + i] = handles[cnt].buffer[2 + i];
            }
        } else {
            cnt = j;
        }
    }
    for (unsigned int j = 0; j < maxCU; ++j) {
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

void opSimilarityDense::loadGraphMultiCardBlocking(int deviceID, int cuID, xf::graph::Graph<int32_t, int32_t> g) {
    int nnz = g.edgeNum;
    int nrows = g.nodeNum;
    int cnt = 0;
    for (unsigned int j = 0; j < maxCU; ++j) {
        if ((handles[j].deviceID == (unsigned int)deviceID) && (handles[j].cuID == cuID) && (handles[j].dupID == 0)) {
            cnt = j;
            loadGraphCoreSimDenseInt(&handles[j], nrows, nnz, cuID, g);
        }
    }
    cnt = 0;
    for (unsigned int j = 0; j < maxCU; ++j) {
        if ((handles[j].deviceID == (unsigned int)deviceID) && (handles[j].cuID == (unsigned int)cuID)) {
            if (handles[j].dupID != 0) {
                for (unsigned int i = 0; i < (unsigned int)(g.splitNum * 4); i++) {
                    handles[j].buffer[2 + i] = handles[cnt].buffer[2 + i];
                }
            } else {
                cnt = j;
            }
        }
    }
};

void opSimilarityDense::loadGraphMultiCardNonBlocking(int deviceID, int cuID, xf::graph::Graph<int32_t, int32_t> g) {
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
            std::packaged_task<void(clHandle*, int, int, int, xf::graph::Graph<int32_t, int32_t>)> t(
                loadGraphCoreSimDenseInt);
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
                for (unsigned int i = 0; i < (unsigned int)(g.splitNum * 4); i++) {
                    handles[j].buffer[2 + i] = handles[cnt].buffer[2 + i];
                }
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

void opSimilarityDense::bufferInit(clHandle* hds,
                                   std::string instanceName0,
                                   xf::graph::Graph<uint32_t, float> g,
                                   int similarityType,
                                   int dataType,
                                   uint32_t topK,
                                   unsigned int sourceNUM,
                                   uint32_t* sourceWeight,
                                   uint32_t* config,
                                   uint32_t* resultID,
                                   float* similarity,
                                   cl::Kernel& kernel0,
                                   std::vector<cl::Memory>& ob_in,
                                   std::vector<cl::Memory>& ob_out) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int fail;

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
    kernel0 = cl::Kernel(program, instanceName, &fail);
    logger.logCreateKernel(fail);
    std::cout << "INFO: Kernel has been created" << std::endl;

    uint32_t splitNm = g.splitNum;
    uint32_t CHANNEL_NUMBER = 8;
    uint32_t startID[splitNm];
    uint32_t tmp = 0;
    for (int i = 0; i < splitNm - 1; i++) { // calculate multi PU start address
        startID[i] = tmp;
        tmp += 4 * g.numVerticesPU[i];
    }
    startID[splitNm - 1] = tmp;
    config[0] = topK;
    config[1] = sourceNUM;
    config[2] = similarityType;
    config[3] = dataType;

    for (int j = 0; j < splitNm; j++) {
        config[4 + j] = startID[j];
        config[4 + splitNm + j] = g.numVerticesPU[j];
        config[4 + 2 * splitNm + j] = g.numEdgesPU[j];
    }

    // declare map of host buffers
    std::vector<cl_mem_ext_ptr_t> mext_o(4);
    mext_o[0] = {(uint32_t)(28) | XCL_MEM_TOPOLOGY, sourceWeight, 0};
    mext_o[1] = {(uint32_t)(28) | XCL_MEM_TOPOLOGY, config, 0};
    mext_o[2] = {(uint32_t)(28) | XCL_MEM_TOPOLOGY, resultID, 0};
    mext_o[3] = {(uint32_t)(28) | XCL_MEM_TOPOLOGY, similarity, 0};

    // declare cl::buffers
    hds[0].buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * (sourceNUM + CHANNEL_NUMBER), &mext_o[0]);

    hds[0].buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * 64, &mext_o[1]);

    hds[0].buffer[18] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(uint32_t) * g.nodeNum, &mext_o[2]);

    hds[0].buffer[19] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(float) * g.nodeNum, &mext_o[3]);

    ob_in.push_back(hds[0].buffer[0]);
    ob_in.push_back(hds[0].buffer[1]);

    ob_out.push_back(hds[0].buffer[18]);
    ob_out.push_back(hds[0].buffer[19]);

    // declare events
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    // set kernel args
    kernel0.setArg(0, hds[0].buffer[0]); // config
    kernel0.setArg(1, hds[0].buffer[1]); // source weight
    for (int k = 0; k < 4 * splitNm; k++) {
        kernel0.setArg(2 + k, hds[0].buffer[2 + k]); // weights
    }
    kernel0.setArg(18, hds[0].buffer[18]); // resultID
    kernel0.setArg(19, hds[0].buffer[19]); // similarity

    // launch kernel and calculate kernel execution time
    std::cout << "INFO: Kernel Start" << std::endl;
};

void opSimilarityDense::bufferInitInt(clHandle* hds,
                                      std::string instanceName0,
                                      xf::graph::Graph<int32_t, int32_t> g,
                                      int cuID,
                                      int similarityType,
                                      int dataType,
                                      int32_t topK,
                                      int sourceNUM,
                                      int32_t* sourceWeight,
                                      uint32_t* config,
                                      int32_t* resultID,
                                      float* similarity,
                                      cl::Kernel& kernel0,
                                      std::vector<cl::Memory>& ob_in,
                                      std::vector<cl::Memory>& ob_out) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int fail;

    cl::Device device = hds[0].device;

    instanceName0 = "denseSimilarityKernel:{" + instanceName0 + "}";
    //    if (cuID == 0) {
    //        instanceName0 = "denseSimilarityKernel_0:{" + instanceName0 + "}";
    //    } else {
    //        instanceName0 = "denseSimilarityKernel_1:{" + instanceName0 + "}";
    //    }
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

    int32_t splitNm = g.splitNum;
    int32_t CHANNEL_NUMBER = 16;
    uint32_t startID[splitNm];
    uint32_t tmp = (uint32_t)g.refID;
    for (unsigned int i = 0; i < (unsigned int)(splitNm - 1); i++) { // calculate multi PU start address
        startID[i] = tmp;
        tmp += g.numVerticesPU[i];
    }
    startID[splitNm - 1] = tmp;
    config[0] = topK;
    config[1] = sourceNUM;
    config[2] = similarityType;
    config[3] = dataType;

    int edgeAlign8 = ((g.edgeNum + CHANNEL_NUMBER - 1) / CHANNEL_NUMBER) * CHANNEL_NUMBER;
    for (unsigned int j = 0; j < (unsigned int)splitNm; j++) {
        config[4 + j] = startID[j];
        config[4 + splitNm + j] = (g.numVerticesPU[j] + 3) / 4;
        config[4 + 2 * splitNm + j] = edgeAlign8;
    }

    // declare map of host buffers
    std::vector<cl_mem_ext_ptr_t> mext_o(4);
    if (cuID == 0) {
        mext_o[0] = {(int32_t)(24) | XCL_MEM_TOPOLOGY, sourceWeight, 0};
        mext_o[1] = {(int32_t)(24) | XCL_MEM_TOPOLOGY, config, 0};
        mext_o[2] = {(int32_t)(24) | XCL_MEM_TOPOLOGY, resultID, 0};
        mext_o[3] = {(int32_t)(24) | XCL_MEM_TOPOLOGY, similarity, 0};
    } else {
        mext_o[0] = {(int32_t)(28) | XCL_MEM_TOPOLOGY, sourceWeight, 0};
        mext_o[1] = {(int32_t)(28) | XCL_MEM_TOPOLOGY, config, 0};
        mext_o[2] = {(int32_t)(28) | XCL_MEM_TOPOLOGY, resultID, 0};
        mext_o[3] = {(int32_t)(28) | XCL_MEM_TOPOLOGY, similarity, 0};
    }

    // declare cl::buffers
    hds[0].buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(int32_t) * (sourceNUM + CHANNEL_NUMBER), &mext_o[0]);

    hds[0].buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(int32_t) * 64, &mext_o[1]);

    hds[0].buffer[18] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(int32_t) * topK, &mext_o[2]);

    hds[0].buffer[19] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(float) * topK, &mext_o[3]);

    ob_in.push_back(hds[0].buffer[0]);
    ob_in.push_back(hds[0].buffer[1]);

    ob_out.push_back(hds[0].buffer[18]);
    ob_out.push_back(hds[0].buffer[19]);

    // declare events
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    // set kernel args
    kernel0.setArg(0, hds[0].buffer[0]); // config
    kernel0.setArg(1, hds[0].buffer[1]); // source weight
    for (unsigned int k = 0; k < (unsigned int)(4 * splitNm); k++) {
        kernel0.setArg(2 + k, hds[0].buffer[2 + k]); // weights
    }
    kernel0.setArg(14, hds[0].buffer[18]); // resultID
    kernel0.setArg(15, hds[0].buffer[19]); // similarity

    // launch kernel and calculate kernel execution time
    std::cout << "INFO: Kernel Start" << std::endl;
};

int opSimilarityDense::cuExecute(
    clHandle* hds, cl::Kernel& kernel0, unsigned int num_runs, std::vector<cl::Event>* evIn, cl::Event* evOut) {
    for (unsigned int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueTask(kernel0, evIn, evOut);
    }
    return 0;
}

void opSimilarityDense::postProcessKNN(
    uint32_t topK, std::string* knownLabels, uint32_t* resultID, float* similarity, std::string* label) {
    std::unordered_map<std::string, int> map;
    for (unsigned int i = 0; i < topK; ++i) {
        if (similarity[i] > 0) {
            map[knownLabels[resultID[i]]]++;
        }
    }
    int counter = 0;
    for (auto it = map.begin(); it != map.end(); it++) {
        if (counter < it->second) {
            counter = it->second;
            label[0] = it->first;
        }
    }
};

int opSimilarityDense::compute(unsigned int deviceID,
                               unsigned int cuID,
                               unsigned int channelID,
                               xrmContext* ctx,
                               xrmCuResource* resR,
                               std::string instanceName,
                               clHandle* handles,
                               uint32_t similarityType,
                               uint32_t dataType,
                               unsigned int sourceNUM,
                               uint32_t* sourceWeight,
                               uint32_t topK,
                               xf::graph::Graph<uint32_t, float> g,
                               uint32_t* resultID,
                               float* similarity) {
    clHandle* hds = &handles[channelID + cuID * dupNmSimDense + deviceID * dupNmSimDense * cuPerBoardSimDense];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    uint32_t* config;
    config = aligned_alloc<uint32_t>(64);

    unsigned int num_runs = 1;

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, similarityType, dataType, topK, sourceNUM, sourceWeight, config, resultID,
               similarity, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    cuRelease(ctx, resR);

    free(config);

    return ret;
};

int opSimilarityDense::computeInt(unsigned int deviceID,
                                  unsigned int cuID,
                                  unsigned int channelID,
                                  xrmContext* ctx,
                                  xrmCuResource* resR,
                                  std::string instanceName,
                                  clHandle* handles,
                                  int32_t similarityType,
                                  int32_t dataType,
                                  int32_t sourceNUM,
                                  int32_t* sourceWeight,
                                  int32_t topK,
                                  xf::graph::Graph<int32_t, int32_t> g,
                                  int32_t* resultID,
                                  float* similarity) {
    clHandle* hds = &handles[channelID + cuID * dupNmSimDense + deviceID * dupNmSimDense * cuPerBoardSimDense];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    uint32_t* config;
    config = aligned_alloc<uint32_t>(64);

    unsigned int num_runs = 1;

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInitInt(hds, instanceName, g, cuID, similarityType, dataType, topK, sourceNUM, sourceWeight, config, resultID,
                  similarity, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    cuRelease(ctx, resR);

    free(config);

    return ret;
};

int opSimilarityDense::computeKNN(unsigned int deviceID,
                                  unsigned int cuID,
                                  unsigned int channelID,
                                  xrmContext* ctx,
                                  xrmCuResource* resR,
                                  std::string instanceName,
                                  clHandle* handles,
                                  uint32_t similarityType,
                                  uint32_t dataType,
                                  unsigned int sourceNUM,
                                  uint32_t* sourceWeight,
                                  uint32_t topK,
                                  xf::graph::Graph<uint32_t, float> g,
                                  std::string* knownLabels,
                                  std::string* label) {
    clHandle* hds = &handles[channelID + cuID * dupNmSimDense + deviceID * dupNmSimDense * cuPerBoardSimDense];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    uint32_t* config;
    config = aligned_alloc<uint32_t>(64);
    uint32_t* resultID = aligned_alloc<uint32_t>(topK);
    float* similarity = aligned_alloc<float>(topK);

    unsigned int num_runs = 1;

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, similarityType, dataType, topK, sourceNUM, sourceWeight, config, resultID,
               similarity, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    postProcessKNN(topK, knownLabels, resultID, similarity, label);

    cuRelease(ctx, resR);

    free(config);
    free(resultID);
    free(similarity);

    return ret;
};

int opSimilarityDense::computeAP(unsigned int deviceID,
                                 unsigned int cuID,
                                 unsigned int channelID,
                                 xrmContext* ctx,
                                 xrmCuResource* resR,
                                 std::string instanceName,
                                 clHandle* handles,
                                 uint32_t similarityType,
                                 uint32_t dataType,
                                 uint32_t sourceID,
                                 uint32_t topK,
                                 xf::graph::Graph<uint32_t, float> g,
                                 uint32_t* resultID,
                                 float* similarity) {
    clHandle* hds = &handles[channelID + cuID * dupNmSimDense + deviceID * dupNmSimDense * cuPerBoardSimDense];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    uint32_t* config = aligned_alloc<uint32_t>(64);
    uint32_t numEdges = g.edgeNum;
    uint32_t* sourceWeight = aligned_alloc<uint32_t>(numEdges);
    f_cast<float> tmp;
    uint32_t id = 0;
    uint32_t row = 0;
    uint32_t offset[g.splitNum + 1];
    offset[0] = 0;
    for (unsigned int i = 0; i < g.splitNum; i++) {
        offset[i + 1] = 4 * g.numVerticesPU[i] + offset[i];
        if ((sourceID >= offset[i]) && (sourceID < offset[i + 1])) {
            id = i;
            row = sourceID - offset[i];
        }
    }

    for (unsigned int i = 0; i < numEdges; i++) {
        tmp.f = g.weightsDense[(id * 4 + row) % (g.splitNum * 4)][i];
        sourceWeight[i] = tmp.i;
    }

    unsigned int num_runs = 1;

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, similarityType, dataType, topK, numEdges, sourceWeight, config, resultID,
               similarity, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    cuRelease(ctx, resR);

    free(config);
    free(sourceWeight);

    return ret;
};

int opSimilarityDense::computeAPKNN(unsigned int deviceID,
                                    unsigned int cuID,
                                    unsigned int channelID,
                                    xrmContext* ctx,
                                    xrmCuResource* resR,
                                    std::string instanceName,
                                    clHandle* handles,
                                    uint32_t similarityType,
                                    uint32_t dataType,
                                    uint32_t sourceID,
                                    uint32_t topK,
                                    xf::graph::Graph<uint32_t, float> g,
                                    std::string* knownLabels,
                                    std::string* label) {
    clHandle* hds = &handles[channelID + cuID * dupNmSimDense + deviceID * dupNmSimDense * cuPerBoardSimDense];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    uint32_t* config = aligned_alloc<uint32_t>(64);
    uint32_t* resultID = aligned_alloc<uint32_t>(topK);
    float* similarity = aligned_alloc<float>(topK);
    memset(resultID, 0, topK * sizeof(uint32_t));
    memset(similarity, 0, topK * sizeof(float));
    uint32_t numEdges = g.edgeNum;
    uint32_t* sourceWeight = aligned_alloc<uint32_t>(numEdges);
    f_cast<float> tmp;
    uint32_t id = 0;
    uint32_t row = 0;
    uint32_t offset[g.splitNum + 1];
    offset[0] = 0;
    for (unsigned int i = 0; i < g.splitNum; i++) {
        offset[i + 1] = 4 * g.numVerticesPU[i] + offset[i];
        if ((sourceID >= offset[i]) && (sourceID < offset[i + 1])) {
            id = i;
            row = sourceID - offset[i];
        }
    }

    for (unsigned int i = 0; i < numEdges; i++) {
        tmp.f = g.weightsDense[(id * 4 + row) % (g.splitNum * 4)][i];
        sourceWeight[i] = tmp.i;
    }

    unsigned int num_runs = 1;

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, similarityType, dataType, topK, numEdges, sourceWeight, config, resultID,
               similarity, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    postProcessKNN(topK, knownLabels, resultID, similarity, label);

    cuRelease(ctx, resR);

    free(config);
    free(resultID);
    free(similarity);
    free(sourceWeight);

    return ret;
};

event<int> opSimilarityDense::addwork(uint32_t similarityType,
                                      uint32_t dataType,
                                      uint32_t sourceNUM,
                                      uint32_t* sourceWeight,
                                      uint32_t topK,
                                      xf::graph::Graph<uint32_t, float> g,
                                      uint32_t* resultID,
                                      float* similarity) {
    return createL3(task_queue[0], &(compute), handles, similarityType, dataType, sourceNUM, sourceWeight, topK, g,
                    resultID, similarity);
};

event<int> opSimilarityDense::addworkInt(int32_t similarityType,
                                         int32_t dataType,
                                         int32_t sourceNUM,
                                         int32_t* sourceWeight,
                                         int32_t topK,
                                         xf::graph::Graph<int32_t, int32_t> g,
                                         int32_t* resultID,
                                         float* similarity) {
    return createL3(task_queue[0], &(computeInt), handles, similarityType, dataType, sourceNUM, sourceWeight, topK, g,
                    resultID, similarity);
};

event<int> opSimilarityDense::addworkKNN(uint32_t similarityType,
                                         uint32_t dataType,
                                         uint32_t sourceNUM,
                                         uint32_t* sourceWeight,
                                         uint32_t topK,
                                         xf::graph::Graph<uint32_t, float> g,
                                         std::string* knownLabels,
                                         std::string& label) {
    return createL3(task_queue[0], &(computeKNN), handles, similarityType, dataType, sourceNUM, sourceWeight, topK, g,
                    knownLabels, &label);
};

event<int> opSimilarityDense::addworkAP(uint32_t similarityType,
                                        uint32_t dataType,
                                        uint32_t sourceID,
                                        uint32_t topK,
                                        xf::graph::Graph<uint32_t, float> g,
                                        uint32_t* resultID,
                                        float* similarity) {
    return createL3(task_queue[0], &(computeAP), handles, similarityType, dataType, sourceID, topK, g, resultID,
                    similarity);
};

event<int> opSimilarityDense::addworkAPKNN(uint32_t similarityType,
                                           uint32_t dataType,
                                           uint32_t sourceID,
                                           uint32_t topK,
                                           xf::graph::Graph<uint32_t, float> g,
                                           std::string* knownLabels,
                                           std::string& label) {
    return createL3(task_queue[0], &(computeAPKNN), handles, similarityType, dataType, sourceID, topK, g, knownLabels,
                    &label);
};

} // L3
} // graph
} // xf
#endif
