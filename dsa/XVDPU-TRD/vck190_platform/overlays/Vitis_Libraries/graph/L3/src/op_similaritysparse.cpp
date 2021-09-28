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

#ifndef _XF_GRAPH_L3_OP_SIMILARITYSPARSE_CPP_
#define _XF_GRAPH_L3_OP_SIMILARITYSPARSE_CPP_

#include "op_similaritysparse.hpp"
#include <unordered_map>

namespace xf {
namespace graph {
namespace L3 {

void createHandleSimSparse(clHandle& handle, const char* kernelName, const char* pXclbin, int32_t IDDevice) {
    // Platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    handle.device = devices[IDDevice];
    handle.context = cl::Context(handle.device);
    handle.q = cl::CommandQueue(handle.context, handle.device,
                                CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    std::string devName = handle.device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    handle.xclBins = xcl::import_binary_file(pXclbin);
    std::vector<cl::Device> devices2;
    devices2.push_back(handle.device);
    handle.program = cl::Program(handle.context, devices2, handle.xclBins);
}

uint32_t opSimilaritySparse::cuPerBoardSimSparse;

uint32_t opSimilaritySparse::dupNmSimSparse;

void opSimilaritySparse::setHWInfo(uint32_t numDev, uint32_t CUmax) {
    maxCU = CUmax;
    deviceNm = numDev;
    cuPerBoardSimSparse = maxCU / deviceNm;
    handles = new clHandle[CUmax];
};

void opSimilaritySparse::freeSimSparse() {
    simSparseThread.join();
    for (int i = 0; i < maxCU; ++i) {
        delete[] handles[i].buffer;
    }
    delete[] handles;
};

void opSimilaritySparse::cuRelease(xrmContext* ctx, xrmCuResource* resR) {
    while (!xrmCuRelease(ctx, resR)) {
    };
    free(resR);
};

void opSimilaritySparse::init(
    char* kernelName, char* xclbinFile, uint32_t* deviceIDs, uint32_t* cuIDs, unsigned int requestLoad) {
    dupNmSimSparse = 100 / requestLoad;
    cuPerBoardSimSparse /= dupNmSimSparse;
    uint32_t bufferNm = 29;
    unsigned int cnt = 0;
    unsigned int cntCU = 0;
    unsigned int* handleID = new unsigned int[maxCU];
    handleID[0] = cnt;
    handles[0].deviceID = deviceIDs[0];
    handles[0].cuID = cuIDs[0];
    handles[0].dupID = 0;
    std::thread th[maxCU];
    // th[0] = std::thread(&createHandleSim, std::ref(handles[cnt]), kernelName, xclbinFile, deviceIDs[cnt]);
    createHandleSimSparse(handles[cnt], kernelName, xclbinFile, deviceIDs[cnt]);
    handles[cnt].buffer = new cl::Buffer[bufferNm];
    unsigned int prev = deviceIDs[0];
    unsigned int prevCU = cuIDs[0];
    deviceOffset.push_back(0);
    for (int i = 1; i < maxCU; ++i) {
        handles[i].deviceID = deviceIDs[i];
        handles[i].cuID = cuIDs[i];
        handles[i].dupID = i % dupNmSimSparse;
        // th[i] = std::thread(&createHandleSim, std::ref(handles[i]), kernelName, xclbinFile, deviceIDs[i]);
        createHandleSimSparse(handles[i], kernelName, xclbinFile, deviceIDs[i]);
        handles[i].buffer = new cl::Buffer[bufferNm];
        if (deviceIDs[i] != prev) {
            prev = deviceIDs[i];
            deviceOffset.push_back(i);
        }
    }
    // for (int j = 0; j < maxCU; ++j) {
    //     th[j].join();
    // }
    delete[] handleID;
}

void opSimilaritySparse::migrateMemObj(clHandle* hds,
                                       bool type,
                                       unsigned int num_runs,
                                       std::vector<cl::Memory>& ob,
                                       std::vector<cl::Event>* evIn,
                                       cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueMigrateMemObjects(ob, type, evIn, evOut); // 0 : migrate from host to dev
    }
};

void loadGraphCoreSimSparse(clHandle* hds, int nrows, int nnz, xf::graph::Graph<uint32_t, float> g) {
    cl::Device device = hds[0].device;
    cl::Context context = hds[0].context;
    cl::CommandQueue q = hds[0].q;
    uint32_t splitNm = g.splitNum;
    uint32_t CHANNEL_NUMBER = 4;
    // declare map of host buffers
    std::vector<cl_mem_ext_ptr_t> mext_o(3 * splitNm);
    for (int i = 0; i < splitNm; i++) {
        mext_o[3 * i + 0] = {(uint32_t)(3 * i) | XCL_MEM_TOPOLOGY, g.offsetsSplitted[i], 0};
        mext_o[3 * i + 1] = {(uint32_t)(3 * i + 1) | XCL_MEM_TOPOLOGY, g.indicesSplitted[i], 0};
        mext_o[3 * i + 2] = {(uint32_t)(3 * i + 2) | XCL_MEM_TOPOLOGY, g.weightsSplitted[i], 0};
    }

    // declare cl::buffers
    for (int i = 0; i < splitNm; i++) {
        hds[0].buffer[3 * i + 3] =
            cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       sizeof(uint32_t) * (g.numVerticesPU[i] + CHANNEL_NUMBER), &mext_o[3 * i + 0]);
        hds[0].buffer[3 * i + 4] =
            cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                       sizeof(uint32_t) * (g.numEdgesPU[i] + CHANNEL_NUMBER), &mext_o[3 * i + 1]);
        hds[0].buffer[3 * i + 5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                              sizeof(float) * (g.numEdgesPU[i] + CHANNEL_NUMBER), &mext_o[3 * i + 2]);
    }

    // add buffers to migrate
    std::vector<cl::Memory> init;
    std::vector<cl::Memory> ob_in;
    for (int i = 0; i < splitNm; i++) {
        init.push_back(hds[0].buffer[3 * i + 3]);
        init.push_back(hds[0].buffer[3 * i + 4]);
        init.push_back(hds[0].buffer[3 * i + 5]);
        ob_in.push_back(hds[0].buffer[3 * i + 3]);
        ob_in.push_back(hds[0].buffer[3 * i + 4]);
        ob_in.push_back(hds[0].buffer[3 * i + 5]);
    }

    std::vector<cl::Event> eventFirst(1);
    std::vector<cl::Event> eventSecond(1);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, &eventFirst[0]);

    q.enqueueMigrateMemObjects(ob_in, 0, &eventFirst, &eventSecond[0]); // 0 : migrate from host to dev

    eventSecond[0].wait();
};

void opSimilaritySparse::loadGraph(xf::graph::Graph<uint32_t, float> g) {
    int nnz = g.edgeNum;
    int nrows = g.nodeNum;
    bool freed[maxCU];

    std::thread* th = new std::thread[maxCU];
    std::future<void>* fut = new std::future<void>[ maxCU ];
    int cnt = 0;
    for (int j = 0; j < maxCU; ++j) {
        if ((handles[j].cuID == 0) && (handles[j].dupID == 0)) {
            cnt = j;
            std::packaged_task<void(clHandle*, int, int, xf::graph::Graph<uint32_t, float>)> t(loadGraphCoreSimSparse);
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
            for (int i = 0; i < g.splitNum * 4; i++) {
                handles[j].buffer[3 + i] = handles[cnt].buffer[3 + i];
            }
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

void opSimilaritySparse::bufferInit(clHandle* hds,
                                    std::string instanceName0,
                                    xf::graph::Graph<uint32_t, float> g,
                                    int similarityType,
                                    int dataType,
                                    uint32_t topK,
                                    unsigned int sourceNUM,
                                    uint32_t* sourceIndice,
                                    uint32_t* sourceWeight,
                                    uint32_t* config,
                                    uint32_t* resultID,
                                    float* similarity,
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
    kernel0 = cl::Kernel(program, instanceName);
    std::cout << "INFO: Kernel has been created" << std::endl;

    uint32_t splitNm = g.splitNum;
    uint32_t CHANNEL_NUMBER = 4;
    uint32_t startID[splitNm];
    uint32_t tmp = 0;
    for (int i = 0; i < splitNm - 1; i++) { // calculate multi PU start address
        startID[i] = tmp;
        tmp += g.numVerticesPU[i];
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
    std::vector<cl_mem_ext_ptr_t> mext_o(3 + 2);
    mext_o[0] = {(uint32_t)(24) | XCL_MEM_TOPOLOGY, sourceIndice, 0};
    mext_o[1] = {(uint32_t)(24) | XCL_MEM_TOPOLOGY, sourceWeight, 0};

    mext_o[2] = {(uint32_t)(24) | XCL_MEM_TOPOLOGY, config, 0};
    mext_o[3] = {(uint32_t)(24) | XCL_MEM_TOPOLOGY, resultID, 0};
    mext_o[4] = {(uint32_t)(24) | XCL_MEM_TOPOLOGY, similarity, 0};

    // declare cl::buffers

    hds[0].buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * (sourceNUM + CHANNEL_NUMBER), &mext_o[0]);

    hds[0].buffer[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * (sourceNUM + CHANNEL_NUMBER), &mext_o[1]);

    hds[0].buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  sizeof(uint32_t) * 64, &mext_o[2]);

    hds[0].buffer[3 * splitNm + 3] = cl::Buffer(
        context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(uint32_t) * topK, &mext_o[3]);

    hds[0].buffer[3 * splitNm + 4] = cl::Buffer(
        context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float) * topK, &mext_o[4]);

    ob_in.push_back(hds[0].buffer[0]);
    ob_in.push_back(hds[0].buffer[1]);
    ob_in.push_back(hds[0].buffer[2]);

    ob_out.push_back(hds[0].buffer[3 * splitNm + 3]);
    ob_out.push_back(hds[0].buffer[3 * splitNm + 4]);

    // declare events
    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(1);
    std::vector<cl::Event> events_read(1);

    // set kernel args
    kernel0.setArg(0, hds[0].buffer[0]); // config
    kernel0.setArg(1, hds[0].buffer[1]); // sourceIndice
    kernel0.setArg(2, hds[0].buffer[2]); // sourceWeight
    for (int k = 0; k < splitNm; k++) {
        kernel0.setArg(3 * k + 3, hds[0].buffer[3 * k + 3]); // offsets
        kernel0.setArg(3 * k + 4, hds[0].buffer[3 * k + 4]); // indices
        kernel0.setArg(3 * k + 5, hds[0].buffer[3 * k + 5]); // weights
    }
    kernel0.setArg(3 * splitNm + 3, hds[0].buffer[3 * splitNm + 3]); // resultID
    kernel0.setArg(3 * splitNm + 4, hds[0].buffer[3 * splitNm + 4]); // similarity

    // launch kernel and calculate kernel execution time
    std::cout << "INFO: Kernel Start" << std::endl;
};

int opSimilaritySparse::cuExecute(
    clHandle* hds, cl::Kernel& kernel0, unsigned int num_runs, std::vector<cl::Event>* evIn, cl::Event* evOut) {
    for (int i = 0; i < num_runs; ++i) {
        hds[0].q.enqueueTask(kernel0, evIn, evOut);
    }
    return 0;
}

void opSimilaritySparse::postProcessKNN(
    uint32_t topK, std::string* knownLabels, uint32_t* resultID, float* similarity, std::string* label) {
    std::unordered_map<std::string, int> map;
    for (int i = 0; i < topK; ++i) {
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

int opSimilaritySparse::compute(unsigned int deviceID,
                                unsigned int cuID,
                                unsigned int channelID,
                                xrmContext* ctx,
                                xrmCuResource* resR,
                                std::string instanceName,
                                clHandle* handles,
                                uint32_t similarityType,
                                uint32_t dataType,
                                unsigned int sourceNUM,
                                uint32_t* sourceIndice,
                                uint32_t* sourceWeight,
                                uint32_t topK,
                                xf::graph::Graph<uint32_t, float> g,
                                uint32_t* resultID,
                                float* similarity) {
    clHandle* hds = &handles[channelID + cuID * dupNmSimSparse + deviceID * dupNmSimSparse * cuPerBoardSimSparse];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    uint32_t* config;
    config = aligned_alloc<uint32_t>(64);

    unsigned int num_runs = 1;

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, similarityType, dataType, topK, sourceNUM, sourceIndice, sourceWeight, config,
               resultID, similarity, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    cuRelease(ctx, resR);

    free(config);

    return ret;
};

int opSimilaritySparse::computeKNN(unsigned int deviceID,
                                   unsigned int cuID,
                                   unsigned int channelID,
                                   xrmContext* ctx,
                                   xrmCuResource* resR,
                                   std::string instanceName,
                                   clHandle* handles,
                                   uint32_t similarityType,
                                   uint32_t dataType,
                                   unsigned int sourceNUM,
                                   uint32_t* sourceIndice,
                                   uint32_t* sourceWeight,
                                   uint32_t topK,
                                   xf::graph::Graph<uint32_t, float> g,
                                   std::string* knownLabels,
                                   std::string* label) {
    clHandle* hds = &handles[channelID + cuID * dupNmSimSparse + deviceID * dupNmSimSparse * cuPerBoardSimSparse];
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

    bufferInit(hds, instanceName, g, similarityType, dataType, topK, sourceNUM, sourceIndice, sourceWeight, config,
               resultID, similarity, kernel0, ob_in, ob_out);

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

int opSimilaritySparse::computeAP(unsigned int deviceID,
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
    clHandle* hds = &handles[channelID + cuID * dupNmSimSparse + deviceID * dupNmSimSparse * cuPerBoardSimSparse];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    uint32_t splitNm = g.splitNum;
    uint32_t* config = aligned_alloc<uint32_t>(64);
    uint32_t sourceNUM;
    uint32_t* sourceIndice;
    uint32_t* sourceWeight;
    f_cast<float> tmp0;
    uint32_t cnt = 0;
    uint32_t prev = 0;
    uint32_t cur;
    uint32_t offset = 0;
    for (int i = 0; i < splitNm; ++i) {
        cur = prev + g.numVerticesPU[i];
        if ((sourceID < cur) && (sourceID >= prev)) {
            uint32_t start = g.offsetsSplitted[i][sourceID - prev];
            uint32_t end = g.offsetsSplitted[i][sourceID - prev + 1];
            sourceNUM = end - start;
            sourceIndice = aligned_alloc<uint32_t>(sourceNUM);
            sourceWeight = aligned_alloc<uint32_t>(sourceNUM);
            for (int k = 0; k < (end - start); ++k) {
                sourceIndice[k] = g.indicesSplitted[i][start - offset + k];
                tmp0.f = g.weightsSplitted[i][start - offset + k];
                sourceWeight[k] = tmp0.i;
            }
        }
        prev = cur;
        offset += g.numEdgesPU[i];
    }

    unsigned int num_runs = 1;

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, similarityType, dataType, topK, sourceNUM, sourceIndice, sourceWeight, config,
               resultID, similarity, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    cuRelease(ctx, resR);

    free(config);
    free(sourceIndice);
    free(sourceWeight);

    return ret;
};

int opSimilaritySparse::computeAPKNN(unsigned int deviceID,
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
    clHandle* hds = &handles[channelID + cuID * dupNmSimSparse + deviceID * dupNmSimSparse * cuPerBoardSimSparse];
    cl::Kernel kernel0;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    uint32_t* config;
    config = aligned_alloc<uint32_t>(64);
    uint32_t* resultID = aligned_alloc<uint32_t>(topK);
    float* similarity = aligned_alloc<float>(topK);
    memset(resultID, 0, topK * sizeof(uint32_t));
    memset(similarity, 0, topK * sizeof(float));

    uint32_t splitNm = g.splitNum;
    uint32_t sourceNUM;
    uint32_t* sourceIndice;
    uint32_t* sourceWeight;
    f_cast<float> tmp0;
    uint32_t cnt = 0;
    uint32_t prev = 0;
    uint32_t cur;
    uint32_t offset = 0;
    for (int i = 0; i < splitNm; ++i) {
        cur = prev + g.numVerticesPU[i];
        if ((sourceID < cur) && (sourceID >= prev)) {
            uint32_t start = g.offsetsSplitted[i][sourceID - prev];
            uint32_t end = g.offsetsSplitted[i][sourceID - prev + 1];
            sourceNUM = end - start;
            sourceIndice = aligned_alloc<uint32_t>(sourceNUM);
            sourceWeight = aligned_alloc<uint32_t>(sourceNUM);
            for (int k = 0; k < (end - start); ++k) {
                sourceIndice[k] = g.indicesSplitted[i][start - offset + k];
                tmp0.f = g.weightsSplitted[i][start - offset + k];
                sourceWeight[k] = tmp0.i;
            }
        }
        prev = cur;
        offset += g.numEdgesPU[i];
    }

    unsigned int num_runs = 1;

    std::vector<cl::Event> events_write(1);
    std::vector<cl::Event> events_kernel(num_runs);
    std::vector<cl::Event> events_read(1);

    bufferInit(hds, instanceName, g, similarityType, dataType, topK, sourceNUM, sourceIndice, sourceWeight, config,
               resultID, similarity, kernel0, ob_in, ob_out);

    migrateMemObj(hds, 0, num_runs, ob_in, nullptr, &events_write[0]);

    int ret = cuExecute(hds, kernel0, num_runs, &events_write, &events_kernel[0]);

    migrateMemObj(hds, 1, num_runs, ob_out, &events_kernel, &events_read[0]);

    events_read[0].wait();

    postProcessKNN(topK, knownLabels, resultID, similarity, label);

    cuRelease(ctx, resR);

    free(config);
    free(resultID);
    free(similarity);
    free(sourceIndice);
    free(sourceWeight);

    return ret;
};

event<int> opSimilaritySparse::addwork(uint32_t similarityType,
                                       uint32_t dataType,
                                       uint32_t sourceNUM,
                                       uint32_t* sourceIndice,
                                       uint32_t* sourceWeight,
                                       uint32_t topK,
                                       xf::graph::Graph<uint32_t, float> g,
                                       uint32_t* resultID,
                                       float* similarity) {
    return createL3(task_queue[0], &(compute), handles, similarityType, dataType, sourceNUM, sourceIndice, sourceWeight,
                    topK, g, resultID, similarity);
};

event<int> opSimilaritySparse::addworkKNN(uint32_t similarityType,
                                          uint32_t dataType,
                                          uint32_t sourceNUM,
                                          uint32_t* sourceIndice,
                                          uint32_t* sourceWeight,
                                          uint32_t topK,
                                          xf::graph::Graph<uint32_t, float> g,
                                          std::string* knownLabels,
                                          std::string& label) {
    return createL3(task_queue[0], &(computeKNN), handles, similarityType, dataType, sourceNUM, sourceIndice,
                    sourceWeight, topK, g, knownLabels, &label);
};

event<int> opSimilaritySparse::addworkAP(uint32_t similarityType,
                                         uint32_t dataType,
                                         uint32_t sourceID,
                                         uint32_t topK,
                                         xf::graph::Graph<uint32_t, float> g,
                                         uint32_t* resultID,
                                         float* similarity) {
    return createL3(task_queue[0], &(computeAP), handles, similarityType, dataType, sourceID, topK, g, resultID,
                    similarity);
};

event<int> opSimilaritySparse::addworkAPKNN(uint32_t similarityType,
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
