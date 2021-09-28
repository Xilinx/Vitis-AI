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

#ifndef _XF_GRAPH_L3_HANDLE_CPP_
#define _XF_GRAPH_L3_HANDLE_CPP_

#include "xf_graph_L3_handle.hpp"

namespace xf {
namespace graph {
namespace L3 {

void Handle::initOpTwoHop(const char* kernelName,
                          char* xclbinFile,
                          char* kernelAlias,
                          unsigned int requestLoad,
                          unsigned int deviceNeeded,
                          unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    optwohop->setHWInfo(deviceNm, maxCU);
    optwohop->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    optwohop->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpSP(const char* kernelName,
                      char* xclbinFile,
                      char* kernelAlias,
                      unsigned int requestLoad,
                      unsigned int deviceNeeded,
                      unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    opsp->setHWInfo(deviceNm, maxCU);
    opsp->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    opsp->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpTriangleCount(const char* kernelName,
                                 char* xclbinFile,
                                 char* kernelAlias,
                                 unsigned int requestLoad,
                                 unsigned int deviceNeeded,
                                 unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    optcount->setHWInfo(deviceNm, maxCU);
    optcount->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    optcount->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpLabelPropagation(const char* kernelName,
                                    char* xclbinFile,
                                    char* kernelAlias,
                                    unsigned int requestLoad,
                                    unsigned int deviceNeeded,
                                    unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    oplprop->setHWInfo(deviceNm, maxCU);
    oplprop->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    oplprop->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpBFS(const char* kernelName,
                       char* xclbinFile,
                       char* kernelAlias,
                       unsigned int requestLoad,
                       unsigned int deviceNeeded,
                       unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    opbfs->setHWInfo(deviceNm, maxCU);
    opbfs->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    opbfs->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpWCC(const char* kernelName,
                       char* xclbinFile,
                       char* kernelAlias,
                       unsigned int requestLoad,
                       unsigned int deviceNeeded,
                       unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    opwcc->setHWInfo(deviceNm, maxCU);
    opwcc->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    opwcc->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpSCC(const char* kernelName,
                       char* xclbinFile,
                       char* kernelAlias,
                       unsigned int requestLoad,
                       unsigned int deviceNeeded,
                       unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    opscc->setHWInfo(deviceNm, maxCU);
    opscc->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    opscc->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpConvertCsrCsc(const char* kernelName,
                                 char* xclbinFile,
                                 char* kernelAlias,
                                 unsigned int requestLoad,
                                 unsigned int deviceNeeded,
                                 unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    opconvertcsrcsc->setHWInfo(deviceNm, maxCU);
    opconvertcsrcsc->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    opconvertcsrcsc->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpPageRank(const char* kernelName,
                            char* xclbinFile,
                            char* kernelAlias,
                            unsigned int requestLoad,
                            unsigned int deviceNeeded,
                            unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    oppg->setHWInfo(deviceNm, maxCU);
    oppg->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    oppg->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpSimDense(const char* kernelName,
                            char* xclbinFile,
                            char* kernelAlias,
                            unsigned int requestLoad,
                            unsigned int deviceNeeded,
                            unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    opsimdense->setHWInfo(deviceNm, maxCU);
    opsimdense->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    opsimdense->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpSimDenseInt(const char* kernelName,
                               char* xclbinFile,
                               char* xclbinFile2,
                               char* kernelAlias,
                               unsigned int requestLoad,
                               unsigned int deviceNeeded,
                               unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    opsimdense->setHWInfo(deviceNm, maxCU);
    opsimdense->initInt((char*)kernelName, xclbinFile, xclbinFile2, deviceID, cuID, requestLoad);
    opsimdense->initThreadInt(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::initOpSimSparse(const char* kernelName,
                             char* xclbinFile,
                             char* kernelAlias,
                             unsigned int requestLoad,
                             unsigned int deviceNeeded,
                             unsigned int cuPerBoard) {
    uint32_t* deviceID;
    uint32_t* cuID;
    xrm->fetchCuInfo(kernelName, kernelAlias, requestLoad, deviceNm, maxChannelSize, maxCU, &deviceID, &cuID);
    opsimsparse->setHWInfo(deviceNm, maxCU);
    opsimsparse->init((char*)kernelName, xclbinFile, deviceID, cuID, requestLoad);
    opsimsparse->initThread(xrm, kernelName, kernelAlias, requestLoad, deviceNeeded, cuPerBoard);
    delete[] cuID;
    delete[] deviceID;
};

void Handle::addOp(singleOP op) {
    ops.push_back(op);
}

int Handle::setUp() {
    getEnv();
    unsigned int opNm = ops.size();
    unsigned int deviceCounter = 0;
    for (int i = 0; i < opNm; ++i) {
        if (strcmp(ops[i].operationName, "twoHop") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpTwoHop(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad,
                         ops[i].deviceNeeded, ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "pagerank") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;

            initOpPageRank(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad,
                           ops[i].deviceNeeded, ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "shortestPathFloat") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpSP(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad, ops[i].deviceNeeded,
                     ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "similarityDense") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::future<int> th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinAsync(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                auto loadedDevId = th[j].get();
                if (loadedDevId < 0) {
                    std::cout << "ERROR: failed to load " << ops[i].xclbinFile << "(Status=" << loadedDevId
                              << "). Please check if it is "
                              << "created for the Xilinx Acceleration card installed on "
                              << "the server." << std::endl;
                    return loadedDevId;
                }
            }
            deviceCounter += boardNm;
            initOpSimDense(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad,
                           ops[i].deviceNeeded, ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "similarityDenseInt") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                if (j > 0) {
                    th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile2);
                } else {
                    th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
                }
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpSimDenseInt(ops[i].kernelName, ops[i].xclbinFile, ops[i].xclbinFile2, ops[i].kernelAlias,
                              ops[i].requestLoad, ops[i].deviceNeeded, ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "similaritySparse") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpSimSparse(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad,
                            ops[i].deviceNeeded, ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "triangleCount") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpTriangleCount(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad,
                                ops[i].deviceNeeded, ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "labelPropagation") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpLabelPropagation(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad,
                                   ops[i].deviceNeeded, ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "BFS") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpBFS(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad, ops[i].deviceNeeded,
                      ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "WCC") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpWCC(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad, ops[i].deviceNeeded,
                      ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "SCC") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpSCC(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad, ops[i].deviceNeeded,
                      ops[i].cuPerBoard);
        } else if (strcmp(ops[i].operationName, "convertCsrCsc") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            if (deviceCounter + boardNm > numDevices) {
                std::cout << "Error: Need more devices" << std::endl;
                exit(1);
            }
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            std::thread th[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                th[j] = loadXclbinNonBlock(deviceCounter + j, ops[i].xclbinFile);
            }
            for (int j = 0; j < boardNm; ++j) {
                th[j].join();
            }
            deviceCounter += boardNm;
            initOpConvertCsrCsc(ops[i].kernelName, ops[i].xclbinFile, ops[i].kernelAlias, ops[i].requestLoad,
                                ops[i].deviceNeeded, ops[i].cuPerBoard);
        } else {
            std::cout << "Error: the operation " << ops[i].operationName << " is not supported" << std::endl;
            exit(1);
        }
    }
    return 0;
}

void Handle::getEnv() {
    cl_uint platformID = NULL;
    cl_platform_id* platforms = NULL;
    char vendor_name[128] = {0};
    cl_uint num_platforms = 0;
    cl_int err2 = clGetPlatformIDs(0, NULL, &num_platforms);
    if (CL_SUCCESS != err2) {
        std::cout << "INFO: get platform failed" << std::endl;
    }
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    if (NULL == platforms) {
        std::cout << "INFO: allocate platform failed" << std::endl;
    }
    err2 = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (CL_SUCCESS != err2) {
        std::cout << "INFO: get platform failed" << std::endl;
    }
    for (cl_uint ui = 0; ui < num_platforms; ++ui) {
        err2 = clGetPlatformInfo(platforms[ui], CL_PLATFORM_VENDOR, 128 * sizeof(char), vendor_name, NULL);
        if (CL_SUCCESS != err2) {
            std::cout << "INFO: get platform failed" << std::endl;
        } else if (!std::strcmp(vendor_name, "Xilinx")) {
            platformID = ui;
        }
    }
    cl_device_id* devices;
    std::vector<cl::Device> devices0 = xcl::get_xil_devices();
    numDevices = devices0.size();
    devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
    err2 = clGetDeviceIDs(platforms[platformID], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    std::cout << "INFO: num devices = " << numDevices << std::endl;
    size_t valueSize;
    char* value;
    for (int i = 0; i < numDevices; ++i) {
        // print device name
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &valueSize);
        value = new char[valueSize];
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, valueSize, value, NULL);
        printf("INFO: Device %d: %s\n", i, value);
        delete[] value;
    }
}

void Handle::free() {
    unsigned int opNm = ops.size();
    unsigned int deviceCounter = 0;
    for (int i = 0; i < opNm; ++i) {
        if (strcmp(ops[i].operationName, "twoHop") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            optwohop->freeTwoHop();
        } else if (strcmp(ops[i].operationName, "pagerank") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            oppg->freePG();
        } else if (strcmp(ops[i].operationName, "shortestPathFloat") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            opsp->freeSP();
        } else if (strcmp(ops[i].operationName, "similarityDense") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            opsimdense->freeSimDense();
        } else if (strcmp(ops[i].operationName, "similarityDenseInt") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            opsimdense->freeSimDense();
            xrm->freeCuGroup(boardNm);
        } else if (strcmp(ops[i].operationName, "similaritySparse") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            opsimsparse->freeSimSparse();
        } else if (strcmp(ops[i].operationName, "triangleCount") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            optcount->freeTriangleCount();
        } else if (strcmp(ops[i].operationName, "labelPropagation") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            oplprop->freeLabelPropagation();
        } else if (strcmp(ops[i].operationName, "BFS") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            opbfs->freeBFS();
        } else if (strcmp(ops[i].operationName, "WCC") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            opwcc->freeWCC();
        } else if (strcmp(ops[i].operationName, "SCC") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            opscc->freeSCC();
        } else if (strcmp(ops[i].operationName, "convertCsrCsc") == 0) {
            unsigned int boardNm = ops[i].deviceNeeded;
            std::thread thUn[boardNm];
            for (int j = 0; j < boardNm; ++j) {
                thUn[j] = xrm->unloadXclbinNonBlock(deviceCounter + j);
            }
            for (int j = 0; j < boardNm; ++j) {
                thUn[j].join();
            }
            deviceCounter += boardNm;
            opconvertcsrcsc->freeConvertCsrCsc();
        }
    }
    xrm->freeXRM();
};

void Handle::loadXclbin(unsigned int deviceId, char* xclbinName) {
    xrm->loadXclbin(deviceId, xclbinName);
};

std::thread Handle::loadXclbinNonBlock(unsigned int deviceId, char* xclbinName) {
    return xrm->loadXclbinNonBlock(deviceId, xclbinName);
};

std::future<int> Handle::loadXclbinAsync(unsigned int deviceId, char* xclbinName) {
    return xrm->loadXclbinAsync(deviceId, xclbinName);
};

} // L3
} // graph
} // xf
#endif
