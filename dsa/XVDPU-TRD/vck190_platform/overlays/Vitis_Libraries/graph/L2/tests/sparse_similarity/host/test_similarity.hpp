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

#ifndef XF_GRAPH_TEST_SIMILARITY_HPP
#define XF_GRAPH_TEST_SIMILARITY_HPP

#ifndef HLS_TEST
#include "xcl2.hpp"
#endif

#include "utils.hpp"
#include "xf_utils_sw/logger.hpp"

template <int PUNUM>
void generateSourceParams(unsigned int numVerticesPU[PUNUM],
                          unsigned int numEdges,
                          int dataType,
                          int sourceID,
                          ap_uint<32>* offset32[PUNUM],
                          ap_uint<32>* indice32[PUNUM],
                          float* weightSparse[PUNUM],
                          unsigned int& sourceNUM,
                          ap_uint<32>** sourceIndice,
                          ap_uint<32>** sourceWeight) {
    int row0, col0, row1, col1;

    bool flag0 = 0;
    int k;
    int sumV = 0;
    for (k = 0; k < PUNUM; ++k) {
        sumV += numVerticesPU[k];
        if (sourceID < sumV) {
            flag0 = 1;
            break;
        }
    }
    if (flag0) {
        row0 = k;
        row1 = k;

        col0 = sourceID + 1 + numVerticesPU[k] - sumV;
        col1 = sourceID + numVerticesPU[k] - sumV;
    } else {
        row0 = PUNUM - 1;
        row1 = PUNUM - 1;

        col0 = numVerticesPU[PUNUM - 1] + 1;
        col1 = numVerticesPU[PUNUM - 1];
    }
    std::cout << "sourceID=" << sourceID << " offset0=" << offset32[row0][col0] << " offset1=" << offset32[row1][col1]
              << std::endl;
    std::cout << "row0=" << row0 << " col0=" << col0 << " row1=" << row1 << " col1=" << col1 << std::endl;
    sourceNUM = (unsigned int)(offset32[row0][col0] - offset32[row1][col1]);
    *sourceIndice = aligned_alloc<ap_uint<32> >(sourceNUM);
    *sourceWeight = aligned_alloc<ap_uint<32> >(sourceNUM);

    int ind = offset32[row1][col1] - offset32[row1][0];
    for (int i = 0; i < sourceNUM; i++) {
        sourceIndice[0][i] = (ap_uint<32>)indice32[row1][ind + i];
        sourceWeight[0][i] = floatToBits<float, uint32_t>(weightSparse[row1][ind + i]);

        std::cout << "sourceWeight[" << i << "]=" << sourceWeight[0][i]
                  << " weightSparse=" << weightSparse[row1][ind + i] << std::endl;
    }
}

template <int PUNUM>
void generateSourceParams(unsigned int numVerticesPU[PUNUM],
                          unsigned int numEdges,
                          int dataType,
                          int sourceID,
                          float* weightDense[4 * PUNUM],
                          unsigned int& sourceNUM,
                          ap_uint<32>** sourceWeight) {
    sourceNUM = (unsigned int)numEdges;
    *sourceWeight = aligned_alloc<ap_uint<32> >(numEdges);

    unsigned int id, row;
    unsigned int offset[4 * PUNUM + 1];
    offset[0] = 0;
    for (int i = 0; i < 4 * PUNUM; i++) {
        offset[i + 1] = numVerticesPU[i / 4] + offset[i];
        if ((sourceID >= offset[i]) && (sourceID < offset[i + 1])) {
            id = i;
            row = sourceID - offset[i];
        }
    }

    std::cout << "id =" << id << " row=" << row << std::endl;
    for (int i = 0; i < sourceNUM; i++) {
        sourceWeight[0][i] = floatToBits<float, uint32_t>(weightDense[id][row * numEdges + i]);

        std::cout << "sourceWeight[" << i << "]=" << sourceWeight[0][i] << std::endl;
    }
}

template <int PUNUM>
int computeSimilarity(std::string xclbinPath,
                      std::string goldenFile,
                      unsigned int numVertices,
                      unsigned int numEdges,
                      int similarityType,
                      int dataType,
                      int sourceID,
                      int sortK,
                      int repInt,
                      unsigned int numVerticesPU[PUNUM],
                      unsigned int numEdgesPU[PUNUM],
                      ap_uint<32>* offset32[PUNUM],
                      ap_uint<32>* indice32[PUNUM],
                      float* weightSparse[PUNUM],
                      unsigned int sourceNUM,
                      ap_uint<32>* sourceIndice,
                      ap_uint<32>* sourceWeight) {
    struct timeval start_time; // End to end time clock start
    gettimeofday(&start_time, 0);

    // output && config////////////////////////////////////////////////////////////////
    std::vector<ap_uint<32>*> config(repInt);
    std::vector<ap_uint<32>*> result_id(repInt);
    std::vector<float*> similarity(repInt);
    unsigned int startID[PUNUM];
    unsigned int tmp = 0;
    for (int i = 0; i < PUNUM - 1; i++) { // calculate multi PU start address
        startID[i] = tmp;
        tmp += numVerticesPU[i];
    }
    startID[PUNUM - 1] = tmp;
    for (int i = 0; i < PUNUM; i++) {
        std::cout << "startID[" << i << "]=" << startID[i] << std::endl;
    }
    for (int i = 0; i < repInt; i++) {
        similarity[i] = aligned_alloc<float>(128);
        result_id[i] = aligned_alloc<ap_uint<32> >(128);
        int base_id = 3;
        config[i] = aligned_alloc<ap_uint<32> >(64);
        config[i][0] = sortK;
        config[i][1] = sourceNUM;
        config[i][2] = similarityType;
        config[i][3] = dataType;

        for (int j = 0; j < PUNUM; j++) {
            config[i][4 + j] = startID[j];
            config[i][4 + PUNUM + j] = numVerticesPU[j];
            config[i][4 + 2 * PUNUM + j] = numEdgesPU[j];
        }
    }
///////////////////////////////////////////////////////////////////////

#ifndef HLS_TEST
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int fail;

    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &fail);
    logger.logCreateContext(fail);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &fail);
    logger.logCreateCommandQueue(fail);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbinPath);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &fail);
    logger.logCreateProgram(fail);

    // create kernels
    std::vector<cl::Kernel> similarity_kernel(repInt);
    for (int i = 0; i < repInt; i++) {
        similarity_kernel[i] = cl::Kernel(program, "sparseSimilarityKernel", &fail);
        logger.logCreateKernel(fail);
    }
    std::cout << "INFO: kernel has been created" << std::endl;

    // declare map of host buffers
    std::vector<cl_mem_ext_ptr_t> mext_o(3 * repInt + 3 * PUNUM + 2);
    for (int i = 0; i < PUNUM; i++) {
        mext_o[3 * i + 0] = {XCL_BANK(3 * i), offset32[i], 0};
        mext_o[3 * i + 1] = {XCL_BANK(3 * i + 1), indice32[i], 0};
        mext_o[3 * i + 2] = {XCL_BANK(3 * i + 2), weightSparse[i], 0};
    }

    mext_o[3 * PUNUM] = {XCL_BANK24, sourceIndice, 0};
    mext_o[3 * PUNUM + 1] = {XCL_BANK24, sourceWeight, 0};

    for (int i = 0; i < repInt; i++) {
        mext_o[3 * PUNUM + 2 + i] = {XCL_BANK24, config[i], 0};
        mext_o[3 * PUNUM + 2 + repInt + i] = {XCL_BANK24, result_id[i], 0};
        mext_o[3 * PUNUM + 2 + 2 * repInt + i] = {XCL_BANK24, similarity[i], 0};
    }

    // create device buffer and map dev buf to host buf
    cl::Buffer offset_buf[PUNUM];
    cl::Buffer indice_buf[PUNUM];
    cl::Buffer weight_buf[PUNUM];
    cl::Buffer empty_buf[PUNUM];
    cl::Buffer source_indice_buf, source_weight_buf;
    std::vector<cl::Buffer> config_buf(repInt);
    std::vector<cl::Buffer> result_id_buf(repInt);
    std::vector<cl::Buffer> similarity_buf(repInt);

    // declare cl::buffers
    for (int i = 0; i < PUNUM; i++) {
        int sizeW = numEdgesPU[i];
        offset_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(ap_uint<32>) * (numVerticesPU[i] + CHANNEL_NUMBER), &mext_o[3 * i + 0]);
        indice_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(ap_uint<32>) * (numEdgesPU[i] + CHANNEL_NUMBER), &mext_o[3 * i + 1]);
        weight_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(ap_uint<32>) * (sizeW + CHANNEL_NUMBER), &mext_o[3 * i + 2]);
    }

    source_indice_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(ap_uint<32>) * (sourceNUM + CHANNEL_NUMBER), &mext_o[3 * PUNUM]);

    source_weight_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(ap_uint<32>) * (sourceNUM + CHANNEL_NUMBER), &mext_o[3 * PUNUM + 1]);

    for (int i = 0; i < repInt; i++) {
        config_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(ap_uint<32>) * 64, &mext_o[3 * PUNUM + 2 + i]);
        result_id_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                      sizeof(ap_uint<32>) * 128, &mext_o[3 * PUNUM + 2 + repInt + i]);
        similarity_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                       sizeof(float) * 128, &mext_o[3 * PUNUM + 2 + 2 * repInt + i]);
    }

    // add buffers to migrate
    std::vector<cl::Memory> init;
    for (int i = 0; i < repInt; i++) {
        init.push_back(config_buf[i]);
    }
    for (int i = 0; i < PUNUM; i++) {
        init.push_back(offset_buf[i]);
        init.push_back(indice_buf[i]);
        init.push_back(weight_buf[i]);
    }
    init.push_back(source_indice_buf);
    init.push_back(source_weight_buf);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);
    q.finish();

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    for (int i = 0; i < repInt; i++) {
        ob_in.push_back(config_buf[i]);
    }
    for (int i = 0; i < PUNUM; i++) {
        ob_in.push_back(offset_buf[i]);
        ob_in.push_back(indice_buf[i]);
        ob_in.push_back(weight_buf[i]);
    }
    ob_in.push_back(source_indice_buf);
    ob_in.push_back(source_weight_buf);

    for (int i = 0; i < repInt; i++) {
        ob_out.push_back(result_id_buf[i]);
        ob_out.push_back(similarity_buf[i]);
    }

    // declare events
    std::vector<cl::Event> events_write(1);
    std::vector<std::vector<cl::Event> > events_kernel(repInt);
    std::vector<cl::Event> events_read(1);
    for (int i = 0; i < repInt; ++i) {
        events_kernel[i].resize(1);
    }

    // set kernel args
    for (int i = 0; i < repInt; i++) {
        int j = 0;
        similarity_kernel[i].setArg(j++, config_buf[i]);
        similarity_kernel[i].setArg(j++, source_indice_buf);
        similarity_kernel[i].setArg(j++, source_weight_buf);

        for (int k = 0; k < PUNUM; k++) {
            similarity_kernel[i].setArg(j++, offset_buf[k]);
            similarity_kernel[i].setArg(j++, indice_buf[k]);
            similarity_kernel[i].setArg(j++, weight_buf[k]);
        }

        similarity_kernel[i].setArg(j++, result_id_buf[i]);
        similarity_kernel[i].setArg(j++, similarity_buf[i]);
    }

    // launch kernel and calculate kernel execution time
    std::cout << "INFO: Kernel Start" << std::endl;

    // migrate data from host to device
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    q.finish();

    // kernel execution
    q.enqueueTask(similarity_kernel[0], &events_write, &events_kernel[0][0]);
    for (int i = 1; i < repInt; i++) {
        q.enqueueTask(similarity_kernel[i], &events_kernel[i - 1], &events_kernel[i][0]);
    }

    // migrate data from device to host
    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel[repInt - 1], &events_read[0]);
    q.finish();

    struct timeval end_time;
    gettimeofday(&end_time, 0);
    std::cout << "INFO: Finish kernel execution" << std::endl;
    std::cout << "INFO: Finish E2E execution" << std::endl;

    // print related times
    unsigned long timeStart, timeEnd, exec_time0;
    std::cout << "-------------------------------------------------------" << std::endl;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "INFO: Data transfer from host to device: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "INFO: Data transfer from device to host: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    exec_time0 = 0;
    for (int i = 0; i < repInt; ++i) {
        events_kernel[i][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
        events_kernel[i][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
        exec_time0 += (timeEnd - timeStart) / 1000.0;
    }
    std::cout << "INFO: Average kernel execution per run: " << exec_time0 / repInt << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    unsigned long exec_timeE2E = diff(&end_time, &start_time);
    std::cout << "INFO: FPGA execution time of " << repInt << " runs:" << exec_timeE2E << " us\n"
              << "INFO: Average execution per run: " << exec_timeE2E - exec_time0 * repInt + exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;

#else
    sparseSimilarityKernel(config[0], sourceIndice, sourceWeight, offsetDDR[0], indiceDDR[0], weightDDR[0],
                           offsetDDR[1], indiceDDR[1], weightDDR[1], offsetDDR[2], indiceDDR[2], weightDDR[2],
                           offsetDDR[3], indiceDDR[3], weightDDR[3], offsetDDR[4], indiceDDR[4], weightDDR[4],
                           offsetDDR[5], indiceDDR[5], weightDDR[5], offsetDDR[6], indiceDDR[6], weightDDR[6],
                           offsetDDR[7], indiceDDR[7], weightDDR[7], result_id[0], similarity[0]);
#endif

    // need to write a compare function in order to compare golden values with results and put it here
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    int err = checkData<PU_NUMBER>(goldenFile, result_id[0], similarity[0]);
    if (err) {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
    } else {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    }
    return err;
}

template <int PUNUM>
int computeSimilarity(std::string xclbinPath,
                      std::string goldenFile,
                      unsigned int numVertices,
                      unsigned int numEdges,
                      int similarityType,
                      int dataType,
                      int sourceID,
                      int sortK,
                      int repInt,
                      unsigned int numVerticesPU[PUNUM],
                      unsigned int numEdgesPU[PUNUM],
                      float* weightDense[4 * PUNUM],
                      unsigned int sourceNUM,
                      ap_uint<32>* sourceWeight) {
    struct timeval start_time; // End to end time clock start
    gettimeofday(&start_time, 0);

    // output && config////////////////////////////////////////////////////////////////
    std::vector<ap_uint<32>*> config(repInt);
    std::vector<ap_uint<32>*> result_id(repInt);
    std::vector<float*> similarity(repInt);
    unsigned int startID[PUNUM];
    unsigned int tmp = 0;
    for (int i = 0; i < PUNUM - 1; i++) { // calculate multi PU start address
        startID[i] = tmp;
        tmp += 4 * numVerticesPU[i];
    }
    startID[PUNUM - 1] = tmp;
    for (int i = 0; i < repInt; i++) {
        similarity[i] = aligned_alloc<float>(128);
        result_id[i] = aligned_alloc<ap_uint<32> >(128);
        int base_id = 3;
        config[i] = aligned_alloc<ap_uint<32> >(64);
        config[i][0] = sortK;
        config[i][1] = sourceNUM;
        config[i][2] = similarityType;
        config[i][3] = dataType;

        for (int j = 0; j < PUNUM; j++) {
            config[i][4 + j] = startID[j];
            config[i][4 + PUNUM + j] = numVerticesPU[j];
            config[i][4 + 2 * PUNUM + j] = numEdgesPU[j];
        }
    }
///////////////////////////////////////////////////////////////////////

#ifndef HLS_TEST
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int fail;

    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &fail);
    logger.logCreateContext(fail);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &fail);
    logger.logCreateCommandQueue(fail);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbinPath);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &fail);
    logger.logCreateProgram(fail);

    // create kernels
    std::vector<cl::Kernel> similarity_kernel(repInt);
    for (int i = 0; i < repInt; i++) {
        similarity_kernel[i] = cl::Kernel(program, "denseSimilarityKernel", &fail);
        logger.logCreateKernel(fail);
    }
    std::cout << "INFO: kernel has been created" << std::endl;

    // declare map of host buffers
    std::vector<cl_mem_ext_ptr_t> mext_o(3 * repInt + 4 * PUNUM + 1);
    for (int i = 0; i < PUNUM; i++) {
        mext_o[4 * i + 0] = {XCL_BANK(8 * i), weightDense[4 * i], 0};
        mext_o[4 * i + 1] = {XCL_BANK(8 * i + 1), weightDense[4 * i + 1], 0};
        mext_o[4 * i + 2] = {XCL_BANK(8 * i + 2), weightDense[4 * i + 2], 0};
        mext_o[4 * i + 3] = {XCL_BANK(8 * i + 3), weightDense[4 * i + 3], 0};
    }

    mext_o[4 * PUNUM] = {XCL_BANK28, sourceWeight, 0};

    for (int i = 0; i < repInt; i++) {
        mext_o[4 * PUNUM + 1 + i] = {XCL_BANK28, config[i], 0};
        mext_o[4 * PUNUM + 1 + repInt + i] = {XCL_BANK28, result_id[i], 0};
        mext_o[4 * PUNUM + 1 + 2 * repInt + i] = {XCL_BANK28, similarity[i], 0};
    }

    // create device buffer and map dev buf to host buf
    cl::Buffer weight_buf[4 * PUNUM];
    cl::Buffer source_weight_buf;
    std::vector<cl::Buffer> config_buf(repInt);
    std::vector<cl::Buffer> result_id_buf(repInt);
    std::vector<cl::Buffer> similarity_buf(repInt);

    // declare cl::buffers
    for (int i = 0; i < 4 * PUNUM; i++) {
        int sizeW = numVerticesPU[i / 4] * numEdges;
        weight_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(ap_uint<32>) * (sizeW + CHANNEL_NUMBER), &mext_o[i]);
    }

    source_weight_buf = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(ap_uint<32>) * (sourceNUM + CHANNEL_NUMBER), &mext_o[4 * PUNUM]);

    for (int i = 0; i < repInt; i++) {
        config_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(ap_uint<32>) * 64, &mext_o[4 * PUNUM + 1 + i]);
        result_id_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                      sizeof(ap_uint<32>) * 128, &mext_o[4 * PUNUM + 1 + repInt + i]);
        similarity_buf[i] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                       sizeof(float) * 128, &mext_o[4 * PUNUM + 1 + 2 * repInt + i]);
    }

    // add buffers to migrate
    std::vector<cl::Memory> init;
    for (int i = 0; i < repInt; i++) {
        init.push_back(config_buf[i]);
    }
    for (int i = 0; i < 4 * PUNUM; i++) {
        init.push_back(weight_buf[i]);
    }
    for (int i = 0; i < repInt; i++) {
        init.push_back(result_id_buf[i]);
    }
    for (int i = 0; i < repInt; i++) {
        init.push_back(similarity_buf[i]);
    }
    init.push_back(source_weight_buf);

    // migrate data from host to device
    q.enqueueMigrateMemObjects(init, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);
    q.finish();

    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;

    for (int i = 0; i < repInt; i++) {
        ob_in.push_back(config_buf[i]);
    }
    for (int i = 0; i < 4 * PUNUM; i++) {
        ob_in.push_back(weight_buf[i]);
    }
    ob_in.push_back(source_weight_buf);

    for (int i = 0; i < repInt; i++) {
        ob_out.push_back(result_id_buf[i]);
        ob_out.push_back(similarity_buf[i]);
    }

    // declare events
    std::vector<cl::Event> events_write(1);
    std::vector<std::vector<cl::Event> > events_kernel(repInt);
    std::vector<cl::Event> events_read(1);
    for (int i = 0; i < repInt; ++i) {
        events_kernel[i].resize(1);
    }

    // set kernel args
    for (int i = 0; i < repInt; i++) {
        int j = 0;
        similarity_kernel[i].setArg(j++, config_buf[i]);
        similarity_kernel[i].setArg(j++, source_weight_buf);

        for (int k = 0; k < 4 * PUNUM; k++) {
            similarity_kernel[i].setArg(j++, weight_buf[k]);
        }

        similarity_kernel[i].setArg(j++, result_id_buf[i]);
        similarity_kernel[i].setArg(j++, similarity_buf[i]);
    }

    // launch kernel and calculate kernel execution time
    std::cout << "INFO: Kernel Start" << std::endl;

    // migrate data from host to device
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    q.finish();

    // kernel execution
    q.enqueueTask(similarity_kernel[0], &events_write, &events_kernel[0][0]);
    for (int i = 1; i < repInt; i++) {
        q.enqueueTask(similarity_kernel[i], &events_kernel[i - 1], &events_kernel[i][0]);
    }

    // migrate data from device to host
    q.enqueueMigrateMemObjects(ob_out, 1, &events_kernel[repInt - 1], &events_read[0]);
    q.finish();

    struct timeval end_time;
    gettimeofday(&end_time, 0);
    std::cout << "INFO: Finish kernel execution" << std::endl;
    std::cout << "INFO: Finish E2E execution" << std::endl;

    // print related times
    unsigned long timeStart, timeEnd, exec_time0;
    std::cout << "-------------------------------------------------------" << std::endl;
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    exec_time0 = (timeEnd - timeStart) / 1000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_H2D_MS, exec_time0);
    std::cout << "-------------------------------------------------------" << std::endl;
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    exec_time0 = (timeEnd - timeStart) / 1000.0;
    logger.info(xf::common::utils_sw::Logger::Message::TIME_D2H_MS, exec_time0);
    std::cout << "-------------------------------------------------------" << std::endl;
    exec_time0 = 0;
    for (int i = 0; i < repInt; ++i) {
        events_kernel[i][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
        events_kernel[i][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
        exec_time0 += (timeEnd - timeStart) / 1000.0;
    }
    logger.info(xf::common::utils_sw::Logger::Message::TIME_KERNEL_MS, exec_time0 / repInt);
    std::cout << "-------------------------------------------------------" << std::endl;
    unsigned long exec_timeE2E = diff(&end_time, &start_time);
    std::cout << "INFO: FPGA execution time of " << repInt << " runs:" << exec_timeE2E << " us\n"
              << "INFO: Average execution per run: " << exec_timeE2E - exec_time0 * repInt + exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;

#else
    denseSimilarityKernel(config[0], sourceWeight, weightDense[0], weightDense[1], weightDense[2], weightDense[3],
                          weightDense[4], weightDense[5], weightDense[6], weightDense[7], weightDense[8],
                          weightDense[9], weightDense[10], weightDense[11], weightDense[12], weightDense[13],
                          weightDense[14], weightDense[15], result_id[0], similarity[0]);
#endif

    // need to write a compare function in order to compare golden values with results and put it here
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    int err = checkData<PU_NUMBER>(goldenFile, result_id[0], similarity[0]);
    if (err) {
        logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL);
    } else {
        logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    }
    return err;
}

#endif //#ifndef VT_GRAPH_SIMILARITY_H
