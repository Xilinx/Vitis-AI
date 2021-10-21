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
#include "svd.hpp"
#include <sys/time.h>
#include <iostream>
#include <vector>
#include "../kernel/kernel_svd.hpp"
#include "util.hpp"
#include "xcl2.hpp"
#define dataAN 4

void benchmark_svd_functions(std::string xclbinName, double& errA) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int cl_err;
    // variables to measure time
    struct timeval tstart, tend;
    // platform related operations
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &cl_err);
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbinName);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);
    cl::Kernel kernel_svd_0(program, "kernel_svd_0", &cl_err);
    logger.logCreateKernel(cl_err);
    std::cout << "kernel has been created" << std::endl;

    // associate input_data to host ddr bank
    int out_size = dataAN * dataAN;
    int out_size1 = dataAN * dataAN;
    int out_size2 = dataAN;
    int input_size = dataAN * dataAN;
    // const static int SZ = 8 * sizeof(double);
    double* sigma_kernel;
    double* U_kernel;
    double* V_kernel;
    double* dataA_svd;
    dataA_svd = aligned_alloc<double>(input_size);
    sigma_kernel = aligned_alloc<double>(out_size2);
    U_kernel = aligned_alloc<double>(out_size);
    V_kernel = aligned_alloc<double>(out_size1);

    int k = 0;
    double dataB_reduced[dataAN];
    double dataA_reduced[dataAN][dataAN];
    double dataU_reduced[dataAN][dataAN];
    double dataV_reduced[dataAN][dataAN];
    double sigma[dataAN][dataAN];
    double sigma2[dataAN][dataAN];
    dataA_reduced[0][0] = 2545;
    dataA_reduced[0][1] = 2137.34902052323241150588728487;
    dataA_reduced[0][2] = 1821.87553334160179474565666169;
    dataA_reduced[0][3] = 16306.0391790706853498704731464;
    dataA_reduced[1][0] = 2137.34902052323241150588728487;
    dataA_reduced[1][1] = 1821.87553334160179474565666169;
    dataA_reduced[1][2] = 1573.78716625872380063810851425;
    dataA_reduced[1][3] = 12618.9394872652155754622071981;
    dataA_reduced[2][0] = 1821.87553334160179474565666169;
    dataA_reduced[2][1] = 1573.78716625872380063810851425;
    dataA_reduced[2][2] = 1375.76089061747416053549386561;
    dataA_reduced[2][3] = 9923.53468331521253276150673628;
    dataA_reduced[3][0] = 16306.0391790706853498704731464;
    dataA_reduced[3][1] = 12618.9394872652155754622071981;
    dataA_reduced[3][2] = 9923.53468331521253276150673628;
    dataA_reduced[3][3] = 147483.987672218354418873786926;
    dataB_reduced[0] = 16270.5645060545830347109586;
    dataB_reduced[1] = 12590.6436801607960660476237535;
    dataB_reduced[2] = 9900.72724799832030839752405882;
    dataB_reduced[3] = 147196.833035749499686062335968;
    int ll = 0;
    for (int r = 0; r < dataAN; r++) {
        for (int j = 0; j < dataAN; j++) {
            dataA_svd[ll] = dataA_reduced[r][j];
            ll++;
        }
    }

    ///////////////////// DDR Settings //////////////////////
    std::vector<cl_mem_ext_ptr_t> mext_i(1);
    std::vector<cl_mem_ext_ptr_t> mext_o(3);

    mext_i[0] = {0, dataA_svd, kernel_svd_0()};
    mext_o[0] = {1, sigma_kernel, kernel_svd_0()};
    mext_o[1] = {2, U_kernel, kernel_svd_0()};
    mext_o[2] = {2, V_kernel, kernel_svd_0()};

    // create device buffer and map dev buf to host buf
    std::vector<cl::Buffer> input_buffer(1), output_buffer(3);

    input_buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 sizeof(double) * input_size, &mext_i[0]);
    output_buffer[0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                  sizeof(double) * out_size2, &mext_o[0]);
    output_buffer[1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                  sizeof(double) * out_size, &mext_o[1]);
    output_buffer[2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                  sizeof(double) * out_size1, &mext_o[2]);

    // data transfer from host buffer to device buffer
    std::vector<std::vector<cl::Event> > kernel_ent(2);
    kernel_ent[0].resize(1);
    kernel_ent[1].resize(1);

    std::vector<cl::Memory> ob_in, ob_out;
    ob_in.push_back(input_buffer[0]);
    ob_out.push_back(output_buffer[0]);
    ob_out.push_back(output_buffer[1]);
    ob_out.push_back(output_buffer[2]);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &kernel_ent[0][0]); // 0 from host to dev
    q.finish();
    std::cout << "finished data transfer from h2d" << std::endl;

    gettimeofday(&tstart, 0);

    kernel_svd_0.setArg(0, input_buffer[0]);
    kernel_svd_0.setArg(1, output_buffer[0]);
    kernel_svd_0.setArg(2, output_buffer[1]);
    kernel_svd_0.setArg(3, output_buffer[2]);
    kernel_svd_0.setArg(4, dataAN);

    // launch kernel and calculate kernel execution time
    q.enqueueTask(kernel_svd_0, nullptr, nullptr);
    q.finish();

    gettimeofday(&tend, 0);
    printf("Kernel 0 done!\n");
    printf("kernel execution time : %lu us\n", diff(&tend, &tstart));

    // data transfer from devive buffer to host buffer
    q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr); // 1 from dev to host
    q.flush();
    q.finish();

    ///////////// post process of data  ////////////
    // calculate U*Sigma*V
    for (int i = 0; i < NA; ++i) {
        for (int j = 0; j < NA; ++j) {
            U_kernel[i * dataAN + j] = U_kernel[i * dataAN + j] * sigma_kernel[j];
        }
    }
    double dataA_out[NA][NA];
    for (int i = 0; i < NA; ++i) {
        for (int j = 0; j < NA; ++j) {
            double tmpSum = 0;
            for (int k = 0; k < NA; ++k) {
                tmpSum += U_kernel[i * dataAN + k] * V_kernel[j * dataAN + k];
            }
            dataA_out[i][j] = tmpSum;
        }
    }

    errA = 0;
    for (int i = 0; i < NA; i++) {
        for (int j = 0; j < NA; j++) {
            errA += (dataA_reduced[i][j] - dataA_out[i][j]) * (dataA_reduced[i][j] - dataA_out[i][j]);
        }
    }
    errA = std::sqrt(errA);
}
