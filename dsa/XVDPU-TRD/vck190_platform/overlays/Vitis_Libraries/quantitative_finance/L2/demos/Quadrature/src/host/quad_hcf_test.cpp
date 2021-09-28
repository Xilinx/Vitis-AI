#include <vector>
#include <chrono>

#include <math.h> /* fabs */

#include "xcl2.hpp"
#include "test_data.hpp"
#include "quad_hcf_engine_def.hpp"
#include "xf_utils_sw/logger.hpp"

#define TEST_TOLERANCE 0.001

extern TEST_DT model_hcfEngine(struct hcfEngineInputDataType* input_data);

static int check(TEST_DT act, TEST_DT exp) {
    if (fabs(act - exp) > TEST_TOLERANCE) {
        return 0;
    }
    return 1;
}

int main(int argc, char* argv[]) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    std::string xclbin_file(argv[1]);
    TEST_DT integration_tolerance = 0.0001;
    if (argc == 3) {
        integration_tolerance = atof(argv[2]);
    }

    int num = sizeof(test_data) / sizeof(test_data_type);
    if (num > MAX_NUMBER_TESTS) {
        std::cout << "ERROR: too many tests. " << num << " specified, ";
        std::cout << MAX_NUMBER_TESTS << " allowed." << std::endl;
        return 1;
    }

    // IO data
    std::vector<struct hcfEngineInputDataType, aligned_allocator<struct hcfEngineInputDataType> > input_data(num);

    std::vector<TEST_DT, aligned_allocator<TEST_DT> > output_data(num);
    size_t bytes_in = sizeof(struct hcfEngineInputDataType) * num;
    size_t bytes_out = sizeof(TEST_DT) * num;

    // fill the input data
    std::cout << "Processing input data" << std::endl;
    for (int i = 0; i < num; i++) {
        input_data[i].s = test_data[i].s;
        input_data[i].k = test_data[i].k;
        input_data[i].t = test_data[i].t;
        input_data[i].v = test_data[i].v;
        input_data[i].r = test_data[i].r;
        input_data[i].rho = test_data[i].rho;
        input_data[i].vvol = test_data[i].vvol;
        input_data[i].vbar = test_data[i].vbar;
        input_data[i].kappa = test_data[i].kappa;
        input_data[i].tol = integration_tolerance;
    }

    // get device
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // get context
    std::cout << "Creating context" << std::endl;
    cl_int err;
    cl::Context ctx(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);

    // create command queue
    std::cout << "Creating command queue" << std::endl;
    cl::CommandQueue cq(ctx, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // import and program the xclbin
    std::cout << "Programming device" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_file);
    devices.resize(1);
    cl::Program program(ctx, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long t_program = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    cl::Kernel krnl(program, "quad_hcf_kernel", &err);
    logger.logCreateKernel(err);

    // memory objects
    std::cout << "Allocating memory objects" << std::endl;
    cl_int overall_err = 0;
    cl::Buffer dev_in(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, bytes_in, input_data.data(), &err);
    overall_err |= err;
    cl::Buffer dev_out(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, bytes_out, output_data.data(), &err);
    overall_err |= err;
    if (overall_err != CL_SUCCESS) {
        std::cout << "ERROR: failed to create buffer" << std::endl;
        return 1;
    }

    // set the arguments
    std::cout << "Set device arguments" << std::endl;
    err = krnl.setArg(0, dev_in);
    err |= krnl.setArg(1, dev_out);
    err |= krnl.setArg(2, num);
    if (err != CL_SUCCESS) {
        std::cout << "ERROR: failed to set args" << std::endl;
        return 1;
    }

    // copy input data to device
    std::cout << "Migrate memory to device" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    err = cq.enqueueMigrateMemObjects({dev_in}, 0);
    cq.finish();
    elapsed = std::chrono::high_resolution_clock::now() - start;
    long long t_mem_to_device = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    if (err != CL_SUCCESS) {
        std::cout << "ERROR: failed to migrate memory to device" << std::endl;
        return 1;
    }

    // launch kernel
    std::cout << "Launching kernel" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    cl::Event kernel_event;
    err = cq.enqueueTask(krnl, NULL, &kernel_event);
    cq.finish();
    elapsed = std::chrono::high_resolution_clock::now() - start;
    long long t_run = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    if (err != CL_SUCCESS) {
        std::cout << "ERROR: failed to launch kernel" << std::endl;
        return 1;
    }

    // copy results back to local memory
    start = std::chrono::high_resolution_clock::now();
    err = cq.enqueueMigrateMemObjects({dev_out}, CL_MIGRATE_MEM_OBJECT_HOST);
    cq.finish();
    elapsed = std::chrono::high_resolution_clock::now() - start;
    long long t_mem_from_device = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    if (err != CL_SUCCESS) {
        std::cout << "ERROR: failed to copy results to local memory" << std::endl;
        return 1;
    }

    // check results
    int fails = 0;
    double error_max = 0;
    for (int i = 0; i < num; i++) {
        if (!check(output_data[i], test_data[i].exp)) {
            std::cout << "    FPGA[" << i << "]: Expected " << test_data[i].exp;
            std::cout << ", got " << output_data[i];
            std::cout << ", diff " << fabs(output_data[i] - test_data[i].exp) << std::endl;
            fails++;
        }

        if (fabs(output_data[i] - test_data[i].exp) > error_max) {
            error_max = fabs(output_data[i] - test_data[i].exp);
        }
    }

    std::cout << "Results:" << std::endl;
    std::cout << "Integration tolerance: " << integration_tolerance << std::endl;
    std::cout << "Result tolerance:      " << TEST_TOLERANCE << std::endl;
    std::cout << "Number of tests:       " << num << std::endl;
    std::cout << "Out of tolerance:      " << fails << std::endl;
    std::cout << "Max error:             " << error_max << std::endl;
    std::cout << "Timings:" << std::endl;
    std::cout << "    program device  = " << t_program << " us" << std::endl;
    std::cout << "    mem to device   = " << t_mem_to_device << " us" << std::endl;
    std::cout << "    run kernel      = " << t_run << " us" << std::endl;
    std::cout << "    mem from device = " << t_mem_from_device << " us" << std::endl;

    int ret = 0;
    if (fails) {
        ret = 1;
    }
    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return ret;
}
