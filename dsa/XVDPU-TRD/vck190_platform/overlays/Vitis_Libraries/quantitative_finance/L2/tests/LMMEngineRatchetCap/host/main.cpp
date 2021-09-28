#include <iostream>
#include <vector>
#include <cmath>
#include <getopt.h>
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#define TEST_DT float
#define TEST_UN (4)
#define NO_TENORS (10)
#define BETA (0.2f)
#define RCKAPPA0_DEF (0.001f)
#define RCSPREAD_DEF (0.0f)
#define PATHS_DEF (100)
#define NOTIONAL (1000000.0f)
#define KRNL_NAME "lmmRatchetCapKernel"

#define EXPECTED (11042.3f) // Expected Ratchet Cap with default parameters

template <typename T>
using al_vec = std::vector<T, aligned_allocator<T> >;

static al_vec<TEST_DT> capletVolas = {0.2366, 0.2487, 0.2573, 0.2564, 0.2476, 0.2376, 0.2252, 0.2246, 0.2223};
static al_vec<TEST_DT> presentFc = {0.0112, 0.0118, 0.0123, 0.0127, 0.0132, 0.0137, 0.0145, 0.0154, 0.0163, 0.0174};

static al_vec<unsigned> getFpgaSeeds() {
    al_vec<unsigned> seeds(TEST_UN);
    for (unsigned i = 0; i < TEST_UN; i++) {
        seeds[i] = 42 + i;
    }
    return seeds;
}

xf::common::utils_sw::Logger logger(std::cout, std::cerr);
TEST_DT runFpga(const std::string& xclbinLoc, unsigned noPaths, TEST_DT rcSpread, TEST_DT rcKappa0) {
    al_vec<TEST_DT> gotPrice(1);
    al_vec<unsigned> seeds = getFpgaSeeds();

    // OPENCL HOST CODE AREA START

    std::cout << "\n\nConnecting to device and loading kernel..." << std::endl;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl_int err;

    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // Load the binary file (using function from xcl2.cpp)
    cl::Program::Binaries bins = xcl::import_binary_file(xclbinLoc);

    devices.resize(1);
    cl::Program program(context, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel krnl_lmm(program, KRNL_NAME, &err);
    logger.logCreateKernel(err);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    std::cout << "Allocating buffers..." << std::endl;
    OCL_CHECK(err, cl::Buffer bufferCapletVolasIn(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                  sizeof(TEST_DT) * (NO_TENORS - 1), capletVolas.data(), &err));
    OCL_CHECK(err, cl::Buffer bufferPresentFc(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                              sizeof(TEST_DT) * (NO_TENORS), presentFc.data(), &err));
    OCL_CHECK(err, cl::Buffer bufferSeeds(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned) * (TEST_UN),
                                          seeds.data(), &err));
    OCL_CHECK(err, cl::Buffer bufferPriceOut(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(TEST_DT),
                                             gotPrice.data(), &err));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bufferCapletVolasIn, bufferSeeds, bufferPresentFc}, 0));

    // Set the arguments
    OCL_CHECK(err, err = krnl_lmm.setArg(0, NO_TENORS));
    OCL_CHECK(err, err = krnl_lmm.setArg(1, noPaths));
    OCL_CHECK(err, err = krnl_lmm.setArg(2, bufferPresentFc));
    OCL_CHECK(err, err = krnl_lmm.setArg(3, BETA));
    OCL_CHECK(err, err = krnl_lmm.setArg(4, bufferCapletVolasIn));
    OCL_CHECK(err, err = krnl_lmm.setArg(5, NOTIONAL));
    OCL_CHECK(err, err = krnl_lmm.setArg(6, rcSpread));
    OCL_CHECK(err, err = krnl_lmm.setArg(7, rcKappa0));
    OCL_CHECK(err, err = krnl_lmm.setArg(8, bufferSeeds));
    OCL_CHECK(err, err = krnl_lmm.setArg(9, bufferPriceOut));

    // Launch the kernel
    std::cout << "Launching kernel..." << std::endl;
    uint64_t nstimestart, nstimeend;
    cl::Event event;
    OCL_CHECK(err, err = q.enqueueTask(krnl_lmm, NULL, &event));
    OCL_CHECK(err, err = q.finish());
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto duration_nanosec = nstimeend - nstimestart;
    std::cout << "  Duration returned by profile API is " << (duration_nanosec * (1.0e-6)) << " ms **** " << std::endl;

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({bufferPriceOut}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    // OPENCL HOST CODE AREA END
    return gotPrice[0];
}

void help(const char* name) {
    std::cout << std::string(name) << " -b xclbin [-p no_paths] [-h]" << std::endl
              << "\t-b --xclbin       Location of xclbin" << std::endl
              << "\t-p --paths        Number of paths (defaults to " << PATHS_DEF << ")" << std::endl
              << "\t-s --spread       Spread parameter (s) of Ratchet Cap (defaults to " << RCSPREAD_DEF << ")"
              << std::endl
              << "\t-k --kappa0       Kappa0 parameter (k0) of Ratceht Cap (defaults to " << RCKAPPA0_DEF << ")"
              << std::endl;
}

static struct option longOps[] = {{"help", no_argument, 0, 'h'},         {"xclbin", required_argument, 0, 'b'},
                                  {"paths", required_argument, 0, 'p'},  {"spread", required_argument, 0, 's'},
                                  {"kappa0", required_argument, 0, 'k'}, {0, 0, 0, 0}};

int main(int argc, char** argv) {
    std::string xclbinLoc;
    TEST_DT rcSpread = RCSPREAD_DEF, rcKappa0 = RCKAPPA0_DEF;
    unsigned noPaths = PATHS_DEF;
    char opt;

    while ((opt = getopt_long(argc, argv, "hb:p:s:k:", longOps, NULL)) != -1) {
        switch (opt) {
            case 'b':
                xclbinLoc = std::string(optarg);
                break;
            case 'p':
                noPaths = atoi(optarg);
                break;
            case 's':
                rcSpread = atof(optarg);
                break;
            case 'k':
                rcKappa0 = atof(optarg);
                break;
            case 'h':
            case '?':
            default:
                help(argv[0]);
                return 1;
        }
    }
    if (xclbinLoc.empty()) {
        std::cerr << "Missing mandatory argument 'xclbin'" << std::endl;
        help(argv[0]);
        return 1;
    }
    std::cout << "Running Cap pricing LMM test:" << std::endl;
    std::cout << "Parameters:" << std::endl
              << "\tMonte-Carlo paths = " << noPaths << std::endl
              << "\tNotional = " << NOTIONAL << std::endl
              << "\ts = " << rcSpread << std::endl
              << "\tK0 = " << rcKappa0 << std::endl;
    const TEST_DT price = runFpga(xclbinLoc, noPaths, rcSpread, rcKappa0);
    std::cout << "Got price: " << price << std::endl;
    int ret = 0;
    if (rcSpread == RCSPREAD_DEF && rcKappa0 == RCKAPPA0_DEF) {
        std::cout << "*** Using default arguments ***" << std::endl;
        double epsilon;
        if (noPaths <= 100) {
            epsilon = 0.06; // 6%
        } else if (noPaths <= 500) {
            epsilon = 0.04; // 4%
        } else {
            epsilon = 0.02; // 2%
        }
        double diff = std::abs((double)((price - EXPECTED) / EXPECTED));
        std::cout << "Expected price = " << EXPECTED << " (diff = " << (diff * 100) << "%)" << std::endl;
        if (diff > epsilon) {
            ret = 1;
        }
    }
    ret ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return ret;
}
