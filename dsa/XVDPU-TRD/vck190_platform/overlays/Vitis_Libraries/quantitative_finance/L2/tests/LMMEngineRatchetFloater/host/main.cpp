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
#define RFX_DEF (0.015f)
#define RFY_DEF (0.015f)
#define RFALPHA_DEF (0.01f)
#define PATHS_DEF (100)
#define NOTIONAL (1000000.0f)
#define KRNL_NAME "lmmRatchetFloaterKernel"

#define EXPECTED (-7753.48f) // The expected ratchet floater with the default values

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
TEST_DT runFpga(const std::string& xclbinLoc, unsigned noPaths, TEST_DT rfX, TEST_DT rfY, TEST_DT rfAlpha) {
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
    OCL_CHECK(err, err = krnl_lmm.setArg(6, rfX));
    OCL_CHECK(err, err = krnl_lmm.setArg(7, rfY));
    OCL_CHECK(err, err = krnl_lmm.setArg(8, rfAlpha));
    OCL_CHECK(err, err = krnl_lmm.setArg(9, bufferSeeds));
    OCL_CHECK(err, err = krnl_lmm.setArg(10, bufferPriceOut));

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
    std::cout << std::string(name) << " -b xclbin [-p no_paths] [-x X] [-y Y] [-a alpha] [-h]" << std::endl
              << "\t-b --xclbin       Location of xclbin" << std::endl
              << "\t-p --paths        Number of paths (defaults to " << PATHS_DEF << ")" << std::endl
              << "\t-x --rfX          X parameter for ratchet floater (defaults to " << RFX_DEF << ")" << std::endl
              << "\t-y --rfY          Y parameter for ratchet floater (defaults to " << RFY_DEF << ")" << std::endl
              << "\t-a --rfAlpha      alpha parameter for ratchet floater (defaults to " << RFALPHA_DEF << ")"
              << std::endl;
}

static struct option longOps[] = {{"help", no_argument, 0, 'h'},
                                  {"xclbin", required_argument, 0, 'b'},
                                  {"paths", required_argument, 0, 'p'},
                                  {"rfX", required_argument, 0, 'x'},
                                  {"rfY", required_argument, 0, 'y'},
                                  {"rfAlpha", required_argument, 0, 'a'},
                                  {0, 0, 0, 0}};

int main(int argc, char** argv) {
    std::string xclbinLoc;
    TEST_DT rfX = RFX_DEF, rfY = RFY_DEF, rfAlpha = RFALPHA_DEF;
    unsigned noPaths = PATHS_DEF;
    char opt;

    while ((opt = getopt_long(argc, argv, "hb:p:x:y:a:", longOps, NULL)) != -1) {
        switch (opt) {
            case 'b':
                xclbinLoc = std::string(optarg);
                break;
            case 'p':
                noPaths = atoi(optarg);
                break;
            case 'x':
                rfX = atof(optarg);
                break;
            case 'y':
                rfY = atof(optarg);
                break;
            case 'a':
                rfAlpha = atof(optarg);
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
              << "\tX = " << rfX << std::endl
              << "\tY = " << rfY << std::endl
              << "\talpha = " << rfAlpha << std::endl;
    const TEST_DT price = runFpga(xclbinLoc, noPaths, rfX, rfY, rfAlpha);
    std::cout << "Got price: " << price << std::endl;

    // If we are using default data, check against expected
    int ret = 0;
    if (rfX == RFX_DEF && rfY == RFY_DEF && rfAlpha == RFALPHA_DEF) {
        std::cout << "*** Using default arguments ***" << std::endl;
        double epsilon;
        if (noPaths <= 1000) {
            epsilon = 0.05; // 5%
        } else {
            epsilon = 0.03; // 3%
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
