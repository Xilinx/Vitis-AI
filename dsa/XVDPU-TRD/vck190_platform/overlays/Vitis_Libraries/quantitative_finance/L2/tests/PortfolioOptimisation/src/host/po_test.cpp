#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <exception>
#include <iomanip>
#include <vector>
#include <cmath>
#include "xcl2.hpp"
#include "xf_utils_sw/logger.hpp"

#define NUM_ASSETS (10)
#define NUM_PRICES (61)

// Temporary copy of this macro definition until new xcl2.hpp is used
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

float tolerance = 0.000003;

float g_prices[NUM_ASSETS][NUM_PRICES] = {
    37.57,  38.74,  41.52,  43.47,  43.65,  41.17,  42.35,  46.84,  51.38,  51.60,  51.99,  47.29,  48.08,
    49.56,  52.94,  52.71,  50.28,  53.47,  55.79,  55.52,  52.13,  53.33,  56.46,  52.67,  47.87,  50.00,
    48.83,  54.05,  53.11,  50.86,  52.41,  53.19,  54.62,  59.45,  58.69,  58.68,  60.53,  63.25,  66.92,
    63.92,  62.97,  66.66,  70.36,  71.76,  74.90,  76.00,  74.51,  71.51,  74.06,  74.46,  76.63,  80.39,
    83.13,  85.26,  89.74,  87.16,  92.16,  95.52,  100.33, 99.05,  99.99,

    84.74,  94.76,  97.03,  108.33, 104.10, 94.09,  90.65,  101.65, 103.53, 93.93,  116.25, 98.03,  104.53,
    122.29, 148.39, 166.72, 178.37, 154.79, 128.73, 75.08,  35.68,  29.69,  36.33,  29.32,  19.38,  20.81,
    26.15,  33.63,  35.26,  39.22,  43.24,  43.83,  34.07,  44.17,  54.52,  43.95,  52.42,  62.90,  54.12,
    46.79,  38.21,  43.94,  42.16,  43.50,  42.39,  48.28,  58.02,  57.28,  57.15,  53.62,  47.43,  45.89,
    45.82,  39.80,  30.01,  21.94,  25.27,  27.26,  26.42,  30.14,  31.01,

    44.84,  46.63,  47.10,  53.19,  55.32,  54.10,  58.99,  69.82,  71.24,  66.07,  69.23,  59.42,  68.04,
    73.58,  80.14,  89.35,  89.45,  78.93,  70.99,  44.80,  23.82,  21.48,  22.74,  20.87,  18.02,  18.69,
    21.99,  31.13,  31.04,  33.82,  33.61,  35.03,  32.09,  37.23,  43.37,  36.67,  36.40,  41.82,  36.99,
    29.07,  25.65,  29.43,  27.96,  31.81,  31.24,  30.73,  36.99,  35.38,  35.75,  35.24,  36.14,  32.79,
    34.08,  30.54,  21.74,  15.74,  20.51,  18.89,  18.19,  20.52,  23.30,

    25.53,  25.26,  27.14,  27.91,  26.80,  26.36,  26.22,  26.88,  33.59,  30.76,  32.59,  29.84,  25.00,
    26.08,  26.21,  26.12,  25.38,  23.72,  25.27,  24.72,  20.68,  18.85,  18.13,  15.94,  15.16,  17.25,
    19.02,  19.74,  22.46,  22.22,  23.42,  24.44,  26.34,  28.06,  29.09,  26.89,  27.49,  28.08,  29.28,
    24.85,  22.16,  24.86,  22.72,  23.71,  25.82,  24.61,  27.19,  27.01,  26.05,  24.88,  25.40,  24.67,
    25.64,  27.02,  26.40,  24.70,  26.43,  25.58,  25.96,  29.53,  30.77,

    84.61,  92.91,  99.80,  121.19, 122.04, 131.76, 138.48, 153.47, 189.95, 182.22, 198.08, 135.36, 125.02,
    143.50, 173.95, 188.75, 167.44, 158.95, 169.53, 113.66, 107.59, 92.67,  85.35,  90.13,  89.31,  105.12,
    125.83, 135.81, 142.43, 163.39, 168.21, 185.35, 188.50, 199.91, 210.73, 192.06, 204.62, 235.00, 261.09,
    256.88, 251.53, 257.25, 243.10, 283.75, 300.98, 311.15, 322.56, 339.32, 353.21, 348.51, 350.13, 347.83,
    335.67, 390.48, 384.83, 381.32, 404.78, 382.20, 405.00, 456.48, 493.17,

    43.45,  44.83,  46.12,  47.31,  45.39,  45.41,  48.42,  49.36,  46.53,  47.91,  46.48,  42.38,  45.24,
    46.88,  45.64,  46.50,  43.09,  47.62,  49.16,  50.66,  45.53,  39.22,  39.91,  39.77,  35.74,  33.64,
    38.67,  40.03,  43.10,  43.96,  43.93,  45.93,  48.09,  49.41,  49.99,  51.14,  49.36,  50.57,  52.00,
    50.93,  47.95,  47.71,  47.74,  48.53,  48.29,  47.70,  49.49,  48.73,  52.29,  52.70,  55.91,  56.04,
    54.40,  54.85,  53.84,  52.72,  53.73,  49.16,  50.57,  49.52,  50.21,

    28.97,  29.34,  30.59,  31.18,  32.00,  32.40,  32.49,  34.84,  34.63,  32.22,  31.46,  30.01,  28.38,
    31.69,  28.00,  26.31,  23.11,  24.50,  24.33,  22.38,  17.12,  15.07,  14.49,  10.85,  7.84,   9.31,
    11.65,  12.41,  10.88,  12.44,  12.90,  15.33,  13.32,  14.96,  14.22,  15.11,  15.19,  17.21,  17.84,
    15.46,  13.72,  15.34,  13.78,  15.58,  15.36,  15.18,  17.67,  19.46,  20.35,  19.51,  19.90,  19.11,
    18.50,  17.57,  16.00,  15.07,  16.55,  15.76,  17.91,  18.71,  19.13,

    44.68,  44.85,  44.74,  45.07,  43.45,  42.14,  45.04,  45.24,  43.45,  41.52,  37.67,  40.31,  36.28,
    35.19,  34.85,  31.57,  22.59,  31.14,  29.48,  33.79,  23.34,  15.69,  13.90,  6.50,   3.90,   6.75,
    8.84,   11.16,  13.08,  14.66,  17.43,  16.78,  14.46,  15.72,  14.94,  15.06,  16.53,  17.72,  17.70,
    15.63,  14.28,  13.95,  12.38,  13.03,  11.39,  10.89,  13.28,  13.66,  14.22,  13.28,  12.23,  11.70,
    10.92,  9.68,   8.15,   6.11,   6.82,   5.44,   5.56,   7.13,   8.18,

    19.57,  19.80,  20.74,  21.78,  20.26,  18.63,  19.92,  19.59,  19.73,  19.29,  18.45,  18.96,  18.34,
    17.23,  16.55,  16.19,  14.61,  15.61,  16.25,  15.68,  15.06,  14.21,  15.32,  12.61,  10.88,  12.04,
    11.81,  13.58,  13.41,  14.24,  15.08,  14.94,  15.37,  16.56,  16.58,  17.01,  16.15,  15.78,  15.38,
    14.16,  13.26,  13.95,  14.96,  16.14,  16.38,  15.48,  16.63,  17.30,  18.47,  19.49,  20.13,  20.79,
    19.96,  18.65,  18.60,  17.32,  18.87,  19.86,  21.42,  21.18,  21.14,

    64.20,  67.58,  71.10,  74.81,  75.45,  76.58,  77.42,  83.59,  83.08,  80.83,  84.94,  77.70,  79.23,
    77.01,  84.74,  81.18,  80.60,  73.56,  73.55,  71.39,  68.14,  74.10,  73.81,  70.71,  63.09,  63.28,
    61.95,  64.83,  65.35,  65.80,  65.03,  64.53,  67.40,  71.01,  64.50,  60.95,  61.89,  63.77,  64.52,
    57.96,  54.71,  57.21,  57.07,  59.65,  64.19,  67.58,  71.04,  78.38,  83.53,  82.16,  85.92,  81.98,
    79.93,  78.37,  73.18,  71.81,  77.20,  80.00,  84.30,  83.28,  84.88,

};

static int ignore_first_col = 0;

std::vector<double> parse_line(std::string line, int* err) {
    *err = 0; // assume success
    std::vector<double> v;

    std::istringstream iss_line(line);
    std::string token;
    if (ignore_first_col) {
        std::getline(iss_line, token, ',');
    }

    while (std::getline(iss_line, token, ',')) {
        try {
            double value = std::stod(token);
            v.push_back(value);
        } catch (std::exception& e) {
            std::cout << "FAIL: exception: " << e.what() << std::endl;
            *err = 1;
            return v;
        }
    }
    return v;
}

std::vector<std::vector<double> > parse_file(std::string file) {
    /*
     * each line is the price for a number of assets
     */
    std::vector<std::vector<double> > res;
    std::ifstream ifs(file, std::ifstream::in);
    if (!ifs.is_open()) {
        std::cout << "ERROR: Failed to open file:" << file << std::endl;
        return res;
    }

    int first_line = 1;
    std::string line;
    while (std::getline(ifs, line)) {
        if (first_line && (line.find("date") != std::string::npos)) {
            ignore_first_col = 1;
            continue;
        }

        int err;
        std::vector<double> v = parse_line(line, &err);
        if (err) {
            std::cout << "ERROR: Failed to parse line:" << line << std::endl;
            res.clear();
            return res;
        }
        res.push_back(v);
        first_line = 0;
    }
    return res;
}

static int golden_test = 1;
int check_results(std::string s,
                  std::vector<float, aligned_allocator<float> >& w,
                  std::vector<float>& exp,
                  int num_assets,
                  float exp_return = 0,
                  float exp_variance = 0,
                  float exp_sharp_ratio = 0) {
    int res = 0;
    for (int i = 0; i < num_assets; i++) {
        std::cout << s << " w[" << i << "] = " << w[i];
        if (std::abs(exp[i] - w[i]) > tolerance && golden_test) {
            std::cout << " FAIL: expected " << exp[i] << " +/- " << tolerance;
            res = 1;
        }
        std::cout << std::endl;
    }

    if (exp_return != 0) {
        std::cout << s << " Expected return = " << w[num_assets];
        if (std::abs(w[num_assets] - exp_return) > tolerance && golden_test) {
            std::cout << " FAIL: expected " << exp_return << " +/- " << tolerance;
            res = 1;
        }
        std::cout << std::endl;
    }

    if (exp_variance != 0) {
        std::cout << s << " Variance = " << w[num_assets + 1];
        if (std::abs(w[num_assets + 1] - exp_variance) > tolerance && golden_test) {
            std::cout << " FAIL: expected " << exp_variance << " +/- " << tolerance;
            res = 1;
        }
        std::cout << std::endl;
    }

    if (exp_sharp_ratio != 0) {
        std::cout << s << " Sharpe Ratio = " << w[num_assets + 2];
        if (std::abs(w[num_assets + 2] - exp_sharp_ratio) > tolerance && golden_test) {
            std::cout << " FAIL: expected " << exp_sharp_ratio << " +/- " << tolerance;
            res = 1;
        }
        std::cout << std::endl;
    }
    return res;
}

int main(int argc, char** argv) {
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    std::string xclbin_file(argv[1]);
    float target_return = 0.02;
    float risk_free_rate = 0.001;

    // calc the amount of asset/price data
    std::string file;
    int num_assets;
    int num_prices;
    std::vector<std::vector<double> > v;
    if (argc == 3) {
        golden_test = 0;
        file = argv[2];
        v = parse_file(file);
        num_prices = v.size();
        num_assets = v[0].size();
    } else {
        num_assets = NUM_ASSETS;
        num_prices = NUM_PRICES;
    }

    // fill in the prices
    std::vector<float, aligned_allocator<float> > prices(num_assets * num_prices);
    std::vector<float, aligned_allocator<float> > gmvp_weights(num_assets + 2); // +2 for portfolio return and variance
    std::vector<float, aligned_allocator<float> > eff_weights(num_assets + 2);
    std::vector<float, aligned_allocator<float> > tan_weights(num_assets + 3); // +3 for the above and the Sharpe Ratio
    std::vector<float, aligned_allocator<float> > rf_weights(num_assets);

    if (argc == 3) {
        int asset = 0;
        int price = 0;
        int i = 0;
        for (auto& vv : v) {
            asset = 0;
            for (auto& value : vv) {
                prices[i++] = value;
                asset++;
            }
            price++;
        }
    } else {
        for (int i = 0; i < num_assets * num_prices; i++) {
            prices[i] = g_prices[i % num_assets][i / num_assets];
        }
    }

    // get device
    std::cout << "Acquiring device ... " << std::endl;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // get context
    std::cout << "Creating context" << std::endl;
    cl_int err;
    cl::Context ctx(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);

    // create command queue
    std::cout << "Creating command queue" << std::endl;
    cl::CommandQueue q(ctx, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    // import and program the xclbin
    std::cout << "Programming device" << std::endl;
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_file);
    devices.resize(1);
    cl::Program program(ctx, devices, bins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel krnl(program, "po_kernel", &err);
    logger.logCreateKernel(err);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    std::cout << "Allocating buffers..." << std::endl;
    size_t num = sizeof(float) * num_assets * num_prices;
    OCL_CHECK(err, cl::Buffer b_prices(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, num, prices.data(), &err));

    num = sizeof(float) * (num_assets + 2);
    OCL_CHECK(err,
              cl::Buffer b_gmpv_weights(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, num, gmvp_weights.data(), &err));
    OCL_CHECK(err,
              cl::Buffer b_eff_weights(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, num, eff_weights.data(), &err));
    num = sizeof(float) * (num_assets + 3);
    OCL_CHECK(err,
              cl::Buffer b_tan_weights(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, num, tan_weights.data(), &err));
    num = sizeof(float) * (num_assets);
    OCL_CHECK(err, cl::Buffer b_rf_weights(ctx, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, num, rf_weights.data(), &err));

    // Set the arguments
    OCL_CHECK(err, err = krnl.setArg(0, b_prices));
    OCL_CHECK(err, err = krnl.setArg(1, num_assets));
    OCL_CHECK(err, err = krnl.setArg(2, num_prices));
    OCL_CHECK(err, err = krnl.setArg(3, target_return));
    OCL_CHECK(err, err = krnl.setArg(4, risk_free_rate));
    OCL_CHECK(err, err = krnl.setArg(5, b_gmpv_weights));
    OCL_CHECK(err, err = krnl.setArg(6, b_eff_weights));
    OCL_CHECK(err, err = krnl.setArg(7, b_tan_weights));
    OCL_CHECK(err, err = krnl.setArg(8, b_rf_weights));

    // Copy input data to device global memory
    std::cout << "Migrate memory to device..." << std::endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({b_prices}, 0));
    OCL_CHECK(err, err = q.finish());

    // Launch the Kernel
    std::cout << "Launching kernel..." << std::endl;
    cl::Event event;
    uint64_t tstart, tend;
    OCL_CHECK(err, err = q.enqueueTask(krnl, NULL, &event));
    OCL_CHECK(err, err = q.finish());
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &tstart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &tend));
    auto duration_ns = tend - tstart;

    // Copy Result from Device Global Memory to Host Local Memory
    std::cout << "Migrate memory from device..." << std::endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({b_gmpv_weights}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({b_eff_weights}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({b_tan_weights}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({b_rf_weights}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());

    // check results GMVP
    int res = 0; // assume success
    std::vector<float> exp = {0.231147, -0.002218, -0.084974, 0.106045, 0.006200,
                              0.411148, -0.069371, -0.035975, 0.123788, 0.314209};
    std::cout << "Global Minimum Variance Portfolio" << std::endl;
    if (check_results("GMVP", gmvp_weights, exp, num_assets, 0.0090025, 0.000977613)) {
        res = 1;
    }

    // check results efficient portfolio
    std::vector<float> exp_eff = {0.603973, -0.042119, -0.081139, -0.044018, 0.207977,
                                  0.247975, -0.087848, -0.030071, 0.080708,  0.144561};
    std::cout << "Efficient Portfolio with target return of " << target_return << std::endl;
    if (check_results("Eff", eff_weights, exp_eff, num_assets, 0.02, 0.0014647)) {
        res = 1;
    }

    // check results tangency portfolio
    std::vector<float> exp_tan = {1.259484,  -0.112273, -0.074397, -0.307861, 0.562747,
                                  -0.038921, -0.120334, -0.019689, 0.004963,  -0.153718};
    std::cout << "Tangency Portfolio for risk free rate of " << risk_free_rate << std::endl;
    if (check_results("Tan", tan_weights, exp_tan, num_assets, 0.0393361, 0.00468327, 0.560187)) {
        res = 1;
    }

    // check results tangency and risk free portfolio with target return
    std::vector<float> exp_rf = {0.624221,  -0.055644, -0.036872, -0.152581, 0.278906,
                                 -0.019290, -0.059640, -0.009758, 0.002460,  -0.076185};
    std::cout << "Tangency Portfolio for risk free rate of " << risk_free_rate << " and target return " << target_return
              << std::endl;
    if (check_results("Eff Tan", rf_weights, exp_rf, num_assets)) {
        res = 1;
    }
    float rf = 0;
    for (int i = 0; i < num_assets; i++) {
        rf += rf_weights[i];
    }
    std::cout << "Eff Tan proportion in risk free = " << 1 - rf << std::endl;

    // kernel execution time
    std::cout << "Duration returned by profile API is " << (duration_ns * (1.0e-6)) << " ms" << std::endl;

    res ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
        : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return res;
}
