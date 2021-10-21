#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <ap_int.h>
#include "x_utils.hpp"
#define INSERT_LEN 1024
#define LEN (INSERT_LEN * 4 * 32) // max length support: 1024*4*512
typedef int KEY_TYPE;
template <typename T>
int load_dat(T* data, const std::string& file_path, size_t n) {
    if (!data) {
        return -1;
    }

    FILE* f = fopen(file_path.c_str(), "rb");
    if (!f) {
        std::cerr << "ERROR: " << file_path << " cannot be opened for binary read." << std::endl;
    }
    size_t cnt = fread((void*)data, sizeof(T), n, f);
    fclose(f);
    if (cnt != n) {
        std::cerr << "ERROR: " << cnt << " entries read from " << file_path << ", " << n << " entries required."
                  << std::endl;
        return -1;
    }
    return 0;
}

template <typename T>
int gen_dat(T* data, const std::string& file_path, size_t n) {
    if (!data) {
        return -1;
    }

    FILE* f = fopen(file_path.c_str(), "wb");
    if (!f) {
        std::cerr << "ERROR: " << file_path << " cannot be opened for binary read." << std::endl;
    }
    size_t cnt = fwrite((void*)data, sizeof(T), n, f);
    fclose(f);
    if (cnt != n) {
        std::cerr << "ERROR: " << cnt << " entries read from " << file_path << ", " << n << " entries required."
                  << std::endl;
        return -1;
    }
    return 0;
}
int main(int argc, const char* argv[]) {
    srand(time(NULL));
    x_utils::ArgParser parser(argc, argv);
    std::string size_s = "1"; //
    int size = 1024 * 1024;
    if (!parser.getCmdOption("-ss", size_s)) {
        size = 1024 * 1024;
    } else {
        size *= std::stoi(size_s);
    }
    std::string out_dir = "./";
    if (!parser.getCmdOption("-out", out_dir)) {
        out_dir = "./";
    }

    int loop_num;
    std::string loop_num_str;
    if (!parser.getCmdOption("-ln", loop_num_str)) {
        loop_num_str = "1";
    }
    loop_num = std::stoi(loop_num_str);
    std::cout << "num_iters:" << loop_num << ", size:" << size / 1024 / 1024 << "M" << std::endl;

    std::cout << "RAND_MAX = " << RAND_MAX << std::endl;
    for (int i = 0; i < loop_num; i++) {
        std::cout << "generating " << i << std::endl;
        ap_uint<64>* inKey_alloc = (ap_uint<64>*)malloc(sizeof(ap_uint<64>) * size);

        for (int j = 0; j < size; j++) {
            // rnd generated
            int tmp = rand();
            inKey_alloc[j].range(31, 0) = tmp;
            inKey_alloc[j].range(63, 32) = tmp + 1;
        }

        gen_dat<ap_uint<64> >(inKey_alloc, out_dir + "/input_" + size_s + "M_" + std::to_string(i) + ".dat", size);
    }
    std::cout << "Generated " << loop_num << " " << size_s << "M int pair data" << std::endl;
}
