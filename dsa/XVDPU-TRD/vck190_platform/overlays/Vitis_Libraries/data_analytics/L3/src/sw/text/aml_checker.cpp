/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include <cassert>
#include <iostream>
#include <sstream>
#include <limits>

#include "xf_data_analytics/text/aml_checker.hpp"
#include "xcl2.hpp"
#include "CL/cl_ext_xilinx.h"

namespace xf {
namespace data_analytics {
namespace text {

int load_csv(const size_t max_entry_num,
             const size_t max_field_len,
             const std::string& file_path,
             const unsigned col,
             std::vector<std::string>& vec) {
    std::ifstream file(file_path.c_str());
    if (!file.is_open()) {
        std::cerr << "\nError:" << file_path << " file cannot be opened for reading!" << std::endl;
        return -1;
    }

    size_t n = 0;
    std::string line;
    // skip the header
    std::getline(file, line);
    while (std::getline(file, line)) {
        unsigned cur_col = 0;
        bool in_escape_str = false;
        std::stringstream ins(line), outs;
        char c;
        while (ins.get(c)) {
            if (c == '"') {
                if (ins.peek() == '"') {
                    // escaped, not state change, just unescape
                    ins.get(c);
                    outs.put(c);
                } else {
                    // toggle state
                    in_escape_str = !in_escape_str;
                }
            } else if (c == ',' && !in_escape_str) {
                if (cur_col == col) {
                    // already got our column
                    break;
                } else {
                    // empty buffer and continue to next column
                    cur_col++;
                    outs = std::stringstream();
                }
            } else if (c == '\r') {
                // in-case we are working with MS new line...
            } else {
                outs.put(c);
            }
        }
        assert(cur_col == col && "not enough column!");
        std::string data = outs.str();
        if (data.length() <= max_field_len) {
            vec.push_back(data);
            // std::cerr << line << "\n\t{" << data << "}" << std::endl;
            n++;
        }
        if (n > max_entry_num) {
            std::cout << "\nWarning: the input file " << file_path << " contains more enties than " << max_entry_num
                      << " which will not be added in this check." << std::endl;
            file.close();
            return 0;
        }
    }
    file.close();
    return 0;
}

namespace internal {

int min(int a, int b) {
    return (a < b ? a : b);
}

int abs(int a, int b) {
    return (a < b ? (b - a) : (a - b));
}

float similarity(std::string str1, std::string str2) {
    const int n = str1.length();
    const int m = str2.length();
    if (n == 0 || m == 0) return 0.0;

    int maxDistance = (int)(0.1 * min(n, m));

    if (maxDistance < abs(m, n)) return 0.0;

    std::vector<int> p(n + 1, 0);
    std::vector<int> d(n + 1, 0);

    for (int i = 0; i <= n; i++) p[i] = i;

    for (int j = 1; j <= m; j++) {
        int bestPossibleEditDistance = m;
        char t_j = str2.at(j - 1);
        d[0] = j;

        for (int i = 1; i <= n; i++) {
            if (t_j != str1.at(i - 1))
                d[i] = min(min(d[i - 1], p[i]), p[i - 1]) + 1;
            else
                d[i] = min(min(d[i - 1] + 1, p[i] + 1), p[i - 1]);
            bestPossibleEditDistance = min(bestPossibleEditDistance, d[i]);
        }

        if (j > maxDistance && bestPossibleEditDistance > maxDistance) return 0.0;

        std::swap_ranges(p.begin(), p.end(), d.begin());
    }

    return (1.0 - ((float)p[n] / (float)min(m, n)));
}

size_t getMaxDistance(size_t len) {
    return (len / 10);
}

int SwiftMT103CPUChecker::initialize(const std::string& stopKeywordsFileName,
                                     const std::string& peopleFileName,
                                     const std::string& entitiesFileName,
                                     const std::string& BICRefDataFileName) {
    std::vector<std::string> vec_people, vec_entity;
    // Read Watch List data
    int nerror = 0;
    std::cout << "Loading people.csv..." << std::flush;
    nerror = load_csv(max_validated_people, -1U, peopleFileName, 1, vec_people);
    if (nerror) {
        std::cout << "Failed to load file: people.csv\n";
        exit(1);
    } else
        std::cout << "completed\n";
    std::cout << "Loading stopkeywords.csv..." << std::flush;
    nerror = load_csv(max_validated_stopword, max_contain_len, stopKeywordsFileName, 1, vec_stopword);
    if (nerror) {
        std::cout << "Failed to load file: stopkeywords.csv\n";
        exit(1);
    } else
        std::cout << "completed\n";
    std::cout << "Loading entities.csv..." << std::flush;
    nerror = load_csv(max_validated_entity, -1U, entitiesFileName, 1, vec_entity);
    if (nerror) {
        std::cout << "Failed to load file: entities.csv\n";
        exit(1);
    } else
        std::cout << "completed\n";
    std::cout << "Loading BIC_ref_data.csv..." << std::flush;
    nerror = load_csv(max_validated_BIC, max_equan_len, BICRefDataFileName, 2, vec_bic);
    if (nerror) {
        std::cout << "Failed to load file: BIC_ref_data.csv\n";
        exit(1);
    } else
        std::cout << "completed\n";

    // do pre-sort on people LIST
    for (std::vector<std::string>::iterator it = vec_people.begin(); it != vec_people.end(); ++it) {
        size_t len = it->length();
        assert(len < max_people_len_in_char && "Defined <max_people_len_in_char> is not enough!");
        vec_grp_people[len].push_back(*it);
    }
    // do pre-sort on entity LIST
    for (std::vector<std::string>::iterator it = vec_entity.begin(); it != vec_entity.end(); ++it) {
        size_t len = it->length();
        assert(len < max_entity_len_in_char && "Defined <max_entity_len_in_char> is not enough!");
        vec_grp_entity[len].push_back(*it);
    }

    return nerror;
}

bool SwiftMT103CPUChecker::doFuzzyTask(int thread_id,
                                       const size_t upper_limit,
                                       const std::string& pattern,
                                       const std::vector<std::vector<std::string> >& vec_grp_str) {
    bool match = false;
    size_t len = pattern.length();
    size_t med = getMaxDistance(len);
    size_t start_len = (len > (upper_limit - 3) && len <= upper_limit) ? (upper_limit + 1) : (len - med);
    size_t end_len = len + med;

    for (size_t n = start_len; n <= end_len; n++) {
        std::vector<std::string> deny_list = vec_grp_str[n];
        int step = (deny_list.size() + totalThreadNum - 1) / totalThreadNum;
        int size = size_t(thread_id * step + step) > deny_list.size() ? (deny_list.size() - thread_id * step) : step;
        for (int i = thread_id * step; i < (thread_id * step + size); i++) {
            float sim = similarity(pattern, deny_list.at(i));
            if (sim >= 0.9) {
                match = true;
                break;
            }
        }

        if (match) break;
    }

    return match;
}

bool SwiftMT103CPUChecker::strFuzzy(const size_t upper_limit,
                                    const std::string& pattern,
                                    std::vector<std::vector<std::string> >& vec_grp_str) {
    for (unsigned i = 0; i < totalThreadNum; i++) {
        worker[i] = std::async(std::launch::async, &SwiftMT103CPUChecker::doFuzzyTask, this, i, upper_limit,
                               std::ref(pattern), std::ref(vec_grp_str));
    }
    bool sw_match = false;
    for (unsigned i = 0; i < totalThreadNum; i++) sw_match = sw_match || worker[i].get();
    return sw_match;
}

bool SwiftMT103CPUChecker::strEqual(const std::string& code1, const std::string& code2) {
    bool match[2] = {false, false};
    for (std::vector<std::string>::const_iterator it = vec_bic.begin(); it != vec_bic.end(); ++it) {
        if (code1.compare(*it) == 0) match[0] = true;
        if (code2.compare(*it) == 0) match[1] = true;
    }

    return (match[0] || match[1]);
}

bool SwiftMT103CPUChecker::strContain(const std::string& info) {
    bool match = false;
    for (std::vector<std::string>::const_iterator it = vec_stopword.begin(); it != vec_stopword.end(); ++it) {
        if (info.find(*it) != std::string::npos) match = true;
    }

    return match;
}

SwiftMT103CheckResult SwiftMT103CPUChecker::check(const SwiftMT103& t) {
    SwiftMT103CheckResult r = {t.id, 0, {0}, 0.0};
    r.matchField.resize(7);
    auto ts = std::chrono::high_resolution_clock::now();
    // check for field TransactionDescription
    r.matchField[0] = strContain(t.transactionDescription);
    // check for field swiftCode1 & swiftCode2
    r.matchField[1] = strEqual(t.swiftCode1, t.swiftCode2);
    // check for field nombrePersona1
    r.matchField[2] = strFuzzy(this->max_fuzzy_len, t.nombrePersona1, vec_grp_people);
    // check for field nombrePersona2
    r.matchField[3] = strFuzzy(this->max_fuzzy_len, t.nombrePersona2, vec_grp_people);
    // check for field bank1
    r.matchField[4] = strFuzzy(this->max_fuzzy_len, t.bank1, vec_grp_entity);
    // check for field bank2
    r.matchField[5] = strFuzzy(this->max_fuzzy_len, t.bank2, vec_grp_entity);
    r.isMatch =
        r.matchField[0] || r.matchField[1] || r.matchField[2] || r.matchField[3] || r.matchField[4] || r.matchField[5];
    auto te = std::chrono::high_resolution_clock::now();
    r.timeTaken = std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count() / 1000.0f;

    return r;
}

} // namespace internal

int getRange(std::string& input, std::vector<int>& vec_base, std::vector<int>& vec_offset, int& base, int& nrow) {
    int cnt = 0;
    int len = input.length();
    if (len == 0) {
        return -1;
    }
    int med = internal::getMaxDistance(len);

    base = vec_base[len - med] * 3; // as 128-bit data width in AXI
    for (int i = len - med; i <= len + med; i++) {
        cnt += vec_offset[i];
    }

    nrow = cnt * 3;

    return 0;
}

void SwiftMT103Checker::preSort(std::vector<std::vector<std::string> >& vec_grp_str1,
                                std::vector<std::vector<std::string> >& vec_grp_str2,
                                std::vector<std::vector<int> >& vec_base,
                                std::vector<std::vector<int> >& vec_offset) {
    this->sum_line[0] = 0;
    for (size_t i = 0; i <= this->max_fuzzy_len; i++) {
        vec_grp_str1[i] = this->vec_grp_people[i];
        int size = this->vec_grp_people[i].size();
        int delta = (size + this->PU_NUM - 1) / this->PU_NUM;

        vec_base[0][i] = this->sum_line[0];
        vec_offset[0][i] = delta;
        this->sum_line[0] += delta;
    }

    this->sum_line[1] = 0;
    for (size_t i = 0; i <= this->max_fuzzy_len; i++) {
        vec_grp_str2[i] = this->vec_grp_entity[i];
        int size = this->vec_grp_entity[i].size();
        int delta = (size + this->PU_NUM - 1) / this->PU_NUM;

        vec_base[1][i] = this->sum_line[1];
        vec_offset[1][i] = delta;
        this->sum_line[1] += delta;
    }
}

int SwiftMT103Checker::initialize(const std::string& xclbinPath,
                                  const std::string& stopKeywordsFileName,
                                  const std::string& peopleFileName,
                                  const std::string& entitiesFileName,
                                  const std::string& BICRefDataFileName,
                                  int cardID,
                                  bool user_setting) {
    SwiftMT103CPUChecker::initialize(stopKeywordsFileName, peopleFileName, entitiesFileName, BICRefDataFileName);

    // get devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    if (cardID >= devices.size()) {
        std::cout << "ERROR: device index out of range [0, " << devices.size() << ")" << std::endl;
        exit(1);
    }

    cl::Device device;
    if (!user_setting) {
        int i = 0;
        for (i = 0; i < devices.size(); i++) {
            std::string dev_n = devices[i].getInfo<CL_DEVICE_NAME>();
            if (dev_n.find("xilinx_u50_gen3x16_xdma_201920_3") != std::string::npos ||
                dev_n.find("xilinx_u200_xdma_201830_2") != std::string::npos ||
                dev_n.find("xilinx_u250_xdma_201830_2") != std::string::npos ||
                dev_n.find("aws-vu9p-f1") != std::string::npos) {
                cardID = i;
                std::cout << "Auto choose card:" << cardID << std::endl;
                break;
            }
        }
        if (i == devices.size()) {
            std::cout << "No supported device was found!" << std::endl;
            exit(1);
        }
    } else {
        std::cout << "User setting card:" << cardID << std::endl;
    }
    device = devices[cardID];
    cl_int err = 0;
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    // Creating Context and Command Queue for selected Device
    ctx = cl::Context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    if (err != CL_SUCCESS) {
        std::cout << "Context creation failed!" << std::endl;
        exit(1);
    }
    queue = cl::CommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    logger.logCreateCommandQueue(err);
    if (err != CL_SUCCESS) {
        std::cout << "CommandQueue creation failed!" << std::endl;
        exit(1);
    }
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    std::string xclbin_path;

    std::cout << "INFO: Found Device=" << devName << std::endl;
    if (xclbinPath == "") {
        xclbin_path = "/opt/xilinx/apps/vt_data_analytis/aml/share/xclbin/" + std::string("fuzzy_") + devName +
                      std::string(".xclbin");
    } else {
        xclbin_path = xclbinPath;
    }
    // Creating Context and Command Queue for selected Device
    ctx = cl::Context(device);
    queue = cl::CommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
#ifdef XVERBOSE
    std::cout << "INFO: Found Device=" << devName << std::endl;
#endif

    // Create program with given xclbin file
    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    devices[0] = device;
    prg = cl::Program(ctx, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);

    std::string krnl_name = "fuzzy_kernel";
    cl_uint cu_num;
    {
        cl_int err_code;
        cl::Kernel k = cl::Kernel(prg, krnl_name.c_str(), &err_code);
        if (err_code != CL_SUCCESS) {
            std::cout << "ERROR: failed to create kernel." << std::endl;
            exit(1);
        }
        k.getInfo<cl_uint>(CL_KERNEL_COMPUTE_UNIT_COUNT, &cu_num);
        std::cout << "INFO: " << krnl_name << " has " << cu_num << " CU(s)" << std::endl;
        if (cu_num == 4)
            boost = 1;
        else
            boost = 0;
    }

    for (cl_uint i = 0; i < cu_num; i++) {
        std::string krnl_full_name = krnl_name + ":{" + krnl_name + "_" + std::to_string(i + 1) + "}";
        fuzzy[i] = cl::Kernel(prg, krnl_full_name.c_str(), &err);
        logger.logCreateKernel(err);
    }

    // Do pre-group process for People deny list to boost Fuzzy function
    for (int i = 0; i < 2; i++) {
        vec_base[i].resize(40, 0);
        vec_offset[i].resize(40, 0);
    }

    std::cout << "Pre-sorting..." << std::flush;
    std::vector<std::vector<std::string> > vec_grp_str1(40);
    std::vector<std::vector<std::string> > vec_grp_str2(40);
    preSort(vec_grp_str1, vec_grp_str2, vec_base, vec_offset);
    int sum_line_num = sum_line[0] + sum_line[1];
    std::cout << "completed\n";

    char* csv_part[PU_NUM];
    for (int i = 0; i < PU_NUM; i++) {
        csv_part[i] = (char*)malloc(16 * 3 * sum_line_num);
    }

    for (uint32_t i = 0; i < vec_grp_str1.size(); i++) {
        for (int j = 0; j < vec_offset[0][i]; j++) {
            for (int p = 0; p < PU_NUM; p++) {
                std::string str = "";
                char len = 0;
                if (unsigned(PU_NUM * j + p) < vec_grp_str1[i].size()) {
                    str = vec_grp_str1[i][PU_NUM * j + p];
                    len = i;
                }

                char tmp[48] = {len, 0};
                for (int j = 1; j < 48; j++) {
                    if (j - 1 < len)
                        tmp[j] = str.at(j - 1);
                    else
                        tmp[j] = 0;
                }

                for (int m = 0; m < 3; m++) {
                    for (int k = 0; k < 16; k++) {
                        csv_part[p][16 * (3 * (vec_base[0][i] + j) + m) + 15 - k] = tmp[m * 16 + k];
                    }
                }
            }
        }
    }

    for (uint32_t i = 0; i < vec_grp_str2.size(); i++) {
        for (int j = 0; j < vec_offset[1][i]; j++) {
            for (int p = 0; p < PU_NUM; p++) {
                std::string str = "";
                char len = 0;
                if (unsigned(PU_NUM * j + p) < vec_grp_str2[i].size()) {
                    str = vec_grp_str2[i][PU_NUM * j + p];
                    len = i;
                }

                char tmp[48] = {len, 0};
                for (int j = 1; j < 48; j++) {
                    if (j - 1 < len)
                        tmp[j] = str.at(j - 1);
                    else
                        tmp[j] = 0;
                }

                for (int m = 0; m < 3; m++) {
                    for (int k = 0; k < 16; k++)
                        csv_part[p][16 * (3 * (sum_line[0] + vec_base[1][i] + j) + m) + 15 - k] = tmp[m * 16 + k];
                }
            }
        }
    }

    // Create device Buffer
    cl_mem_ext_ptr_t mext_i1[2], mext_i2[2], mext_o1[2], mext_o2[2];
    cl_mem_ext_ptr_t mext_i3[2], mext_i4[2], mext_o3[2], mext_o4[2];
    for (int i = 0; i < 2; i++) { // call kernel twice
        mext_i1[i] = {2, nullptr, fuzzy[0]()};
        mext_o1[i] = {11, nullptr, fuzzy[0]()};
        mext_i2[i] = {2, nullptr, fuzzy[1]()};
        mext_o2[i] = {11, nullptr, fuzzy[1]()};

        // create device buffer
        buf_field_i1[i] = cl::Buffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(uint32_t) * 9, &mext_i1[i]);
        buf_field_o1[i] = cl::Buffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY, sizeof(uint32_t), &mext_o1[i]);
        buf_field_i2[i] = cl::Buffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(uint32_t) * 9, &mext_i2[i]);
        buf_field_o2[i] = cl::Buffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY, sizeof(uint32_t), &mext_o2[i]);

        if (boost) { // set for kernel 3 & 4
            mext_i3[i] = {2, nullptr, fuzzy[2]()};
            mext_o3[i] = {11, nullptr, fuzzy[2]()};
            mext_i4[i] = {2, nullptr, fuzzy[3]()};
            mext_o4[i] = {11, nullptr, fuzzy[3]()};

            buf_field_i3[i] =
                cl::Buffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(uint32_t) * 9, &mext_i3[i]);
            buf_field_o3[i] = cl::Buffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY, sizeof(uint32_t), &mext_o3[i]);
            buf_field_i4[i] =
                cl::Buffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, sizeof(uint32_t) * 9, &mext_i4[i]);
            buf_field_o4[i] = cl::Buffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY, sizeof(uint32_t), &mext_o4[i]);
        }
    }

    cl_mem_ext_ptr_t mext_t[4 * PU_NUM];
    for (int i = 0; i < PU_NUM; i++) { // kernel-0
        mext_t[i] = {unsigned(i + 3), csv_part[i], fuzzy[0]()};
    }
    for (int i = 0; i < PU_NUM; i++) { // kernel-1
        mext_t[PU_NUM + i] = {unsigned(i + 3), csv_part[i], fuzzy[1]()};
    }
    if (boost) {
        for (int i = 0; i < PU_NUM; i++) { // kernel-2
            mext_t[2 * PU_NUM + i] = {unsigned(i + 3), csv_part[i], fuzzy[2]()};
        }
        for (int i = 0; i < PU_NUM; i++) { // kernel-3
            mext_t[3 * PU_NUM + i] = {unsigned(i + 3), csv_part[i], fuzzy[3]()};
        }
    }

    int dup = (boost == 0) ? 2 : 4;
    for (int i = 0; i < dup * PU_NUM; i++) {
        buf_csv[i] = cl::Buffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                (size_t)16 * 3 * sum_line_num, &mext_t[i]);
    }

    // h2d Transfer
    std::vector<cl::Memory> ob_in;
    for (int i = 0; i < dup * PU_NUM; i++) ob_in.push_back(buf_csv[i]);
    queue.enqueueMigrateMemObjects(ob_in, 0, nullptr, nullptr);
    queue.finish();

    // free memory
    for (int i = 0; i < PU_NUM; i++) free(csv_part[i]);

    return 0;
}

SwiftMT103CheckResult SwiftMT103Checker::check(const SwiftMT103& t) {
    uint32_t buf_f_i0[2][9]; // {<Persona1>, <Bank1>}
    uint32_t buf_f_i1[2][9]; // {<Persona2>, <Bank2>}
    uint32_t buf_f_o0[2] = {0, 0};
    uint32_t buf_f_o1[2] = {0, 0};
    uint32_t buf_f_o2[2] = {0, 0};
    uint32_t buf_f_o3[2] = {0, 0};

    std::string name_list[4] = {"nombrePersona1", "nombrePersona2", "bank1", "bank2"};
    std::string vec_str1[4] = {t.nombrePersona1, t.nombrePersona2, t.bank1, t.bank2};
    for (int i = 0; i < 4; i++) {
        std::string str = vec_str1[i];
        char len = str.length() > 35 ? 35 : str.length(); // truncate
        char tmp[36] = {len, str[0], str[1], str[2], 0};
        for (int j = 4; j < 36; j++) {
            if (j - 1 < len)
                tmp[j] = str.at(j - 1);
            else
                tmp[j] = 0;
        }

        for (int m = 0; m < 9; m++) {
            char str_tmp[4];
            for (int k = 0; k < 4; k++) str_tmp[3 - k] = tmp[m * 4 + k];
            if (i % 2 == 0)
                memcpy(&buf_f_i0[i / 2][m], str_tmp, 4);
            else
                memcpy(&buf_f_i1[i / 2][m], str_tmp, 4);
        }
    }

    bool skip_field[4] = {false, false, false, false};
    bool sw_fuzzy_result[4] = {false, false, false, false};
    int base_trans[2 * 2], nrow_trans[2 * 2];
    for (int i = 0; i < 2 * 2; i++) {
        if (vec_str1[i].length() > (max_fuzzy_len - 3)) {
            if (i < 2)
                sw_fuzzy_result[i] = strFuzzy(max_fuzzy_len, vec_str1[i], vec_grp_people);
            else
                sw_fuzzy_result[i] = strFuzzy(max_fuzzy_len, vec_str1[i], vec_grp_entity);
        }

        if (vec_str1[i].length() > max_fuzzy_len || vec_str1[i].length() == 0) // disable this fuzzy on FPGA
            skip_field[i] = true;
        else
            getRange(vec_str1[i], vec_base[i / 2], vec_offset[i / 2], base_trans[i], nrow_trans[i]);
    }
    base_trans[2] += sum_line[0] * 3;
    base_trans[3] += sum_line[0] * 3;
    // std::cout << "i:" << t.id << " " << skip_field[0] << " " << skip_field[1] << " " << skip_field[2] << " "
    //          << skip_field[3] << " | " << sw_fuzzy_result[0] << " " << sw_fuzzy_result[1] << " " <<
    //          sw_fuzzy_result[2]
    //          << " " << sw_fuzzy_result[3] << std::endl;

    int dup = (boost == 0) ? 2 : 4;
    for (int i = 0; i < 2; i++) {
        events_write[i].resize(dup);
        events_kernel[i].resize(dup);
        events_read[i].resize(dup);
    }

    // struct timeval start_time, end_time;
    // std::cout << "INFO: kernel start------" << std::endl;
    // gettimeofday(&start_time, 0);

    // launch kernel and calculate kernel execution time
    for (int i = 0; i < 2; i++) {
        if (i == 0) {
            if (!skip_field[2 * i])
                queue.enqueueWriteBuffer(buf_field_i1[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i0[i], nullptr,
                                         &events_write[i][0][0]);
            if (!skip_field[2 * i + 1])
                queue.enqueueWriteBuffer(buf_field_i2[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i1[i], nullptr,
                                         &events_write[i][1][0]);
            if (boost) {
                if (!skip_field[2 * i])
                    queue.enqueueWriteBuffer(buf_field_i3[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i0[i], nullptr,
                                             &events_write[i][2][0]);
                if (!skip_field[2 * i + 1])
                    queue.enqueueWriteBuffer(buf_field_i4[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i1[i], nullptr,
                                             &events_write[i][3][0]);
            }
        } else {
            if (skip_field[0] && !skip_field[2 * i])
                queue.enqueueWriteBuffer(buf_field_i1[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i0[i], nullptr,
                                         &events_write[i][0][0]);
            else if (!skip_field[2 * i])
                queue.enqueueWriteBuffer(buf_field_i1[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i0[i],
                                         &events_read[i - 1][0], &events_write[i][0][0]);

            if (skip_field[1] && !skip_field[2 * i + 1])
                queue.enqueueWriteBuffer(buf_field_i2[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i1[i], nullptr,
                                         &events_write[i][1][0]);
            else if (!skip_field[2 * i + 1])
                queue.enqueueWriteBuffer(buf_field_i2[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i1[i],
                                         &events_read[i - 1][1], &events_write[i][1][0]);
            if (boost) {
                if (skip_field[0] && !skip_field[2 * i])
                    queue.enqueueWriteBuffer(buf_field_i3[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i0[i], nullptr,
                                             &events_write[i][2][0]);
                else if (!skip_field[2 * i])
                    queue.enqueueWriteBuffer(buf_field_i3[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i0[i],
                                             &events_read[i - 1][2], &events_write[i][2][0]);

                if (skip_field[1] && !skip_field[2 * i + 1])
                    queue.enqueueWriteBuffer(buf_field_i4[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i1[i], nullptr,
                                             &events_write[i][3][0]);
                else if (!skip_field[2 * i + 1])
                    queue.enqueueWriteBuffer(buf_field_i4[i], CL_FALSE, 0, sizeof(uint32_t) * 9, buf_f_i1[i],
                                             &events_read[i - 1][3], &events_write[i][3][0]);
            }
        }

        if (boost) {
            int nrow00 = (1 + (nrow_trans[i * 2] / 3)) / 2 * 3;
            int nrow01 = nrow_trans[i * 2] - nrow00;
            int nrow10 = (1 + (nrow_trans[i * 2 + 1] / 3)) / 2 * 3;
            int nrow11 = nrow_trans[i * 2 + 1] - nrow10;

            int j = 0;
            fuzzy[0].setArg(j++, base_trans[i * 2]);
            fuzzy[0].setArg(j++, nrow00);
            fuzzy[0].setArg(j++, buf_field_i1[i]);
            for (int k = 0; k < PU_NUM; k++) fuzzy[0].setArg(j++, buf_csv[k]);
            fuzzy[0].setArg(j++, buf_field_o1[i]);

            j = 0;
            fuzzy[1].setArg(j++, base_trans[i * 2 + 1]);
            fuzzy[1].setArg(j++, nrow10);
            fuzzy[1].setArg(j++, buf_field_i2[i]);
            for (int k = PU_NUM; k < 2 * PU_NUM; k++) fuzzy[1].setArg(j++, buf_csv[k]);
            fuzzy[1].setArg(j++, buf_field_o2[i]);

            j = 0;
            fuzzy[2].setArg(j++, (base_trans[i * 2] + nrow00));
            fuzzy[2].setArg(j++, nrow01);
            fuzzy[2].setArg(j++, buf_field_i3[i]);
            for (int k = 2 * PU_NUM; k < 3 * PU_NUM; k++) fuzzy[2].setArg(j++, buf_csv[k]);
            fuzzy[2].setArg(j++, buf_field_o3[i]);

            j = 0;
            fuzzy[3].setArg(j++, (base_trans[i * 2 + 1] + nrow10));
            fuzzy[3].setArg(j++, nrow11);
            fuzzy[3].setArg(j++, buf_field_i4[i]);
            for (int k = 3 * PU_NUM; k < 4 * PU_NUM; k++) fuzzy[3].setArg(j++, buf_csv[k]);
            fuzzy[3].setArg(j++, buf_field_o4[i]);
        } else {
            int j = 0;
            fuzzy[0].setArg(j++, base_trans[i * 2]);
            fuzzy[0].setArg(j++, nrow_trans[i * 2]);
            fuzzy[0].setArg(j++, buf_field_i1[i]);
            for (int k = 0; k < PU_NUM; k++) fuzzy[0].setArg(j++, buf_csv[k]);
            fuzzy[0].setArg(j++, buf_field_o1[i]);

            j = 0;
            fuzzy[1].setArg(j++, base_trans[i * 2 + 1]);
            fuzzy[1].setArg(j++, nrow_trans[i * 2 + 1]);
            fuzzy[1].setArg(j++, buf_field_i2[i]);
            for (int k = PU_NUM; k < 2 * PU_NUM; k++) fuzzy[1].setArg(j++, buf_csv[k]);
            fuzzy[1].setArg(j++, buf_field_o2[i]);
        }

        if (!skip_field[2 * i]) queue.enqueueTask(fuzzy[0], &events_write[i][0], &events_kernel[i][0][0]);
        if (!skip_field[2 * i + 1]) queue.enqueueTask(fuzzy[1], &events_write[i][1], &events_kernel[i][1][0]);
        if (boost) {
            if (!skip_field[2 * i]) queue.enqueueTask(fuzzy[2], &events_write[i][2], &events_kernel[i][2][0]);
            if (!skip_field[2 * i + 1]) queue.enqueueTask(fuzzy[3], &events_write[i][3], &events_kernel[i][3][0]);
        }

        if (!skip_field[2 * i])
            queue.enqueueReadBuffer(buf_field_o1[i], CL_FALSE, 0, sizeof(uint32_t), &buf_f_o0[i], &events_kernel[i][0],
                                    &events_read[i][0][0]);
        if (!skip_field[2 * i + 1])
            queue.enqueueReadBuffer(buf_field_o2[i], CL_FALSE, 0, sizeof(uint32_t), &buf_f_o1[i], &events_kernel[i][1],
                                    &events_read[i][1][0]);
        if (boost) {
            if (!skip_field[2 * i])
                queue.enqueueReadBuffer(buf_field_o3[i], CL_FALSE, 0, sizeof(uint32_t), &buf_f_o2[i],
                                        &events_kernel[i][2], &events_read[i][2][0]);
            if (!skip_field[2 * i + 1])
                queue.enqueueReadBuffer(buf_field_o4[i], CL_FALSE, 0, sizeof(uint32_t), &buf_f_o3[i],
                                        &events_kernel[i][3], &events_read[i][3][0]);
        }
    }
    queue.flush();
    queue.finish();

    bool sw_equal, sw_contain;
    sw_equal = strEqual(t.swiftCode1, t.swiftCode2);
    sw_contain = strContain(t.transactionDescription);

    SwiftMT103CheckResult r;
    r.id = t.id;
    r.isMatch = 0;
    r.matchField.resize(7);
    for (int i = 0; i < 7; i++) r.matchField[i] = 0;

    if (sw_contain) r.matchField[0] = 1;                                                 // description
    if (sw_equal) r.matchField[1] = 1;                                                   // swiftcode
    if (sw_fuzzy_result[0] || buf_f_o0[0] == 1 || buf_f_o2[0] == 1) r.matchField[2] = 1; // person1
    if (sw_fuzzy_result[1] || buf_f_o1[0] == 1 || buf_f_o3[0] == 1) r.matchField[3] = 1; // person2
    if (sw_fuzzy_result[2] || buf_f_o0[1] == 1 || buf_f_o2[1] == 1) r.matchField[4] = 1; // bank1
    if (sw_fuzzy_result[3] || buf_f_o1[1] == 1 || buf_f_o3[1] == 1) r.matchField[5] = 1; // bank2
    if (sw_contain || sw_equal || buf_f_o0[0] == 1 || buf_f_o1[0] == 1 || buf_f_o0[1] == 1 || buf_f_o1[1] == 1 ||
        buf_f_o2[0] == 1 || buf_f_o3[0] == 1 || buf_f_o2[1] == 1 || buf_f_o3[1] == 1 || sw_fuzzy_result[0] ||
        sw_fuzzy_result[1] || sw_fuzzy_result[2] || sw_fuzzy_result[3])
        r.isMatch = 1;

    return r;
}

} // namespace text
} // namespace data_analytics
} // namespace xf
