/*
 * Copyright 2021 Xilinx, Inc.
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

#include "predicate.hpp"

namespace dup_match {
namespace internal {

// creat inverted index of 2-gram
void TwoGramPredicate::index(const std::vector<std::string>& column) {
    std::map<std::string, uint32_t> col_map;
    std::vector<std::pair<std::string, uint32_t> > unique_field;
    uint32_t id = 0;
    // deduplication field
    for (uint32_t i = 0; i < column.size(); i++) {
        std::string line_str = column[i];
        std::string pre_str = preTwoGram(line_str);
        std::map<std::string, uint32_t>::iterator iter = col_map.find(pre_str);
        if (iter == col_map.end()) {
            unique_field.push_back(std::pair<std::string, int>(pre_str, id)); // get unique field
            col_map[pre_str] = id++;
        }
    }
    std::cout << "TwoGramPredicate: column map size=" << col_map.size() << std::endl;

    uuMT tmap;
    std::vector<std::vector<udPT> > word_info;
    uint32_t wid = 0;
    // calculate idf for creating index
    for (int u = 0; u < unique_field.size(); u++) {
        std::vector<uint16_t> terms;
        twoGram(unique_field[u].first, terms); // split terms based on 2-gram
        std::vector<uint32_t> wids;
        uuMT dict; // terms dictionary
        for (int i = 0; i < terms.size(); i++) {
            uint16_t term = terms[i];
            uuMT::iterator iter = tmap.find(term);
            if (iter == tmap.end()) { // add
                tmap[term] = wid;
                wids.push_back(wid);
                dict[wid] = 1;
                wid++;
            } else { // update
                wids.push_back(iter->second);
                dict[iter->second]++;
            }
        }
        // std::cout << "dict size=" << dict.size() << std::endl;
        double W = 0.0;
        std::vector<udPT> udpt;
        for (uuMT::iterator iter = dict.begin(); iter != dict.end(); iter++) {
            double w = std::log(iter->second) + 1.0;
            W += w * w;
            udpt.push_back(udPT(iter->first, w));
            // std::cout << "count=" << iter->second << ", w=" << w << std::endl;
        }
        W = std::sqrt(W);
        for (int i = 0; i < udpt.size(); i++) {
            double temp = udpt[i].second / W;
            if (udpt[i].first >= word_info.size()) {
                word_info.resize(word_info.size() + 256);
            }
            word_info[udpt[i].first].push_back(udPT(u, temp));
            // std::cout << "w[" << i << "]=" << udpt[i].second / W << std::endl;
        }
    }

    int N = unique_field.size();
    // set mux index number
    int threshold = int(1000 > N * 0.05 ? 1000 : N * 0.05);
    std::cout << "threshold=" << threshold << std::endl;
    for (int i = 0; i < 4096; i++) {
        idf_value_[i] = 0.0;
        tf_addr_[i] = 0;
    }
    // calculate tf for creating index
    uint64_t begin = 0, end = 0, skip = 0;
    for (uuMT::iterator iter = tmap.begin(); iter != tmap.end(); iter++) {
        int sn = iter->first;
        int wid = iter->second;
        int size = word_info[wid].size();
        if (size > threshold) {
            // std::cout << "size=" << size << ", threshold=" << threshold << std::endl;
            skip++;
            continue;
        }
        idf_value_[sn] = std::log(1.0 + (double)N / size);
        begin = end;
        for (int j = 0; j < size; j++) {
            tf_value_[end++] = word_info[wid][j].first;
            DTConvert64 dtc;
            dtc.dt1 = word_info[wid][j].second;
            tf_value_[end++] = dtc.dt0;
        }
        tf_addr_[sn] = (begin >> 1) + (end << 31);
    }
    std::cout << "tf_value_ size is " << end / 2 << ", index count=" << col_map.size() << ", term count=" << tmap.size()
              << ", skip=" << skip << std::endl;
}

// character encode: '0'~'9' -> 0~9, 'a'/'A'~'z'/'Z' -> 10~25, others -> 36
char TwoGramPredicate::charEncode(char in) {
    char out;
    if (in >= 48 && in <= 57)
        out = in - 48;
    else if (in >= 97 && in <= 122)
        out = in - 87;
    else if (in >= 65 && in <= 90)
        out = in - 55;
    else
        out = 36;
    return out;
}

// splite string into 2-gram terms
void TwoGramPredicate::twoGram(std::string& inStr, std::vector<uint16_t>& terms) {
    for (int i = 0; i < inStr.size(); i += 2) {
        uint16_t term = charEncode(inStr[i]) + charEncode(inStr[i + 1]) * 64;
        terms.push_back(term);
    }
}

char TwoGramPredicate::charFilter(char in) {
    char out;
    if (in >= 48 && in <= 57) // 0~9
        out = in;
    else if (in >= 97 && in <= 122) // a~z
        out = in;
    else if (in >= 65 && in <= 90) // A~Z
        out = in + 32;
    else if (in == 32 || in == 10) // space & newline
        out = 32;
    else
        out = 255; // drop
    return out;
}

// split field into term of 2 chars, and merge terms into one string after sort terms
std::string TwoGramPredicate::preTwoGram(std::string& inStr) {
    uint8_t last_code = 32;
    std::string pre_str = "";
    for (int i = 0; i < inStr.size(); i++) {
        uint8_t ch = inStr[i];
        uint8_t ch_code = charFilter(ch);
        if (ch_code == 255) continue;
        if (!(ch_code == last_code && ch_code == 32)) {
            pre_str += ch_code;
        }
        last_code = ch_code;
    }
    if (last_code == 32) pre_str.pop_back();
    std::vector<uint16_t> terms;
    for (int i = 1; i < pre_str.size(); i++) {
        uint16_t term = pre_str[i - 1] + pre_str[i] * 256;
        terms.push_back(term);
    }
    std::sort(terms.begin(), terms.end());
    std::string out_str = "";
    for (int i = 0; i < terms.size(); i++) {
        uint8_t ch = terms[i] % 256;
        out_str += ch;
        ch = terms[i] / 256;
        out_str += ch;
    }
    return out_str;
}

// call Two Gram Predicate (TGP) Kernel
void TwoGramPredicate::search(std::string& xclbinPath, std::vector<std::string>& column, uint32_t* indexId[2]) {
    std::vector<ConfigParam> config(CU);
    uint8_t* fields[CU];
    uint32_t* offsets[CU];
    uint32_t* index_id[CU];
    uint32_t blk_sz = column.size() / CU;
    for (int i = 0; i < CU; i++) {
        fields[i] = aligned_alloc<uint8_t>(BS);
        offsets[i] = aligned_alloc<uint32_t>(RN);
        int offset = 0;
        int begin = blk_sz * i;
        int end = blk_sz * i + blk_sz;
        if (i == CU - 1) end = column.size();
        for (int j = begin; j < end; j++) {
            memcpy(fields[i] + offset, column[j].data(), column[j].size());
            offset += column[j].size();
            offsets[i][j - begin] = offset;
        }
        config[i].docSize = end - begin;
        config[i].fldSize = offset;
        std::cout << "config=" << config[i].docSize << ", " << config[i].fldSize << std::endl;
    }
    // do pre-process on CPU
    // platform related operations
    struct timeval tk1, tk2;
    gettimeofday(&tk1, 0);
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl_int cl_err;
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);

    // Creating Context and Command Queue for selected Device
    cl::Context context(device, NULL, NULL, NULL, &cl_err);
    logger.logCreateContext(cl_err);
    queue_ = new cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                  &cl_err);
    logger.logCreateCommandQueue(cl_err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("Found Device=%s\n", devName.c_str());

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbinPath);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &cl_err);
    logger.logCreateProgram(cl_err);
    std::vector<cl::Kernel> PKernel(2);
    PKernel[0] = cl::Kernel(program, "TGP_Kernel:{TGP_Kernel_1}", &cl_err);
    logger.logCreateKernel(cl_err);
    PKernel[1] = cl::Kernel(program, "TGP_Kernel:{TGP_Kernel_2}", &cl_err);
    logger.logCreateKernel(cl_err);
    std::cout << "kernel has been created" << std::endl;

    std::vector<std::array<cl_mem_ext_ptr_t, 6> > mext_o(CU);
    std::vector<std::array<cl::Buffer, 6> > buff(CU);
    for (int i = 0; i < CU; i++) {
        mext_o[i][0] = {1, fields[i], PKernel[i]()};
        mext_o[i][1] = {2, offsets[i], PKernel[i]()};
        mext_o[i][2] = {3, idf_value_, PKernel[i]()};
        mext_o[i][3] = {4, tf_addr_, PKernel[i]()};
        mext_o[i][4] = {5, tf_value_, PKernel[i]()};
        mext_o[i][5] = {9, indexId[i], PKernel[i]()};

        // create device buffer and map dev buf to host buf
        // input buffer
        buff[i][0] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(uint8_t) * BS, &mext_o[i][0]);
        buff[i][1] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(uint32_t) * RN, &mext_o[i][1]);
        buff[i][2] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(double) * 4096, &mext_o[i][2]);
        buff[i][3] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(uint64_t) * 4096, &mext_o[i][3]);
        buff[i][4] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(uint64_t) * TFLEN, &mext_o[i][4]);
        // output buffer
        buff[i][5] = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                sizeof(uint32_t) * RN, &mext_o[i][5]);
        // push buffer
        for (int j = 0; j < 5; j++) {
            ob_in.push_back(buff[i][j]);
        }
        ob_out.push_back(buff[i][5]);

        // launch kernel and calculate kernel execution time
        int j = 0;
        PKernel[i].setArg(j++, config[i]);
        for (int k = 0; k < 5; k++) {
            PKernel[i].setArg(j++, buff[i][k]);
        }
        PKernel[i].setArg(j++, buff[i][4]);
        PKernel[i].setArg(j++, buff[i][4]);
        PKernel[i].setArg(j++, buff[i][4]);
        PKernel[i].setArg(j++, buff[i][5]);
    }

    events_write.resize(1);
    events_kernel.resize(CU);
    events_read.resize(1);
    std::cout << "kernel start------" << std::endl;
    queue_->enqueueMigrateMemObjects(ob_in, 0, nullptr, &events_write[0]);
    for (int i = 0; i < CU; i++) queue_->enqueueTask(PKernel[i], &events_write, &events_kernel[i]);
    queue_->enqueueMigrateMemObjects(ob_out, 1, &events_kernel, &events_read[0]);
    // queue_->finish();
    // std::cout << "kernel end------" << std::endl;

    // unsigned long time1, time2, total_time;
    // events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    // events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    // std::cout << "Write DDR Execution time " << (time2 - time1) / 1000000.0 << " ms" << std::endl;
    // total_time = time2 - time1;
    // events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    // events_kernel[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    // std::cout << "Kernel Execution time " << (time2 - time1) / 1000000.0 << " ms" << std::endl;
    // total_time += time2 - time1;
    // events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    // events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    // std::cout << "Read DDR Execution time " << (time2 - time1) / 1000000.0 << " ms" << std::endl;
    // events_write[0].getProfilingInfo(CL_PROFILING_COMMAND_START, &time1);
    // events_read[0].getProfilingInfo(CL_PROFILING_COMMAND_END, &time2);
    // total_time = time2 - time1;
    // std::cout << "FPGA Execution time " << total_time / 1000000.0 << " ms" << std::endl;
    // gettimeofday(&tk2, 0);
    // std::cout << "Kernel Execution time " << tvdiff(&tk1, &tk2) / 1000.0 << "ms" << std::endl;
}

// creat word inverted index
void WordPredicate::index(const std::vector<std::string>& column) {
    std::vector<std::pair<std::string, uint32_t> > unique_fields;
    uint32_t id = 0;
    // deduplication field
    for (int i = 0; i < column.size(); i++) {
        std::string lineStr = column[i];
        std::vector<std::string> terms;
        std::string s_str;
        splitWord(lineStr, terms, s_str);
        MT::iterator iter = doc_to_id_.find(s_str); // doc_to_id_.insert belong to C++17
        if (iter == doc_to_id_.end()) {             // add
            // std::cout << ",id=" << id << ", 1: " << lineStr << ", 2: " << s_str << std::endl;
            unique_fields.push_back(std::pair<std::string, int>(s_str, id));
            doc_to_id_[s_str] = id++;
        }
    }

    uint32_t wid = 0, did = 0;
    // calculate idf for creating index
    for (int did = 0; did < unique_fields.size(); did++) { // doc id
        std::vector<std::string> terms;
        std::string s_str;
        splitWord(unique_fields[did].first, terms, s_str);
        std::vector<uint32_t> wids;
        uuMT dict; // terms dictionary
        for (int i = 0; i < terms.size(); i++) {
            // std::cout << "terms[" << i << "]=" << terms[i] << std::endl;
            MT::iterator iter = term_to_id_.find(terms[i]);
            if (iter == term_to_id_.end()) { // add
                term_to_id_[terms[i]] = wid;
                wids.push_back(wid);
                dict[wid] = 1;
                wid++;
            } else { // update
                wids.push_back(iter->second);
                dict[iter->second]++;
            }
        }
        // std::cout << "dict size=" << dict.size() << std::endl;
        double W = 0.0;
        std::vector<udPT> udpt;
        for (uuMT::iterator iter = dict.begin(); iter != dict.end(); iter++) {
            double w = std::log(iter->second) + 1.0;
            W += w * w;
            udpt.push_back(udPT(iter->first, w));
        }
        W = std::sqrt(W);
        for (int i = 0; i < udpt.size(); i++) {
            double temp = udpt[i].second / W;
            if (udpt[i].first >= tf_value_.size()) {
                tf_value_.resize(tf_value_.size() + 256);
            }
            // std::cout << "tf_value_ size=" << tf_value_.size() << ", udpt[i].first=" << udpt[i].first << std::endl;
            tf_value_[udpt[i].first].push_back(udPT(did, temp));
        }
    }
    int N = doc_to_id_.size();
    int threshold = int(1000 > N * 0.05 ? 1000 : N * 0.05); // set mux index number
    std::cout << "threshold=" << threshold << std::endl;
    // calculate tf for creating index
    uint64_t skip = 0, cnt = 0;
    idf_value_.resize(term_to_id_.size());
    for (MT::iterator iter = term_to_id_.begin(); iter != term_to_id_.end(); iter++) {
        // std::cout << "term_to_id_[" << cnt++ << "]=" << iter->first << ", " << iter->second << std::endl;
        std::string term = iter->first;
        int wid = iter->second;
        int size = tf_value_[wid].size();
        if (size > threshold) {
            // std::cout << "size=" << size << ", threshold=" << threshold << std::endl;
            tf_value_[wid].clear();
            skip++;
            continue;
        }
        idf_value_[wid] = std::log(1.0 + (double)N / size);
    }
    std::cout << "index count=" << doc_to_id_.size() << ", term count=" << term_to_id_.size() << ", skip=" << skip
              << std::endl;
}

// search index id
void WordPredicate::search(std::vector<std::string>& column, std::vector<uint32_t>& indexId) {
    std::vector<int> canopy(doc_to_id_.size(), -1);
    indexId.resize(column.size());
    for (int i = 0; i < column.size(); i++) {
        std::string line_str = column[i];
        std::vector<std::string> terms;
        std::string s_str;
        splitWord(line_str, terms, s_str);
        uint32_t doc_id = doc_to_id_[s_str];
        // std:: cout << "\n---- i=" << i << ", doc_id=" << doc_id << ", s_str=" << s_str << std::endl;
        if (canopy[doc_id] == -1) {
            std::vector<uint32_t> vec_wid(terms.size(), -1);
            for (int j = 0; j < terms.size(); j++) {
                std::map<std::string, uint32_t>::iterator iter = term_to_id_.find(terms[j]);
                if (iter == term_to_id_.end()) continue;
                vec_wid[j] = iter->second;
            }
            std::sort(vec_wid.begin(), vec_wid.end());
            uint32_t wid_now, wid_last;
            double threshold = 0.0;
            int cnt = 1;
            std::vector<std::vector<udPT> > tf_value_l;
            for (int j = 0; j < vec_wid.size() + 1; j++) {
                if (j == vec_wid.size()) {
                    // std::cout << "wid_last=" << wid_last << std::endl;
                    double weight = idf_value_[wid_last];
                    threshold += weight * weight * cnt;
                    weight *= cnt;
                    tf_value_l.push_back(tf_value_[wid_last]);
                    for (int k = 0; k < tf_value_l[tf_value_l.size() - 1].size(); k++)
                        tf_value_l[tf_value_l.size() - 1][k].second *= weight;
                } else if (j < vec_wid.size()) {
                    wid_now = vec_wid[j];
                    if (j > 0) {
                        if (wid_now == wid_last)
                            cnt++;
                        else {
                            // std::cout << "wid_last=" << wid_last << std::endl;
                            double weight = idf_value_[wid_last];
                            threshold += weight * weight * cnt;
                            weight *= cnt;
                            tf_value_l.push_back(tf_value_[wid_last]);
                            for (int k = 0; k < tf_value_l[tf_value_l.size() - 1].size(); k++)
                                tf_value_l[tf_value_l.size() - 1][k].second *= weight;
                            // tf_value_l[tf_value_l.size() - 1].push_back(udPT(-1, weight * cnt));
                            cnt = 1;
                        }
                    }
                }
                wid_last = wid_now;
            }
            // if (LOG > 1) std::cout << "threshold[" << i << "]=" << threshold << std::endl;
            threshold = std::sqrt(threshold) * 0.8;
            std::vector<int> l_offset(tf_value_l.size(), 0);
            std::vector<udPT> l_result;
            std::vector<uint32_t> canopy_members;
            // merge and compare
            if (tf_value_l.size() >= 2) {
                std::vector<std::vector<udPT> > tmp_value(tf_value_l.size() - 1);
                addMerge(tf_value_l[0], tf_value_l[1], tmp_value[0]);
                for (int j = 2; j < tf_value_l.size(); j++) {
                    addMerge(tf_value_l[j], tmp_value[j - 2], tmp_value[j - 1]);
                }
                for (int j = 0; j < tmp_value[tf_value_l.size() - 2].size(); j++) {
                    if (tmp_value[tf_value_l.size() - 2][j].second > threshold) {
                        canopy_members.push_back(j);
                        if (canopy[tmp_value[tf_value_l.size() - 2][j].first] == -1)
                            canopy[tmp_value[tf_value_l.size() - 2][j].first] = doc_id;
                    }
                }
            } else {
                for (int j = 0; j < tf_value_l[0].size(); j++) {
                    if (tf_value_l[0][j].second > threshold) { // output value more than threshold
                        canopy_members.push_back(j);
                        if (canopy[tf_value_l[0][j].first] == -1) canopy[tf_value_l[0][j].first] = doc_id;
                    }
                }
            }
            // if (LOG > 1)
            //    std::cout << "l_result size=" << l_result.size() << ", canopy_members size=" << canopy_members.size()
            //              << std::endl;
            if (canopy_members.empty())
                indexId[i] = -1;
            else
                indexId[i] = doc_id;

        } else {
            indexId[i] = canopy[doc_id];
        }
        // if (LOG > 1) std::cout << "indexId[" << i << "]=" << indexId[i] << std::endl;
    }
}

} // internal
} // dup_match
