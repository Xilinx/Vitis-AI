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
#ifndef _XL_INDEX_HPP_
#define _XL_INDEX_HPP_

#include <iostream>
#include <cstdint>
#include <sstream>
#include <fstream>
#include <algorithm>

#include <vector>
#include <map>
#include <string>
#include <cmath>

typedef std::map<std::string, uint32_t> suMT;
typedef std::map<uint32_t, uint32_t> uuMT;
typedef std::map<uint32_t, double> udMT;
typedef std::pair<uint32_t, double> udPT;

union DTConvert64 {
    uint64_t dt0;
    double dt1;
};

char charEncode(char in) {
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

// for print test
std::string printEncodeChar(std::string str) {
    std::string pStr = "";
    for (int i = 0; i < str.size(); i++) {
        char code = str[i];
        char ch;
        if (code <= 9)
            ch = code + 48;
        else if (code < 36)
            ch = code + 87;
        else
            ch = 32;
        pStr += ch;
    }
    return pStr;
}

void twoGram(std::string& inStr, std::vector<uint16_t>& terms) {
    for (int i = 0; i < inStr.size(); i += 2) {
        uint16_t term = charEncode(inStr[i]) + charEncode(inStr[i + 1]) * 64;
        terms.push_back(term);
        // std::cout << "twoGram: " << printEncodeChar(inStr.substr(i - 1, 2)) << std::endl;
    }
}

char charFilter(char in) {
    char out;
    if (in >= 48 && in <= 57) // 0~9
        out = in;
    else if (in >= 97 && in <= 122) // a~z
        out = in;
    else if (in >= 65 && in <= 90) // A~Z
        out = in + 32;
    else if (in == 32 || in == 10) // ' ' & '\n'
        out = 32;
    else
        out = 255;
    return out;
}

std::string twoGramStr(std::string& inStr) {
    uint8_t last_code = 32;
    std::string pStr = "";
    for (int i = 0; i < inStr.size(); i++) {
        uint8_t ch = inStr[i];
        uint8_t ch_code = charFilter(ch);
        if (ch_code == 255) continue;
        if (!(ch_code == last_code && ch_code == 32)) {
            pStr += ch_code;
        }
        last_code = ch_code;
    }
    if (last_code == 32) pStr.pop_back();
    std::vector<uint16_t> terms;
    for (int i = 1; i < pStr.size(); i++) {
        uint16_t term = pStr[i - 1] + pStr[i] * 256;
        terms.push_back(term);
        // std::cout << "twoGram: " << printEncodeChar(inStr.substr(i - 1, 2)) << std::endl;
    }
    std::sort(terms.begin(), terms.end());
    std::string outStr = "";
    for (int i = 0; i < terms.size(); i++) {
        // std::cout << "term[" << i << "]=" << terms[i] << std::endl;
        uint8_t ch = terms[i] % 256;
        outStr += ch;
        ch = terms[i] / 256;
        outStr += ch;
    }
    return outStr;
}

// calculate 2-gram index
int twoGramIndex(std::vector<std::string>& vecField, double* idfValue, uint64_t* tfAddr, uint64_t* tfValue) {
    std::cout << "======== 2 Gram Index =========\n";
    suMT map;
    std::vector<std::pair<std::string, uint32_t> > unique_fields;
    uint32_t id = 0;
    // deduplication field
    for (int i = 0; i < vecField.size(); i++) {
        std::string lineStr = vecField[i];
        std::string pStr = twoGramStr(lineStr);
        suMT::iterator iter = map.find(pStr);
        if (iter == map.end()) {
            unique_fields.push_back(std::pair<std::string, int>(pStr, id));
            map[pStr] = id++;
        }
    }
    std::cout << "index2Gram: map size=" << map.size() << std::endl;

    uuMT tmap;
    std::vector<std::vector<udPT> > word_info;
    uint32_t wid = 0, did = 0;
    // calculate idf for creating index
    for (int u = 0; u < unique_fields.size(); u++) {
        std::vector<uint16_t> terms;
        twoGram(unique_fields[u].first, terms); // split terms
        std::vector<uint32_t> wids;
        uuMT dict; // terms dictionary
        for (int i = 0; i < terms.size(); i++) {
            uint16_t term = terms[i];
            uuMT::iterator iter = tmap.find(term);
            if (iter == tmap.end()) {
                tmap[term] = wid;
                wids.push_back(wid);
                dict[wid] = 1;
                wid++;
            } else {
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
            word_info[udpt[i].first].push_back(udPT(did, temp));
            // std::cout << "w[" << i << "]=" << udpt[i].second / W << std::endl;
        }
        // std::cout << "W=" << W << std::endl;
        did++;
    }

    int N = map.size();
    // set mux index number
    int threshold = int(1000 > N * 0.05 ? 1000 : N * 0.05);
    std::cout << "threshold=" << threshold << std::endl;
    for (int i = 0; i < 4096; i++) {
        idfValue[i] = 0.0;
        tfAddr[i] = 0;
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
        idfValue[sn] = std::log(1.0 + (double)N / size);
        begin = end;
        for (int j = 0; j < size; j++) {
            tfValue[end++] = word_info[wid][j].first;
            DTConvert64 dtc;
            dtc.dt1 = word_info[wid][j].second;
            tfValue[end++] = dtc.dt0;
        }
        tfAddr[sn] = (begin >> 1) + (end << 31);
    }
    std::cout << "tfValue size is " << end / 2 << ", index count=" << map.size() << ", term count=" << tmap.size()
              << ", skip=" << skip << std::endl;
    return 0;
}

void readStringField(std::string& file,
                     int& docSize,
                     int& fieldSize,
                     uint8_t* fields,
                     uint32_t* offsets,
                     std::vector<std::string>& vec_field) {
    std::ifstream inFile(file, std::ios::in);
    std::string lineStr;
    while (getline(inFile, lineStr)) {
        vec_field.push_back(lineStr);
    }
    inFile.close();

    docSize = vec_field.size();
    int offset = 0;
    for (int i = 0; i < docSize; i++) {
        memcpy(fields + offset, vec_field[i].data(), vec_field[i].size());
        offset += vec_field[i].size();
        offsets[i] = offset;
    }
    fieldSize = offset;
    std::cout << "docSize=" << docSize << ", fieldSize=" << fieldSize << std::endl;
}

int checkResult(std::string& file, uint32_t* indexID) {
    int nerr = 0;
    std::ifstream inFile(file, std::ios::in);
    std::string lineStr;

    int i = 0;
    while (getline(inFile, lineStr)) {
        if (indexID[i++] != std::stoi(lineStr)) nerr++;
    }
    inFile.close();
    return nerr;
}

#endif
