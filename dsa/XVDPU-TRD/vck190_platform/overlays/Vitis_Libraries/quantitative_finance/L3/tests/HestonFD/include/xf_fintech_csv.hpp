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

#ifndef _XF_FINTECH_CSV_HPP_
#define _XF_FINTECH_CSV_HPP_

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xf_fintech_csv_rec.hpp"

using namespace std;

class CSV {
    std::vector<int> numberMismatches;
    std::vector<CSVRecord> csvRecVec; // holds all CSV record

   public:
    CSV(){};

    void init(std::istream& ss) {
        do // read all rows of file
        {
            CSVRecord csvRec;

            std::string s;
            (void)std::getline(ss, s);

            // Ignore lines starting with #
            if (s[0] == '#') {
                continue;
            }

            if (0 == s.size()) break;
            assert(s.size());
            extractText(s, csvRec.testcase); // Testcase
            assert(s.size());
            extractText(s, csvRec.scheme); // Scheme
            assert(s.size());
            extractDoubles(s, csvRec.testParameters, 15); // handle 15 fields
            csvRecVec.push_back(csvRec);                  // capture

            numberMismatches.push_back(0);
            if (ss.eof()) break;

        } while (1);
    }

    std::string showTestCase(int i) {
        string testcase = csvRecVec[i].showTestCase();
        return std::string(testcase);
    }

    std::string showTestScheme(int i) {
        string testscheme = csvRecVec[i].showTestScheme();
        return std::string(testscheme);
    }

    std::vector<double> showTestParameters(int i) { return (csvRecVec[i].showTestParameters()); }

    int showNumberOfTests() {
        size_t size = csvRecVec.size();
        return std::int32_t(size);
    }

    void setNumberMismatches(int testcase, int num) { numberMismatches[testcase] = num; }

    int getNumberMismatches(int testcase) { return (numberMismatches[testcase]); }

   private:
    void extractText(std::string& s, std::string& s2) {
        size_t indx1 = 0;
        assert(indx1 != std::string::npos);

        size_t indx2 = s.find(',', indx1 + 1);
        assert(indx2 != std::string::npos);

        size_t rng1 = indx2 - indx1 + 1;

        s2 = s.substr(indx1, rng1 - 1);

        s.erase(indx1, rng1);
    }

    void extractDoubles(std::string& s, std::vector<double>& testParameters, int howmany) {
        std::stringstream ss(s);

        double t = 0;
        for (int i = 0; i < howmany; ++i) {
            ss >> t;
            ss.ignore(1);      // skip ','
            assert(!ss.bad()); // confirm ok
            testParameters.push_back(t);
        }
        s.erase(0, std::string::npos);
    }

}; // class CSV

#endif