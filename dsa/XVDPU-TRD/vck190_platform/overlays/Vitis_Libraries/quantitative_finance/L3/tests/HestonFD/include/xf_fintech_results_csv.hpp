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

#ifndef _XF_FINTECH_RESULTS_CSV_HPP_
#define _XF_FINTECH_RESULTS_CSV_HPP_
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cassert>
#include "xf_fintech_results_csv_rec.hpp"

using namespace std;

class CSVResults {
    std::vector<GoldenResultsCSVRecord> csvRecGoldenVec; // holds all csv record

   public:
    CSVResults(){};

    void init(std::istream& ss) {
        do // read all rows of file
        {
            GoldenResultsCSVRecord GoldenResults;

            std::string s;
            (void)std::getline(ss, s);

            // Ignore lines starting with #
            if (s[0] == '#') {
                continue;
            }

            // std::cout << "line:" << s << std::endl;

            if (0 == s.size()) break;
            assert(s.size());
            extractDoubles(s, GoldenResults.csvTableEntry);
            csvRecGoldenVec.push_back(GoldenResults); // capture
            if (ss.eof()) break;

        } while (1);
    }

    std::vector<double> data(int i) { return (csvRecGoldenVec[i].showCSVTableEntry()); }

    int showNumberOfRows() {
        size_t size = csvRecGoldenVec.size();
        return std::int32_t(size);
    }

    int showNumberOfColumns() {
        size_t size = csvRecGoldenVec[0].csvTableEntry.size();
        return std::int32_t(size);
    }

    void displayTableEntry(int i) {
        std::cout << "Table Entry: " << i << std::endl;
        for (std::vector<double>::iterator it = csvRecGoldenVec[i].csvTableEntry.begin();
             it != csvRecGoldenVec[i].csvTableEntry.end(); ++it)
            std::cout << ' ' << *it << '\n';
        std::cout << std::endl;
    }

   private:
    void extractDoubles(std::string& s,
                        std::vector<double>& csvTableEntry) //, int howmany)
    {
        std::istringstream ss(s);

        double t = 0;

        while (ss >> t) {
            ss.ignore(1);      // skip ','
            assert(!ss.bad()); // confirm ok
            csvTableEntry.push_back(t);
        }

        s.erase(0, std::string::npos);
    }

}; // class CSVResults

#endif
