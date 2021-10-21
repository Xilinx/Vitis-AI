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
#ifndef FD_UTIL_H_
#define FD_UTIL_H_
//#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
//#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include <cmath>
#include <string>
#include <vector>
//#include <boost/iostreams/filtering_stream.hpp>
//#include <boost/iostreams/filtering_streambuf.hpp>
#include "xcl2.hpp"

using namespace std;
namespace fd {

// class XTimer
//{
//  public:
//    XTimer() : beg_(clock_::now()) {}
//    void reset() { beg_ = clock_::now(); }
//    double elapsed() const {
//      return chrono::duration_cast<second_>
//        (clock_::now() - beg_).count(); }
//
//  private:
//    typedef chrono::high_resolution_clock clock_;
//    typedef chrono::duration<double, ratio<1> > second_;
//    chrono::time_point<clock_> beg_;
//};
//
template <typename DT, unsigned int t_MemWidth, unsigned int Dim2Size1 = 3, unsigned int Dim2Size2 = 5>
class FdUtil {
   public:
    /// @brief Read sparse matrix from .csv file (in co-ordinate format)
    int ReadSparse(std::string filename,
                   std::vector<DT, aligned_allocator<DT> >& A,
                   std::vector<unsigned int, aligned_allocator<unsigned int> >& Ar,
                   std::vector<unsigned int, aligned_allocator<unsigned int> >& Ac,
                   unsigned int& Annz,
                   unsigned int& padA,
                   unsigned int& padAr,
                   const unsigned int size) {
        std::ifstream file(filename);

        std::vector<std::string> v;
        std::string line;

        if (file.good()) {
            std::cout << "Opened " << filename << " OK" << std::endl;
            getline(file, line);
            Annz = std::stoi(line);
            std::cout << "  - File has " << Annz << " non-zeroes" << std::endl;

            unsigned int i = 0;
            while (file.good()) {
                getline(file, line);
                boost::split(v, line, [](char c) { return c == ' '; });
                if (v.size() == 3) {
                    Ar[i] = std::stoi(v[0]);
                    Ac[i] = std::stoi(v[1]);
                    A[i] = (DT)std::stod(v[2]);
                    i++;
                }
                if (i > size) {
                    std::cout << "Warning! File has more lines than can be held in "
                                 "sparse matrix..."
                              << std::endl;
                    break;
                }
            }

            // Pad the data to a multiple of the DDR width (depends on data size)
            padA = i;
            while ((padA % t_MemWidth) != 0) {
                A[padA] = 0;
                padA++;
                if (padA > size) {
                    std::cout << "Warning! Need more space for storing sparse matrix data..." << std::endl;
                    break;
                }
            }
            if (padA > 0) {
                std::cout << "  Padded data to " << padA << std::endl;
            }

            // Pad the row/column indices to a multiple of the DDR width (16 indices
            // per 512-bit DDR word)
            padAr = i;
            while ((padAr % 16) != 0) {
                Ar[padAr] = 0;
                Ac[padAr] = 0;
                padAr++;
                if (padAr > size) {
                    std::cout << "Warning! Need more space for storing sparse matrix indices..." << std::endl;
                    break;
                }
            }
            if (padAr > 0) {
                std::cout << "  Padded row/column indices to " << padAr << std::endl;
            }

        } else {
            std::cout << "Couldn't open " << filename << std::endl;
            return 1;
        }
        return 0;
    }

    /// @brief Read diagonal matrix from .csv (three columns)
    int ReadDiag3(std::string filename, std::vector<DT, aligned_allocator<DT> >& A, const unsigned int size) {
        std::ifstream file(filename);

        std::vector<std::string> v;
        std::string line;

        if (file.good()) {
            std::cout << "Opened " << filename << " OK" << std::endl;

            unsigned int i = 0;
            while (file.good()) {
                getline(file, line);
                boost::split(v, line, [](char c) { return c == ','; });
                if (v.size() == Dim2Size1) {
                    // Write to vector such that given diagonal is adjacent in memory
                    A[size * 0 + i] = (DT)std::stod(v[0]);
                    A[size * 1 + i] = (DT)std::stod(v[1]);
                    A[size * 2 + i] = (DT)std::stod(v[2]);
                    i++;
                }
                if (i > size) {
                    std::cout << "Warning! File has more than expected " << size << " lines..." << std::endl;
                    break;
                }
            }
        } else {
            std::cout << "Couldn't open " << filename << std::endl;
            return 1;
        }
        return 0;
    }

    /// @brief Read diagonal matrix from .csv (five columns)
    int ReadDiag5(std::string filename, std::vector<DT, aligned_allocator<DT> >& A, const unsigned int size) {
        std::ifstream file(filename);

        std::vector<std::string> v;
        std::string line;

        if (file.good()) {
            std::cout << "Opened " << filename << " OK" << std::endl;

            unsigned int i = 0;
            while (file.good()) {
                getline(file, line);
                boost::split(v, line, [](char c) { return c == ','; });
                if (v.size() == Dim2Size2) {
                    A[size * 0 + i] = (DT)std::stod(v[0]);
                    A[size * 1 + i] = (DT)std::stod(v[1]);
                    A[size * 2 + i] = (DT)std::stod(v[2]);
                    A[size * 3 + i] = (DT)std::stod(v[3]);
                    A[size * 4 + i] = (DT)std::stod(v[4]);
                    i++;
                }
                if (i > size) {
                    std::cout << "Warning! File has more than expected " << size << " lines..." << std::endl;
                    break;
                }
            }
        } else {
            std::cout << "Couldn't open " << filename << std::endl;
            return 1;
        }
        return 0;
    }

    /// @brief Read vector from .csv
    int ReadVector(const std::string filename, std::vector<DT, aligned_allocator<DT> >& A, const unsigned int size) {
        std::ifstream file(filename);

        std::vector<std::string> v;
        std::string line;

        if (file.good()) {
            std::cout << "Opened " << filename << " OK" << std::endl;

            unsigned int i = 0;
            while (file.good()) {
                getline(file, line);
                // std::cout << line << std::endl;
                if (line.length()) {
                    A[i] = (DT)std::stod(line);
                    i++;
                }
                if (i > size) {
                    std::cout << "Warning! File has more than expected " << size << " lines..." << std::endl;
                    break;
                }
            }
        } else {
            std::cout << "Couldn't open " << filename << std::endl;
            return 1;
        }
        return 0;
    }

    /// @brief Read params from .csv
    // std::unordered_map<std::string, DT> readParam(const std::string filename)
    //{
    //  std::unordered_map<std::string, DT> l_res;
    //  std::ifstream l_fs(filename.c_str(),
    //  std::ios_base::in|std::ios_base::binary);
    //  boost::iostreams::filtering_istream l_bs;
    //  l_bs.push(l_fs);
    //  while (l_bs.good()) {
    //    std::string l_paramName;
    //    DT l_paramVal;
    //    l_bs >> l_paramName >> l_paramVal;
    //    l_res[l_paramName] = l_paramVal;
    //  }
    //  boost::iostreams::close(l_bs);
    //  return l_res;
    //}

    /// @brief Pretty print a vector
    void PrintVector(const std::string str,
                     const std::vector<DT, aligned_allocator<DT> >& vIn,
                     const unsigned int size) {
        std::cout << str << std::endl << "[";
        int d = size / 4;
        int r = size - 4 * d;

        std::cout << std::setprecision(9) << std::scientific;

        for (int i = 0; i < d; i++) {
            if (i) std::cout << " ";
            std::cout << vIn[4 * i + 0] << ", " << vIn[4 * i + 1] << ", " << vIn[4 * i + 2] << ", " << vIn[4 * i + 3]
                      << ", ";
            if (i < (d - 1)) std::cout << std::endl;
        }
        if (r) {
            std::cout << std::endl << " ";
            for (int i = 0; i < r; i++) {
                std::cout << vIn[4 * d + i];
                if (i < (r - 1)) std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl << std::endl;
    }

    /// @brief Compare a vector to reference
    DT CompareReference(std::vector<DT, aligned_allocator<DT> >& vIn,
                        std::vector<DT, aligned_allocator<DT> >& vRef,
                        const int size) {
        DT diff[size];
        DT maxv = 0.0;
        int maxi = 0;
        for (int i = 0; i < size; ++i) {
            diff[i] = vIn[i] - vRef[i];
            if (std::abs(diff[i]) > std::abs(maxv)) {
                maxv = diff[i];
                maxi = i;
            }
        }
        std::cout << "Maximum difference is " << maxv << ", found at array index " << maxi << std::endl
                  << std::endl
                  << std::endl;

        return maxv;
    }

    /// @brief Round to next multiple
    int RoundToNextMultiple(int numToRound, int multiple) {
        if (multiple == 0) return numToRound;

        int remainder = abs(numToRound) % multiple;
        if (remainder == 0) return numToRound;

        if (numToRound < 0)
            return -(abs(numToRound) - remainder);
        else
            return numToRound + multiple - remainder;
    }
};
}
#endif
