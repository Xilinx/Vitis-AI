/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DSPLIB_TEST_STIM_HPP_
#define _DSPLIB_TEST_STIM_HPP_

/*
Testbench stimulus class.
This file contains definition of the class used to generate stimulus data
for  reference model and UUT.
*/

#include <adf.h>
#include <stdio.h> /* printf, scanf, puts, NULL */
#include <string>
//#include <stdlib.h>  //clashes with utils.h definition of srand
#include "utils.hpp"

#define FILENAME_MAX_LEN 255

namespace xf {
namespace dsp {
namespace aie {
namespace testcase {
// Template class, one class fits all cases. Generate data, based on template parameters.
template <typename TT_DATA,  // type of data output
          size_t TP_SAMPLES, // number of samples to generate
          size_t TP_OFFSET   // number of empty samples to generate
          >
class test_stim {
   public:
    // Constructor
    test_stim(std::string filename = std::string("data/input.txt")) {
        std::strcpy(m_fileName, filename.c_str());
        m_fileReady = true;
    }

    // Main operation function
    int gen(T_STIM_MODE mode, TT_DATA* d_out, bool symmetry = false) {
        int ret = RETURN_SUCCESS;
        size_t samples = TP_SAMPLES;
        if (symmetry) {
            samples = (TP_SAMPLES + 1) / 2;
        }
        // Based on mode, generate an array of TT_DATA elements
        switch (mode) {
            case STIM_GEN_SINE:
                sineData(samples, TP_OFFSET, d_out);
                break;
            case STIM_GEN_SYMINCONES:
                symIncOnesData(samples, TP_OFFSET, d_out);
                break;
            case STIM_GEN_INCONES:
                inconesData(samples, TP_OFFSET, d_out);
                break;
            case STIM_GEN_ALLONES:
                allonesData(samples, TP_OFFSET, d_out);
                break;
            case STIM_GEN_IMPULSE:
                impulseData(samples, TP_OFFSET, d_out);
                break;
            case STIM_GEN_PRBS:
                printf("Unsupported mode \n");
                ret = RETURN_ERROR;
                break;
            case STIM_GEN_FROM_FILE:
                if (m_fileReady) {
                    ret = read_array(samples, m_fileName, TP_OFFSET, d_out);
                } else {
                    printf("read file not ready\n");
                    ret = RETURN_ERROR;
                }
                break;
            case STIM_GEN_RANDOM:
                srand(m_seed);
                randData(samples, TP_OFFSET, d_out);
                break;
            default:
                printf("Unknown stimulus mode\n");
                ret = RETURN_ERROR;
                break;
        }
        // mirror
        /*
        if (symmetry) {
            for (int i=1;i<(TP_SAMPLES+1)/2;i++) {
                d_out[TP_SAMPLES-i] = d_out[i-1];
            }
        }
        */
        return ret;
    }

    // Clear file
    void prepFile(char* filename) {
        strcpy(m_fileName, filename);
        m_fileReady = true;
    }
    // Set seed
    inline void prepSeed(int seed) { m_seed = seed; }

   private:
    int m_seed;
    bool m_fileReady = false;
    char m_fileName[FILENAME_MAX_LEN]; // max filename size
};

template <typename T_D>
inline T_D addError(T_D inVal) {
    T_D retVal;
    // Increase input by arbitrary amount
    retVal = inVal * 6 / 11;
    return retVal;
}

template <>
inline cint16 addError<cint16>(cint16 inVal) {
    cint16 retVal;
    // Increase input by arbitrary amount
    retVal.real = inVal.real * 6 / 11;
    retVal.imag = inVal.imag * 13 / 27;
    return retVal;
}

template <>
inline cint32 addError<cint32>(cint32 inVal) {
    cint32 retVal;
    // Increase input by arbitrary amount
    retVal.real = inVal.real * 6 / 11;
    retVal.imag = inVal.imag * 13 / 27;
    return retVal;
}

template <>
inline cfloat addError<cfloat>(cfloat inVal) {
    cfloat retVal;
    // Increase input by arbitrary amount
    retVal.real = inVal.real * 6 / 11;
    retVal.imag = inVal.imag * 13 / 27;
    return retVal;
}
}
}
}
}
#endif // _DSPLIB_TEST_STIM_HPP_
