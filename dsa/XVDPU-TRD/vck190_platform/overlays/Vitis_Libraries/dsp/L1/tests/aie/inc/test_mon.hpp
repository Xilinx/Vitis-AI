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
#ifndef _DSPLIB_TEST_MON_HPP_
#define _DSPLIB_TEST_MON_HPP_

/*
Testbench monitor class.
This file contains definition of the class used to compare the output results
from reference model and UUT.
*/
#include <adf.h>
#include <string.h>
#include "utils.hpp"

#define MAX_FILENAME_SIZE 255

namespace xf {
namespace dsp {
namespace aie {
namespace testcase {
// Monitor class - takes 2 inputs and displays the comparison results.
template <typename TT_DATA, // type of data input and output
          size_t TP_SAMPLES // number of samples
          >
class test_mon {
   public:
    // Constructor
    test_mon();

    // Main operation function
    // This is the default variant of the templatized filter function. For all legal combinations, the specialised
    // variants will be called, so this
    // variant need only call out that the default is invalid
    void compare(TT_DATA* d_in, TT_DATA* r_in) { printf("Class type not supported"); }
};

// Now use partial template specialization to describe the main function for particular data types
// template class for int16 data
template <size_t TP_SAMPLES>
class test_mon<int16_t, TP_SAMPLES> {
   public:
    // arguments to the operator function are run-time variables
    unsigned int compare(const int16_t* d_in, const int16_t* r_in, const int debug_lvl

                         ) {
        unsigned int errors = 0;
        // Compare each input sample with reference output
        for (unsigned int i = 0; i < TP_SAMPLES; i++) {
            if (!isSame(d_in[i], r_in[i])) {
                // error
                errors++;
                printf("ERROR=%4d, SAMPLE=%d, GOT=(%d), REF=(%d)\n", errors, i, d_in[i], r_in[i]);
            } else {
                // no error
                if (debug_lvl >= DEBUG_LVL_NOTE) {
                    printf("PASS,       SAMPLE=%d, GOT=(%d), REF=(%d)\n", i, d_in[i], r_in[i]);
                }
            }
        }

        // Write output samples to an output file
        if (m_fileReady) {
            FILE* pFile;
            int real = 0;

            pFile = fopen(m_fileName, "a");
            for (int i = 0; i < TP_SAMPLES; i++) {
                fprintf(pFile, "%d, %d\n", d_in[i], r_in[i]);
            }
            fclose(pFile);
            pFile = NULL;
        }
        return errors;
    }
    // Clear file
    void prepFile(char* filename) {
        FILE* pFile;

        pFile = fopen(filename, "w");
        if (pFile != NULL) {
            strcpy(m_fileName, filename);
            m_fileReady = true;
            fclose(pFile);
            pFile = NULL;
        }
    }

   private:
    bool m_fileReady = false;
    char m_fileName[MAX_FILENAME_SIZE]; // max filename size
};

// template class for cint16 data
template <size_t TP_SAMPLES>
class test_mon<cint16_t, TP_SAMPLES> {
   public:
    unsigned int compare(const cint16_t* d_in, const cint16_t* r_in, const int debug_lvl) {
        unsigned int errors = 0;
        // Compare each input sample with reference output
        for (unsigned int i = 0; i < TP_SAMPLES; i++) {
            if (!isSame(d_in[i], r_in[i])) {
                // error
                errors++;
                printf("ERROR=%d,SAMPLE=%d, GOT=(%d,%d), REF=(%d,%d)\n", errors, i, d_in[i].real, d_in[i].imag,
                       r_in[i].real, r_in[i].imag);
            } else {
                // no error
                if (debug_lvl >= DEBUG_LVL_NOTE) {
                    printf("PASS,SAMPLE=%d, GOT=(%d,%d), REF=(%d,%d)\n", i, d_in[i].real, d_in[i].imag, r_in[i].real,
                           r_in[i].imag);
                }
            }
        }
        // Write output samples to an output file
        if (m_fileReady) {
            FILE* pFile;
            int real, imag;
            pFile = fopen(m_fileName, "a");
            for (int i = 0; i < (TP_SAMPLES); i++) {
                fprintf(pFile, "%d %d, %d %d \n", d_in[i].real, d_in[i].imag, r_in[i].real, r_in[i].imag);
            }
            fclose(pFile);
            pFile = NULL;
        }
        return errors;
    }
    // Clear file
    void prepFile(char* filename) {
        FILE* pFile;

        pFile = fopen(filename, "w");
        if (pFile != NULL) {
            strcpy(m_fileName, filename);
            m_fileReady = true;
            fclose(pFile);
            pFile = NULL;
        }
    }

   private:
    bool m_fileReady = false;
    char m_fileName[MAX_FILENAME_SIZE]; // max filename size
};

// Now use partial template specialization to describe the main function for particular data types
// template class for int32 data
template <size_t TP_SAMPLES>
class test_mon<int32_t, TP_SAMPLES> {
   public:
    // arguments to the operator function are run-time variables
    unsigned int compare(const int32_t* d_in, const int32_t* r_in, const int debug_lvl) {
        unsigned int errors = 0;
        // Compare each input sample with reference output
        for (unsigned int i = 0; i < TP_SAMPLES; i++) {
            if (!isSame(d_in[i], r_in[i])) {
                // error
                errors++;
                printf("ERROR=%d, SAMPLE=%d, GOT=(%d), REF=(%d)\n", errors, i, d_in[i], r_in[i]);
            } else {
                // no error
                if (debug_lvl >= DEBUG_LVL_NOTE) {
                    printf("PASS, SAMPLE=%d, GOT=(%d), REF=(%d)\n", i, d_in[i], r_in[i]);
                }
            }
        }

        // Write output samples to an output file
        if (m_fileReady) {
            FILE* pFile;
            int real = 0;

            pFile = fopen(m_fileName, "a");
            for (int i = 0; i < TP_SAMPLES; i++) {
                fprintf(pFile, "%d, %d\n", d_in[i], r_in[i]);
            }
            fclose(pFile);
            pFile = NULL;
        }
        return errors;
    }
    // Clear file
    void prepFile(char* filename) {
        FILE* pFile;

        pFile = fopen(filename, "w");
        if (pFile != NULL) {
            strcpy(m_fileName, filename);
            m_fileReady = true;
            fclose(pFile);
            pFile = NULL;
        }
    }

   private:
    bool m_fileReady = false;
    char m_fileName[MAX_FILENAME_SIZE]; // max filename size
};

// template class for cint32 data
template <size_t TP_SAMPLES>
class test_mon<cint32_t, TP_SAMPLES> {
   public:
    unsigned int compare(const cint32_t* d_in, const cint32_t* r_in, const int debug_lvl) {
        unsigned int errors = 0;
        // Compare each input sample with reference output
        for (unsigned int i = 0; i < TP_SAMPLES; i++) {
            if (!isSame(d_in[i], r_in[i])) {
                // error
                errors++;
                printf("ERROR=%d,SAMPLE=%d, GOT=(%d,%d), REF=(%d,%d)\n", errors, i, (int)d_in[i].real,
                       (int)d_in[i].imag, (int)r_in[i].real, (int)r_in[i].imag);
            } else {
                // no error
                if (debug_lvl >= DEBUG_LVL_NOTE) {
                    printf("PASS,SAMPLE=%d, GOT=(%d,%d), REF=(%d,%d)\n", i, d_in[i].real, d_in[i].imag, r_in[i].real,
                           r_in[i].imag);
                }
            }
        }
        // Write output samples to an output file
        if (m_fileReady) {
            FILE* pFile;
            int real, imag;
            pFile = fopen(m_fileName, "a");
            for (int i = 0; i < (TP_SAMPLES); i++) {
                fprintf(pFile, "%d %d, %d %d \n", d_in[i].real, d_in[i].imag, r_in[i].real, r_in[i].imag);
            }
            fclose(pFile);
            pFile = NULL;
        }
        return errors;
    }
    // Clear file
    void prepFile(char* filename) {
        FILE* pFile;

        pFile = fopen(filename, "w");
        if (pFile != NULL) {
            strcpy(m_fileName, filename);
            m_fileReady = true;
            fclose(pFile);
            pFile = NULL;
        }
    }

   private:
    bool m_fileReady = false;
    char m_fileName[MAX_FILENAME_SIZE]; // max filename size
};

// template class for float data
template <size_t TP_SAMPLES>
class test_mon<float, TP_SAMPLES> {
   public:
    // arguments to the operator function are run-time variables
    unsigned int compare(const float* d_in, const float* r_in, const int debug_lvl) {
        unsigned int errors = 0;
        // Compare each input sample with reference output
        for (unsigned int i = 0; i < TP_SAMPLES; i++) {
            if (!isSame(d_in[i], r_in[i])) {
                // error
                errors++;
                printf("ERROR=%d, SAMPLE=%d, GOT=(%f), REF=(%f)\n", errors, i, d_in[i], r_in[i]);
            } else {
                // no error
                if (debug_lvl >= DEBUG_LVL_NOTE) {
                    printf("PASS, SAMPLE=%d, GOT=(%f), REF=(%f)\n", i, d_in[i], r_in[i]);
                }
            }
        }

        // Write output samples to an output file
        if (m_fileReady) {
            FILE* pFile;
            int real = 0;

            pFile = fopen(m_fileName, "a");
            for (int i = 0; i < TP_SAMPLES; i++) {
                fprintf(pFile, "%f, %f\n", d_in[i], r_in[i]);
            }
            fclose(pFile);
            pFile = NULL;
        }
        return errors;
    }
    // Clear file
    void prepFile(char* filename) {
        FILE* pFile;

        pFile = fopen(filename, "w");
        if (pFile != NULL) {
            strcpy(m_fileName, filename);
            m_fileReady = true;
            fclose(pFile);
            pFile = NULL;
        }
    }

   private:
    bool m_fileReady = false;
    char m_fileName[MAX_FILENAME_SIZE]; // max filename size
};

// template class for cfloat data
template <size_t TP_SAMPLES>
class test_mon<cfloat, TP_SAMPLES> {
   public:
    unsigned int compare(const cfloat* d_in, const cfloat* r_in, const int debug_lvl) {
        unsigned int errors = 0;
        // Compare each input sample with reference output
        for (unsigned int i = 0; i < TP_SAMPLES; i++) {
            if (!isSame(d_in[i], r_in[i])) {
                // error
                errors++;
                printf("ERROR=%d,SAMPLE=%d, GOT=(%f,%f), REF=(%f,%f)\n", errors, i, (float)d_in[i].real,
                       (float)d_in[i].imag, (float)r_in[i].real, (float)r_in[i].imag);
            } else {
                // no error
                if (debug_lvl >= DEBUG_LVL_NOTE) {
                    printf("PASS,SAMPLE=%d, GOT=(%f,%f), REF=(%f,%f)\n", i, d_in[i].real, d_in[i].imag, r_in[i].real,
                           r_in[i].imag);
                }
            }
        }
        // Write output samples to an output file
        if (m_fileReady) {
            FILE* pFile;
            int real, imag;
            pFile = fopen(m_fileName, "a");
            for (int i = 0; i < (TP_SAMPLES); i++) {
                fprintf(pFile, "%f %f, %f %f \n", d_in[i].real, d_in[i].imag, r_in[i].real, r_in[i].imag);
            }
            fclose(pFile);
            pFile = NULL;
        }
        return errors;
    }
    // Clear file
    void prepFile(char* filename) {
        FILE* pFile;

        pFile = fopen(filename, "w");
        if (pFile != NULL) {
            strcpy(m_fileName, filename);
            m_fileReady = true;
            fclose(pFile);
            pFile = NULL;
        }
    }

   private:
    bool m_fileReady = false;
    char m_fileName[MAX_FILENAME_SIZE]; // max filename size
};
}
}
}
}
#endif // _DSPLIB_TEST_MON_HPP_
