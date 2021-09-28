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
/*
Widget API Cast reference model
*/

#include "widget_api_cast_ref.hpp"
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace widget {
namespace api_cast {

// Widget API Cast - default/base 'specialization'
template <typename TT_DATA,
          unsigned int TP_IN_API,
          unsigned int TP_OUT_API,
          unsigned int TP_NUM_INPUTS,
          unsigned int TP_WINDOW_VSIZE,
          unsigned int TP_NUM_OUTPUT_CLONES>
void widget_api_cast_ref<TT_DATA, TP_IN_API, TP_OUT_API, TP_NUM_INPUTS, TP_WINDOW_VSIZE, TP_NUM_OUTPUT_CLONES>::
    transferData(input_window<TT_DATA>* inWindow0, output_window<TT_DATA>* outWindow0) {
    TT_DATA d_in;

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
    }
};

// Widget API Cast - dual output 'specialization'
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2>::transferData(
    input_window<TT_DATA>* inWindow0, output_window<TT_DATA>* outWindow0, output_window<TT_DATA>* outWindow1) {
    TT_DATA d_in;

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
        window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
    }
};

// Widget API Cast - triple output 'specialization'
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kWindowAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3>::transferData(
    input_window<TT_DATA>* inWindow0,
    output_window<TT_DATA>* outWindow0,
    output_window<TT_DATA>* outWindow1,
    output_window<TT_DATA>* outWindow2) {
    TT_DATA d_in;

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
        window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
        window_writeincr((output_window<TT_DATA>*)outWindow2, d_in);
    }
};

// Widget API Cast - 1 stream input 1 window output
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 1>::transferData(
    input_stream<TT_DATA>* inStream0, output_window<TT_DATA>* outWindow0) {
    TT_DATA d_in;

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(outWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = readincr(inStream0); // read input data
        window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
    }
};

// Widget API Cast - 1 stream input 2 window output
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 2>::transferData(
    input_stream<TT_DATA>* inStream0, output_window<TT_DATA>* outWindow0, output_window<TT_DATA>* outWindow1) {
    TT_DATA d_in;

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(outWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = readincr(inStream0); // read input data
        window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
        window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
    }
};

// Widget API Cast - 1 stream input 3 window output
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 3>::transferData(
    input_stream<TT_DATA>* inStream0,
    output_window<TT_DATA>* outWindow0,
    output_window<TT_DATA>* outWindow1,
    output_window<TT_DATA>* outWindow2) {
    TT_DATA d_in;

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(outWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = readincr(inStream0); // read input data
        window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
        window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
        window_writeincr((output_window<TT_DATA>*)outWindow2, d_in);
    }
};

// Widget API Cast - 1 stream input 4 window output
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kStreamAPI, kWindowAPI, 1, TP_WINDOW_VSIZE, 4>::transferData(
    input_stream<TT_DATA>* inStream0,
    output_window<TT_DATA>* outWindow0,
    output_window<TT_DATA>* outWindow1,
    output_window<TT_DATA>* outWindow2,
    output_window<TT_DATA>* outWindow3) {
    TT_DATA d_in;

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(outWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = readincr(inStream0); // read input data
        window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
        window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
        window_writeincr((output_window<TT_DATA>*)outWindow2, d_in);
        window_writeincr((output_window<TT_DATA>*)outWindow3, d_in);
    }
};

// Dual stream in
// Widget API Cast - 2 stream input 1 window output
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 1>::transferData(
    input_stream<TT_DATA>* inStream0, input_stream<TT_DATA>* inStream1, output_window<TT_DATA>* outWindow0) {
    TT_DATA d_in;
    constexpr unsigned int kWriteSize = 256 / 8; // in bytes
    constexpr unsigned int kStreamReadSize = 128 / 8;
    constexpr unsigned int kNumStreams = 2;
    constexpr unsigned int kSampleSize = sizeof(TT_DATA);
    constexpr unsigned int Lsize = TP_WINDOW_VSIZE * kSampleSize / (kWriteSize);
#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(outWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < Lsize; i++) {
        for (int k = 0; k < kStreamReadSize / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream0); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
        }
        for (int k = 0; k < 128 / 8 / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream1); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
        }
    }
    if (TP_WINDOW_VSIZE * kSampleSize / (kWriteSize / 2) % 2 == 1) { // odd number of chunks
        for (int k = 0; k < kStreamReadSize / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream0); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
        }
    }
};

// Widget API Cast - 2 stream input 2 window output
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 2>::transferData(
    input_stream<TT_DATA>* inStream0,
    input_stream<TT_DATA>* inStream1,
    output_window<TT_DATA>* outWindow0,
    output_window<TT_DATA>* outWindow1) {
    TT_DATA d_in;
    constexpr unsigned int kWriteSize = 256 / 8; // in bytes
    constexpr unsigned int kStreamReadSize = 128 / 8;
    constexpr unsigned int kNumStreams = 2;
    constexpr unsigned int kSampleSize = sizeof(TT_DATA);
    constexpr unsigned int Lsize = TP_WINDOW_VSIZE * kSampleSize / (kWriteSize);

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(outWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < Lsize; i++) {
        for (int k = 0; k < kStreamReadSize / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream0); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
        }
        for (int k = 0; k < 128 / 8 / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream1); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
        }
    }
    if (TP_WINDOW_VSIZE * kSampleSize / (kWriteSize / 2) % 2 == 1) { // odd number of chunks
        for (int k = 0; k < kStreamReadSize / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream0); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
        }
    }
};

// Widget API Cast - 2 stream input 3 window output
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 3>::transferData(
    input_stream<TT_DATA>* inStream0,
    input_stream<TT_DATA>* inStream1,
    output_window<TT_DATA>* outWindow0,
    output_window<TT_DATA>* outWindow1,
    output_window<TT_DATA>* outWindow2) {
    TT_DATA d_in;
    constexpr unsigned int kWriteSize = 256 / 8; // in bytes
    constexpr unsigned int kStreamReadSize = 128 / 8;
    constexpr unsigned int kNumStreams = 2;
    constexpr unsigned int kSampleSize = sizeof(TT_DATA);
    constexpr unsigned int Lsize = TP_WINDOW_VSIZE * kSampleSize / (kWriteSize);

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(outWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < Lsize; i++) {
        for (int k = 0; k < kStreamReadSize / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream0); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow2, d_in);
        }
        for (int k = 0; k < 128 / 8 / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream1); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow2, d_in);
        }
    }
    if (TP_WINDOW_VSIZE * kSampleSize / (kWriteSize / 2) % 2 == 1) { // odd number of chunks
        for (int k = 0; k < kStreamReadSize / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream0); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow2, d_in);
        }
    }
};

// Widget API Cast - 2 stream input 4 window output
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kStreamAPI, kWindowAPI, 2, TP_WINDOW_VSIZE, 4>::transferData(
    input_stream<TT_DATA>* inStream0,
    input_stream<TT_DATA>* inStream1,
    output_window<TT_DATA>* outWindow0,
    output_window<TT_DATA>* outWindow1,
    output_window<TT_DATA>* outWindow2,
    output_window<TT_DATA>* outWindow3) {
    TT_DATA d_in;
    constexpr unsigned int kWriteSize = 256 / 8; // in bytes
    constexpr unsigned int kStreamReadSize = 128 / 8;
    constexpr unsigned int kNumStreams = 2;
    constexpr unsigned int kSampleSize = sizeof(TT_DATA);
    constexpr unsigned int Lsize = TP_WINDOW_VSIZE * kSampleSize / (kWriteSize);

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(outWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < Lsize; i++) {
        for (int k = 0; k < kStreamReadSize / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream0); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow2, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow3, d_in);
        }
        for (int k = 0; k < 128 / 8 / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream1); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow2, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow3, d_in);
        }
    }
    if (TP_WINDOW_VSIZE * kSampleSize / (kWriteSize / 2) % 2 == 1) { // odd number of chunks
        for (int k = 0; k < kStreamReadSize / sizeof(TT_DATA); k++) {
            d_in = readincr(inStream0); // read input data
            window_writeincr((output_window<TT_DATA>*)outWindow0, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow1, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow2, d_in);
            window_writeincr((output_window<TT_DATA>*)outWindow3, d_in);
        }
    }
};

// Widget API Cast - window to stream, 1 to 1
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 1>::transferData(
    input_window<TT_DATA>* inWindow0, output_stream<TT_DATA>* outStream0) {
    TT_DATA d_in;

#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        writeincr(outStream0, d_in);
    }
};

// Widget API Cast - window to stream, 1 to 2
template <typename TT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_api_cast_ref<TT_DATA, kWindowAPI, kStreamAPI, 1, TP_WINDOW_VSIZE, 2>::transferData(
    input_window<TT_DATA>* inWindow0, output_stream<TT_DATA>* outStream0, output_stream<TT_DATA>* outStream1) {
    TT_DATA d_in;
    constexpr unsigned int kSamplesIn128b = 16 / sizeof(TT_DATA);
    constexpr unsigned int Lsize = TP_WINDOW_VSIZE / (2 * kSamplesIn128b);
#ifdef _DSPLIB_WIDGET_API_CAST_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_API_CAST_REF_DEBUG_

    for (unsigned int i = 0; i < Lsize; i++) {
        for (int i = 0; i < kSamplesIn128b; i++) {
            d_in = window_readincr(inWindow0); // read input data samplewise
            writeincr(outStream0, d_in);
        }
        for (int i = 0; i < kSamplesIn128b; i++) {
            d_in = window_readincr(inWindow0); // read input data
            writeincr(outStream1, d_in);
        }
    }
    if (TP_WINDOW_VSIZE / kSamplesIn128b % 2 == 1) {
        d_in = window_readincr(inWindow0); // read input data samplewise
        writeincr(outStream0, d_in);
    }
};
}
}
}
}
}
