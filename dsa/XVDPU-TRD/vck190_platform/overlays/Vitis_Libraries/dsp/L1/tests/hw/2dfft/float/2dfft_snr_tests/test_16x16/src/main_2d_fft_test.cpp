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
//================================== End Lic =================================================
#define TEST_2D_FFT_
#ifdef TEST_2D_FFT_
#include <math.h>
#include <string>
#include <assert.h>
#include <stdio.h>
#include "top_2d_fft_test.hpp"
#include "mVerificationUtlityFunctions.hpp"
#include "vitis_fft/hls_ssr_fft_2d_modeling_utilities.hpp"

int main(int argc, char** argv) {
    // 2d input matrix
    T_elemType l_inMat[k_fftKernelSize][k_fftKernelSize];
    T_outType l_outMat[k_fftKernelSize][k_fftKernelSize];
    T_outType l_data2d_golden[k_fftKernelSize][k_fftKernelSize];

    /////////////////////////// File Paths for stimulus files////////////////
    std::string root_path = "";
    std::string inputStimulusDataVerifFileName;
    std::stringstream strStream;
    ////////////////////////// Log Stream for writing output messages////////
    std::ofstream logStreamBB;
    std::string blockBoxFilename = root_path + "2d_fft_verification_log.log";
    logStreamBB.open(blockBoxFilename.c_str(), std::fstream::out);

    strStream << root_path << "fft2DStimulusIn_L" << k_fftKernelSize * k_fftKernelSize << ".verif";
    inputStimulusDataVerifFileName = strStream.str();
    std::cout << "The Stimulus file and constructed path: " << inputStimulusDataVerifFileName << "\n";

    std::stringstream strStream1;
    std::string goldenOutputDataVerifFileName;
    strStream1 << root_path << "fft2DGoldenOut_L" << k_fftKernelSize * k_fftKernelSize << ".verif";
    goldenOutputDataVerifFileName = strStream1.str();

    // Data Arrays to read stimulus data from file
    T_compleFloat* l_fileData = new T_compleFloat[k_fftKernelSize * k_fftKernelSize];

    // readComplexArrayFromFile<T_innerFloat>(logStreamBB, "din_file", inputStimulusDataVerifFileName, &l_inMat[0][0],
    // k_fftKernelSize*k_fftKernelSize);
    readComplexArrayFromFile<T_innerFloat>(logStreamBB, "din_file", inputStimulusDataVerifFileName, l_fileData,
                                           k_fftKernelSize * k_fftKernelSize);

    // init input matrix with stimuls, this loop will also take care of casting if types doesnt match
    int k = 0;
    for (int r = 0; r < k_fftKernelSize; ++r) {
        for (int c = 0; c < k_fftKernelSize; ++c) {
            l_inMat[r][c] = l_fileData[k];
            k++;
        }
    }
    delete l_fileData;
    // Wide Stream for reading and streaming a 2-d matrix
    MemWideIFStreamTypeIn l_matToStream("matrixToStreaming");
    MemWideIFStreamTypeOut fftOutputStream("fftOutputStream");
    // Pass same data stream multiple times to measure the II correctly
    for (int runs = 0; runs < 5; ++runs) {
        stream2DMatrix<k_fftKernelSize, k_fftKernelSize, k_memWidth, T_elemType, MemWideIFTypeIn>(l_inMat,
                                                                                                  l_matToStream);
        top_fft2d(l_matToStream, fftOutputStream);

        // printMatStream<k_fftKernelSize, k_fftKernelSize, k_memWidth>(fftOutputStream, "2D FFT Output Natural
        // Order...");
        streamToMatrix<k_fftKernelSize, k_fftKernelSize, k_memWidth, T_elemType, MemWideIFTypeOut>(fftOutputStream,
                                                                                                   l_outMat);
    } // runs loop
    l_fileData = new T_compleFloat[k_fftKernelSize * k_fftKernelSize];
    k = 0;
    for (int r = 0; r < k_fftKernelSize; ++r) {
        for (int c = 0; c < k_fftKernelSize; ++c) {
            l_fileData[k] = l_outMat[r][c];
            k++;
        }
    }
    std::ofstream logStream;
    VerificationResults verif_res_cpp_float_vs_octave;
    verif_res_cpp_float_vs_octave = verifyArrayOutput_with_snr<T_innerFloat>(
        logStream, "Golden Output", goldenOutputDataVerifFileName, l_fileData, 10, 5,
        k_fftKernelSize * k_fftKernelSize); // error given in %

    std::cout << "================================================================" << std::endl;
    std::cout << verif_res_cpp_float_vs_octave << std::endl;
    std::cout << "================================================================" << std::endl;
    return verif_res_cpp_float_vs_octave.m_statusFlag;
}
#endif
