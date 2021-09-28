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
#include "DEBUG_CONSTANTS.hpp"
#include <iostream>
#include <assert.h>
#include <bitset>
#include "hls_ssr_fft_data_path.hpp"
#include "../../../common/utils/spu.hpp"
#include "../../../common/utils/mVerificationUtlityFunctions.hpp"
#include "../../../common/utils/suf.hpp"
#include "../../../common/utils/dsp_utilities.hpp"
#include "../../../common/utils/sorting.hpp"
#ifdef TEST_SSR_FFT

void fft_top(T_SSR_FFT_IN inD[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R], T_SSR_FFT_OUT outD[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R]) {
#pragma HLS TOP

    xf::dsp::fft::fft<ssr_fft_params>(inD, outD);
}

void fft_top_c(T_SSR_FFT_IN inD[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R],
               T_SSR_FFT_OUT outD[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R]) {
    xf::dsp::fft::fft<ssr_fft_params>(inD, outD);
}

#ifndef __SYNTHESIS__
int main(int argc, char** argv) {
    complex_wrapper<double> parFFTTable[SSR_FFT_R];
    complex_wrapper<double> twiddleTable[(SSR_FFT_R - 1) * ((SSR_FFT_L / SSR_FFT_R) - 1) + 1];
    const int tw_len = (SSR_FFT_R - 1) * ((SSR_FFT_L / SSR_FFT_R) - 1) + 1;
    complex_wrapper<tip_fftInType> in[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    complex_wrapper<tip_fftOutType> out[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    complex_wrapper<tip_fftOutType> out1[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];

    int stage = 0;
    int dim1 = SSR_FFT_L / SSR_FFT_R;
    for (int i = 0; i < SSR_FFT_L / SSR_FFT_R; i++) {
        for (int j = 0; j < SSR_FFT_R; j++) {
            in[j][i] = complex_wrapper<tip_fftInType>((tip_fftInType)(i * (SSR_FFT_R) + j),
                                                      (tip_fftInType)(i * (SSR_FFT_R) + j));
        }
    }

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     * din and dout define 2-dimensional arrays for the storage of input and output complex samples for
     *  complex<double> ssr fft call
     * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */
    complex_wrapper<double_type> din[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    complex_wrapper<double_type> dout[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     * din_float and dout_float define 2-dimensional arrays for the storage of input and output complex samples for
     *  complex<float> ssr fft call that will synthesize to RTL for implementation
     * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */
    T_SSR_FFT_IN din_float[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    T_SSR_FFT_OUT dout_float[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     * din_float_c and dout_float_c define 2-dimensional arrays for the storage of input and output complex samples for
     *  complex<float> ssr fft call that is NOT SYNTHESIZED create bit true output for comparison during COSIM
     * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */
    T_SSR_FFT_IN din_float_c[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    T_SSR_FFT_OUT dout_float_c[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// dine_file 1-Dimensional array that will be used to store stimulus input data read from the file, stimulus
    // is created using octave scripts.
    complex_wrapper<double_type> din_file[SSR_FFT_L];

    // golden_output_file is 1-D array that is used to read GOLDEN OUTPUT test vectors for functional verification
    complex_wrapper<double_type> golden_output_file[SSR_FFT_L];

    // The output from the golden_output_file is transformed and stored in 2-D array depending on the choice of radix
    // and the length and stored in array golden_output that is finall used for verification function calls
    complex_wrapper<double_type> golden_output[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];

    std::string root_path = "";        // Define root path where the input stimulus files are stored
    std::string export_root_path = ""; // Define root path where output log files will be stored

    std::ofstream logStreamBB;
    std::string blockBoxFilename = root_path + "hls_fft_blackBox_verification_log.vsum";
    logStreamBB.open(blockBoxFilename.c_str(), std::fstream::out);
    std::string inputStimulusDataVerifFileName;
    std::stringstream strStream;
    strStream << root_path << "fftStimulusIn_L" << SSR_FFT_L << ".verif";
    inputStimulusDataVerifFileName = strStream.str();

    std::cout << "The Stimulus file with Path : " << inputStimulusDataVerifFileName << "\n";

    std::stringstream strStream1;
    std::string goldenOutputDataVerifFileName;
    strStream1 << root_path << "fftGoldenOut_L" << SSR_FFT_L << ".verif";
    goldenOutputDataVerifFileName = strStream1.str();
    strStream1.str("");
    strStream1 << root_path << "RecursiveSSRFFT_int_output_R" << SSR_FFT_R << "_L" << SSR_FFT_L << ".verif";
    std::string RecInOutFileName;
    RecInOutFileName = strStream1.str();
    std::ofstream logStream;
    std::string verifLogFileName = root_path + "hls_fft_verification_log.vsum";
    std::string exportDataFile1 = export_root_path + "file1.data";

    // Read input stimulus from the file
    readComplexArrayFromFile<double_type>(logStreamBB, "din_file", inputStimulusDataVerifFileName, din_file, SSR_FFT_L);
    // Read GOLDEN OUTPUT vectore for comparison
    readComplexArrayFromFile<double_type>(logStreamBB, "golden_output_file", goldenOutputDataVerifFileName,
                                          golden_output_file, SSR_FFT_L);

    // This loop will transform 1-D data read from the file in 2-D for passing to ssr fft function calls and verificaton
    for (int i = 0; i < SSR_FFT_L / SSR_FFT_R; i++) {
        for (int j = 0; j < SSR_FFT_R; j++) {
            din[j][i] = din_file[i * (SSR_FFT_R) + j];
            din_float[j][i] = din_file[i * (SSR_FFT_R) + j];
            din_float_c[j][i] = din_file[i * (SSR_FFT_R) + j];
            golden_output[j][i] = golden_output_file[i * (SSR_FFT_R) + j];
        }
    }

    /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     *
     *                          F
     *                             F
     *                                 T
     *                                 					*CALLs*
     *
     *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */
    for (int no_ffts = 0; no_ffts < NO_DATA_FRAMES_TO_SIMULATE; no_ffts++) {
        /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         * CAll SSR FFT with complex double type : Reference model call to calculate FLOATING POINT fft that
         * will be used for verifying the implementation and then for comparison with floating (SP) POINT model to
         * calculate the SNR
         * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         */
        fft<ssr_fft_params>(din, dout);
        /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         * CAll SSR FFT with complex float type : This the actual model that will be synthesized, to generate RTL
         * for implementation
         * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         */
        fft_top(din_float, dout_float);
        /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         * CAll SSR FFT with complex flaot type : This function wraps the same C++ model that is wrapped in for
         * synthesis, this function is not synthesized it is only used to compare the final RTL ouput and the C++
         * output during cosimulation for verification. The comparison done with this output has no SIGNIFICANE when
         * the csim/ c++ simulation is run. But during RTL/cosim the output of this model will be used to compare
         * RTL model output and the c++ output ( Bit true verification of RTL model vs. C++ model)
         * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         */
        fft_top_c(din_float_c, dout_float_c);
    }

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Verify Double Precision Model Against the GOLDEN OUTPUT VECTORS////////////////////////////////////////////START
    double floating_point_model_snrDBs = snr<SSR_FFT_R, SSR_FFT_L / SSR_FFT_R>(golden_output, dout);
    std::cout << "\n\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Verification Messages:\"Double Precision Model\" VS. GOLDEN OUTPUT Comparison\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n\n";
    std::cout << "The Signal to Noise Ratio Compares Signal generated by \"Double Precision C++ model\" and the "
                 "\"GOLDEN OUTPUT VECTORS \".\n";
    std::cout << "The SNR is           :"
              << "     " << floating_point_model_snrDBs << " db \n\n";

// exit(0);
#if 0
	std::cout<<"SSR Double Precision Type vs: SSR Float (SP) Type Comparison..........\n";
	SidebySidePrintComplexNumArrays<SSR_FFT_R,SSR_FFT_L/SSR_FFT_R>(dout,float_fft_out);


	std::cout<<"SSR Double Precision Type vs: Golden Expected output..........\n";
	SidebySidePrintComplexNumArrays<SSR_FFT_R,SSR_FFT_L/SSR_FFT_R>(dout,golden_output);
#endif

    std::cout << "Comparing C++ \"Double Precision model Output\" with the expected \"GOLDEN OUTPUT\" generated from "
                 "Octave FFT model. \n\n";

    complex_wrapper<tip_fftOutType>
        p_fftOutDataTemp[SSR_FFT_L]; // 1-D array used for conversion of fft_top output from 2-D 1-D
    for (int a1 = 0; a1 < (SSR_FFT_L / SSR_FFT_R); a1++) {
        for (int a2 = 0; a2 < SSR_FFT_R; a2++) {
            p_fftOutDataTemp[a1 * SSR_FFT_R + a2] = dout[a2][a1]; /// convert it to one day array for verification
        }
    }

    int vf3 = verifyArrayOutput<tip_fftOutType>(
        logStream, "GOLDEN OUTPUT", goldenOutputDataVerifFileName, p_fftOutDataTemp, MAX_PERCENT_ERROR_IN_SAMPLE,
        MAX_ALLOWED_PERCENTAGE_OF_SAMPLES_IN_ERROR, SSR_FFT_L); // error given in %

    exportArrayToFile<tip_fftOutType, SSR_FFT_L>(p_fftOutDataTemp,
                                                 RecInOutFileName); // export Floating POINT(SP) fft ouput to file

    if (vf3 != 0) {
        std::cout << "\n\nVerification of C++ Double Precision model Failed ..... FAIL " << std::endl;

    } else {
        std::cout << "\n\n\nVerification of C++ Double Precision model PASSED..... SUCCESS " << std::endl;
        std::cout << "VERIFIED.....\n" << std::endl;
    }
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "=======================================================================\n\n\n";

    std::cout << "\n\n\n=======================================================================\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Verification Messages:\"Floating POINT (SP) Model\" VS. GOLDEN OUTPUT Comparison\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Comparing \"C++ Floating POINT (SP) model\" output with the expected \"GOLDEN OUTPUT\" generated "
                 "from Octave FFT model \n";

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     * Cast complex<float> to complex <double> for comparison with reference double output
     *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */
    complex_wrapper<double_type> float_fft_out[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    cast_to_double<SSR_FFT_R, SSR_FFT_L / SSR_FFT_R>(dout_float, float_fft_out);
    double float_point_model_snrDBs = snr<SSR_FFT_R, SSR_FFT_L / SSR_FFT_R>(golden_output, float_fft_out);
    std::cout << "The Signal to Noise Ratio Compares Signal generated by \"Floating Single Precision C++ model\" and "
                 "the \"GOLDEN OUTPUT VECTORS \".\n";
    std::cout << "The SNR is           :"
              << "     " << float_point_model_snrDBs << " db \n\n";

    for (int a1 = 0; a1 < (SSR_FFT_L / SSR_FFT_R); a1++) {
        for (int a2 = 0; a2 < SSR_FFT_R; a2++) {
            p_fftOutDataTemp[a1 * SSR_FFT_R + a2] = dout_float[a2][a1]; // convert to 1d array and also cast
        }
    }

    std::stringstream strFile;
    strFile.str("");
    std::cout << "Comparing C++ \"Floating Single Precision model Output\" with the expected \"GOLDEN OUTPUT\" "
                 "generated from Octave FFT model. \n\n";

    int vf4 = verifyArrayOutput<tip_fftOutType>(logStream, "GOLDEN OUTPUT", goldenOutputDataVerifFileName,
                                                p_fftOutDataTemp, MAX_PERCENT_ERROR_IN_SAMPLE,
                                                MAX_ALLOWED_PERCENTAGE_OF_SAMPLES_IN_ERROR, SSR_FFT_L); // 50,5
    if (vf4 != 0) {
        std::cout << "Verification of \"C++ Floating Single Precision model\" Failed..... FAIL " << std::endl;
    } else {
        std::cout << "\n\n\nVerification of \"C++ Floating Single Precision model\" PASSED..... SUCCESS " << std::endl;
        std::cout << "VERIFIED.....\n\n" << std::endl;
    }
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "=======================================================================\n\n\n";

    std::cout << "\n\n\n=======================================================================\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout
        << "Verification Messages:\"Floating Single Precision MODEL\" VS. \"Double Precision MODEL\" Comparison\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Comparing \"C++ Floating Single Precision MODEL\" output with \"Double Precision MODEL\" output \n";
    double float_point_vs_floating_model_snrDBs =
        snr<SSR_FFT_R, SSR_FFT_L / SSR_FFT_R>(dout, float_fft_out); // dour= ref signal and float_fft_out= noisy signal
    std::cout << "The Signal to Noise Ratio Compares Signal generated by \"Double Precision MODEL C++ model\" and the "
                 "\"FLOATING POINT Octave MODEL\".\n";
    std::cout << "The SNR is           :"
              << "     " << float_point_vs_floating_model_snrDBs << " db \n\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "=======================================================================\n\n\n";

    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "COSIM Relevant Verification Messages----------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Comparing RTL output with C++ Floating Single Precision MODEL, BIT-TRUE Model Verification \n";
    std::cout << "This message is useful for -----COSIM--- only otherwise it compare C++ model with itself......... \n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";

    int rtl_verif_flag = 0;
    for (int lll = 0; lll < SSR_FFT_L / SSR_FFT_R; lll++) {
        for (int rrr = 0; rrr < SSR_FFT_R; rrr++) {
            if ((dout_float[rrr][lll].real() != dout_float_c[rrr][lll].real()) ||
                (dout_float[rrr][lll].imag() != dout_float_c[rrr][lll].imag())) {
                rtl_verif_flag++;
                std::cout << "Expected Output : " << dout_float_c[rrr][lll] << "\n";
                std::cout << "RTL      Output : " << dout_float[rrr][lll] << "\n";
            }
        }
    }
    if (rtl_verif_flag != 0) {
        std::cout << "\"C++ MODEL\" Floating Single Precision MODEL output and the synthesized Floating Single "
                     "Precision MODEL \"RTL MODEL\" output does not match verification failed..... FAIL "
                  << std::endl;
        std::cout << "Number of Mis-Matches=" << rtl_verif_flag << "\n";
    } else {
        std::cout << "Floating Single Precision MODEL \"C++ MODEL\"  output and the synthesized Floating Single "
                     "Precision MODEL \"RTL MODEL\" outputs matched..... SUCCESS "
                  << std::endl;
        std::cout << "VERIFIED.....\n" << std::endl;
    }
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "----------------------------------------------------------------------\n";

#if 0
	SidebySidePrintComplexNumArrays<SSR_FFT_R,SSR_FFT_L/SSR_FFT_R,double_type>(dout,float_fft_out);
	std::cout<<"Comparison with FLOATING POINT output ....\n";
	SidebySidePrintComplexNumArrays<SSR_FFT_L/SSR_FFT_R,SSR_FFT_R,T_INNER_SSR_FFT_OUT,double_type>(dout_float,dout);
#endif
    if ((vf3 | vf4 | rtl_verif_flag) == 0)
        std::cout << "\n=============================\nOVERL ALL Simulation was a SUCCESS Done with L=" << SSR_FFT_L
                  << "  R=" << SSR_FFT_R << "\n============================\n"
                  << std::endl;
    return (vf3 | vf4 | rtl_verif_flag);
}
#endif
#endif
