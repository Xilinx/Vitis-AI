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
#include "utils/spu.hpp"
#include "utils/mVerificationUtlityFunctions.hpp"
#include "utils/suf.hpp"
#include "utils/dsp_utilities.hpp"
#include "utils/sorting.hpp"

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
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     * din and dout define 2-dimensional arrays for the storage of input and output complex samples for
     *  complex<double> ssr fft call
     * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */
    T_ComplexDouble din[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    T_ComplexDouble dout[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     * din_fix and dout_fix define 2-dimensional arrays for the storage of input and output complex samples for
     *  complex<ap_fixed> ssr fft call that will synthesize to RTL for implementation
     * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */
    T_SSR_FFT_IN din_fix[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    T_SSR_FFT_OUT dout_fix[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     * din_fix_c and dout_fix_c define 2-dimensional arrays for the storage of input and output complex samples for
     *  complex<ap_fixed> ssr fft call that is NOT SYNTHESIZED creates a bit true output for comparison after COSIM
     * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */
    T_SSR_FFT_IN din_fix_c[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    T_SSR_FFT_OUT dout_fix_c[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// dine_file 1-Dimensional array that will be used to store stimulus input data read from the file, stimulus
    // is created using octave scripts.
    T_ComplexDouble din_file[SSR_FFT_L];

    // golden_output_file is 1-D array that is used to read GOLDEN OUTPUT test vectors for functional verification
    T_ComplexDouble golden_output_file[SSR_FFT_L];

    // The output from the golden_output_file is transformed and stored in 2-D array depending on the choice of radix
    // and the length and stored in array golden_output that is finall used for verification function calls
    T_ComplexDouble golden_output[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];

    std::string root_path = "";        // Define root path where the input stimulus files are stored
    std::string export_root_path = ""; // Define root path where output log files will be stored

    std::ofstream logStreamBB;
    std::string blockBoxFilename = root_path + "ssr_fft_verification_log.log";
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
    strStream1 << root_path << "output_data_R" << SSR_FFT_R << "_L" << SSR_FFT_L << ".verif";
    std::string RecInOutFileName;
    RecInOutFileName = strStream1.str();
    std::ofstream logStream;
    std::string verifLogFileName = root_path + "hls_fft_verification_log.vsum";
    std::string exportDataFile1 = export_root_path + "file1.data";
    // Read input stimulus from the file
    readComplexArrayFromFile<T_innerDB>(logStreamBB, "din_file", inputStimulusDataVerifFileName, din_file, SSR_FFT_L);
    // Read GOLDEN OUTPUT vector for comparison
    readComplexArrayFromFile<T_innerDB>(logStreamBB, "golden_output_file", goldenOutputDataVerifFileName,
                                        golden_output_file, SSR_FFT_L);

    // This loop will transform 1-D data read from the file in 2-D for passing to ssr fft function calls and
    // verification
    for (int i = 0; i < SSR_FFT_L / SSR_FFT_R; i++) {
        for (int j = 0; j < SSR_FFT_R; j++) {
            din[j][i] = din_file[i * (SSR_FFT_R) + j];
            din_fix[j][i] = din_file[i * (SSR_FFT_R) + j];
            din_fix_c[j][i] = din_file[i * (SSR_FFT_R) + j];
            golden_output[j][i] = golden_output_file[i * (SSR_FFT_R) + j];
        }
    }

    /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     *
     *                         F
     *                             F
     *                                 T
     *                                 					**CALLS**
     *
     *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */
    for (int no_ffts = 0; no_ffts < NO_DATA_FRAMES_TO_SIMULATE; no_ffts++) {
        /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         * CAll SSR FFT with complex double type : Reference model call to calculate FLOATING POINT fft that
         * will be used for verifying the implementation and then for comparison with FIXED POINT model to calculate
         *  the SNR
         * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         */
        fft<ssr_fft_params>(din, dout);
        /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         * CAll SSR FFT with complex ap_fixed type : This the actual model that will be synthesized, to generate RTL
         * for implementation
         * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         */
        fft_top(din_fix, dout_fix);
        /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         * CAll SSR FFT with complex ap_fixed type : This function wraps the same C++ model that is wrapped in for
         * synthesis, this function is not synthesized it is only used to compare the final RTL ouput and the C++
         * output during cosimulation for verification. The comparison done with this output has no SIGNIFICANE when
         * the csim/ c++ simulation is run. But during RTL/cosim the output of this model will be used to compare
         * RTL model output and the c++ output ( Bit true verification of RTL model vs. C++ model)
         * ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         */
        fft_top_c(din_fix_c, dout_fix_c);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Verify if the ssr fft double precision model is functionally correct by comparing its output with
    // double precision Octave model output, it functionally verifies SSR FFT model independent of the
    // fact that if fixed point data-path bit-widths are appropriate in case of fixed point ssr fft
    double snrDBs_ssr_vs_octave = snr<SSR_FFT_R, SSR_FFT_L / SSR_FFT_R>(golden_output, dout);
    print_phase_header();
    std::cout << get_section_header()
              << "Comparing C++  SSR FFT double precision model vs. Octave double precision model:\n";
    std::cout << get_section_header() << "Verification Messages\n";
    std::cout << get_section_header() << "The SNR ( SSR FFT double precision vs. Octave double precision): "
              << "     " << snrDBs_ssr_vs_octave << " db \n";
    print_phase_footer();
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compare SSR fft double precision model output with golden reference Octave model to verify the
    // functionality of ssr fft in general, exculding fixed point data-path effects
    print_phase_header();
    std::cout << get_section_header() << "Verification Messages\n";
    std::cout << get_section_header()
              << "Comparing SSR FFT double precision output with golden Octave model output: \n";

    // 1-D array used for conversion of fft_top output from 2-D 1-D
    T_ComplexDouble p_fftOutDataTemp[SSR_FFT_L];
    for (int a1 = 0; a1 < (SSR_FFT_L / SSR_FFT_R); a1++) {
        for (int a2 = 0; a2 < SSR_FFT_R; a2++) {
            p_fftOutDataTemp[a1 * SSR_FFT_R + a2] = dout[a2][a1]; /// convert it to one day array for verification
        }
    }

    int vf3;
    VerificationResults verifResultsFFTvsOctave = verifyArrayOutput_with_snr<T_innerDBOut>(
        logStream, "GOLDEN OUTPUT", goldenOutputDataVerifFileName, p_fftOutDataTemp, MAX_PERCENT_ERROR_IN_SAMPLE,
        MAX_ALLOWED_PERCENTAGE_OF_SAMPLES_IN_ERROR, SSR_FFT_L); // error given in %
    vf3 = verifResultsFFTvsOctave.m_statusFlag;
    exportArrayToFile<T_innerDBOut, SSR_FFT_L>(p_fftOutDataTemp,
                                               RecInOutFileName); // export FIXED POINT fft ouput to file
    if (vf3 != 0) {
        print_error_header();
        std::cout << get_section_header()
                  << "Verification of SSR FFT double precision output vs. Octave double precision output : FAILED"
                  << std::endl;
        std::cout << "Exiting ..." << std::endl;
        print_error_footer();
        return 1;

    } else {
        std::cout << get_section_header()
                  << "Verification of SSR FFT double precision output vs. Octave double precision output : PASSED"
                  << std::endl;
        std::cout << "VERIFIED.....\n" << std::endl;
    }
    std::cout << get_section_header()
              << "The SNR (SSR FFT<double> vs. Octave/Matlab <double>): " << verifResultsFFTvsOctave.m_snr << std::endl;
    print_phase_footer();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     * Cast complex<ap_fixed> to complex <double> for comparison with reference double output
     *++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     */

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Verify if the ssr fft FIXED point model is functionally correct by comparing its output with
    // double precision Octave model output, It verifies if the data-path choice is good enough
    // like the scaling mode, input, output and twiddle bit-width are sufficient for given ssr fft
    T_ComplexDouble fix_fft_out[SSR_FFT_R][SSR_FFT_L / SSR_FFT_R];
    cast_to_double<SSR_FFT_R, SSR_FFT_L / SSR_FFT_R>(dout_fix, fix_fft_out);
    double snrDBs_ssr_fix_vs_octave = snr<SSR_FFT_R, SSR_FFT_L / SSR_FFT_R>(golden_output, fix_fft_out);

    print_phase_header();
    std::cout << get_section_header()
              << "Comparing C++  SSR FFT Single Precision  model vs. Octave double precision model:\n";
    std::cout << get_section_header() << "Verification Messages\n";
    std::cout << get_section_header() << "The SNR ( SSR FFT Single Precision vs. Octave double precision): "
              << "     " << snrDBs_ssr_fix_vs_octave << " db \n";
    print_phase_footer();
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Compare SSR fft fixed point model output against golden reference Octave model to verify the
    // functionality of ssr fft with fixed point data-path effects
    print_phase_header();
    std::cout << get_section_header() << "Verification Messages\n";
    std::cout << get_section_header()
              << "Comparing SSR FFT Single Precision output with golden Octave model output: \n";
    for (int a1 = 0; a1 < (SSR_FFT_L / SSR_FFT_R); a1++) {
        for (int a2 = 0; a2 < SSR_FFT_R; a2++) {
            p_fftOutDataTemp[a1 * SSR_FFT_R + a2] = dout_fix[a2][a1]; // convert to 1d array and also cast
        }
    }

    int vf4;
    VerificationResults verifResultsFixVsOctave;
    verifResultsFixVsOctave = verifyArrayOutput_with_snr<T_innerDBOut>(
        logStream, "GOLDEN OUTPUT", goldenOutputDataVerifFileName, p_fftOutDataTemp, MAX_PERCENT_ERROR_IN_SAMPLE,
        MAX_ALLOWED_PERCENTAGE_OF_SAMPLES_IN_ERROR, SSR_FFT_L); // 50,5
    vf4 = verifResultsFixVsOctave.m_statusFlag;
    if (vf4 != 0) {
        print_error_header();
        std::cout << get_section_header()
                  << "Verification of SSR FFT Single Precision output vs. Octave double precision output : FAILED"
                  << std::endl;
        std::cout << "Exiting ..." << std::endl;
        print_error_footer();
        return 1;
    } else {
        std::cout << get_section_header()
                  << "Verification of SSR FFT Single Precision output vs. Octave double precision output : PASSED"
                  << std::endl;
        std::cout << "VERIFIED.....\n" << std::endl;
    }
    print_phase_footer();
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Compare SSR fft DOUBLE precision model with SSR FFT fixed point model to verify the closeness
    // of fixed point data-path with double precision SSR FFT model.
    print_phase_header();
    std::cout << get_section_header() << "Verification Messages\n";
    std::cout << get_section_header()
              << "Comparing SSR FFT Single Precision model with SSR FFT double precision model: \n";
    double snrDBs_fix_point_vs_floating_model = snr<SSR_FFT_R, SSR_FFT_L / SSR_FFT_R>(dout, fix_fft_out);
    std::cout << "The SNR( SSR FFT Single Precision vs. SSR FFT double precision) is  :"
              << "     " << snrDBs_fix_point_vs_floating_model << " db \n\n";
    print_phase_footer();
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    print_phase_header();
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "These messages aree useful for -----COSIM-----only \notherwise it compares C++ model with itself \n";
    std::cout << get_section_header() << "Verification Messages\n";
    std::cout << get_section_header() << "Comparing RTL model output with C++ fixed Single Precision model. \n";
    std::cout << "------------------COSIM Relevant Verification Messages----------------\n";
    std::cout << "Comparing RTL output with C++ Single Precision model, BIT-TRUE Model Verification \n";
    std::cout << "----------------------------------------------------------------------\n";

    int rtl_verif_flag = 0;
    for (int lt = 0; lt < SSR_FFT_L / SSR_FFT_R; lt++) {
        for (int rad = 0; rad < SSR_FFT_R; rad++) {
            if ((dout_fix[rad][lt].real() != dout_fix_c[rad][lt].real()) ||
                (dout_fix[rad][lt].imag() != dout_fix_c[rad][lt].imag())) {
                rtl_verif_flag++;
                std::cout << "Expected Output : " << dout_fix_c[rad][lt] << "\n";
                std::cout << "RTL      Output : " << dout_fix[rad][lt] << "\n";
                std::cout << "Indices : l:r" << lt << ": " << rad << std::endl;
            }
        }
    }
    if (rtl_verif_flag != 0) {
        print_error_header();
        std::cout << get_section_header()
                  << "Bit-True verification of RTL vs. synthesized C++ Single Precision model FAILED." << std::endl;
        std::cout << "Exiting ..." << std::endl;
        print_error_footer();
        std::cout << "No. of Errors: " << rtl_verif_flag << "\n";
        return 1;
    } else {
        std::cout << get_section_header()
                  << "Bit-True verification of RTL vs. synthesized C++ Single Precision model PASEED." << std::endl;
        std::cout << "VERIFIED.....\n" << std::endl;
    }
    print_phase_footer();

    print_phase_header();

    if ((vf3 | vf4 | rtl_verif_flag) == 0) {
        std::cout << "OVERL ALL Simulation was SUCCESSFULL Done with L=" << SSR_FFT_L << "  R=" << SSR_FFT_R
                  << std::endl;
        std::cout << "SNR Double precision Model  : " << snrDBs_ssr_vs_octave << " dbs" << std::endl;
        std::cout << "SNR Single Precision Model        : " << snrDBs_ssr_fix_vs_octave << " dbs" << std::endl;

    } else {
        std::cout << "OVERL ALL Simulation has FAILED Done with L=" << SSR_FFT_L << "  R=" << SSR_FFT_R << std::endl;
    }

    print_phase_footer();
    return (vf3 | vf4 | rtl_verif_flag);
}
#endif
