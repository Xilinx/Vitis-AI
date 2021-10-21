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
#ifndef MVERIFICATIONUTLITYFUNCTIONS_H_
#define MVERIFICATIONUTLITYFUNCTIONS_H_
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include "dateAndTime.hpp"
#include <math.h>
#include <complex>
#include <assert.h>
#include "vitis_fft/fft_complex.hpp"
struct VerificationResults {
    int m_statusFlag;
    double m_snr; // in dbs
    double m_rms; // in dbs
};
std::ostream& operator<<(std::ostream& p_os, const VerificationResults& p_results) {
    return p_os << "Status Flag : " << p_results.m_statusFlag << "\nSNR       : " << p_results.m_snr
                << "\nRMS Error : " << p_results.m_rms;
}
struct DataMinMaxStats {
    double m_maxReal = 0;
    double m_minReal = 0;
    double m_maxImag = 0;
    double m_minImag = 0;
    double m_maxAbs = 0;
    double m_minAbs = 0;
    double m_averageReal = 0;
    double m_averageImag = 0;
    double m_averageAbs = 0;
};
std::ostream& operator<<(std::ostream& p_os, const DataMinMaxStats& p_dataStats) {
    return p_os << "Maximum Real : " << p_dataStats.m_maxReal << "\nMinimum Real : " << p_dataStats.m_minReal

                << "\nMax Imag     : " << p_dataStats.m_maxImag << "\nMinimum Imag : " << p_dataStats.m_minImag

                << "\nMax Abs      : " << p_dataStats.m_maxAbs << "\nMinimum Abs  : " << p_dataStats.m_minAbs

                << "\nAverage Abs  : " << p_dataStats.m_averageAbs << "\nAverage Real : " << p_dataStats.m_averageReal

                << "\nAverage Imag : " << p_dataStats.m_averageImag << std::endl;
}
template <typename T_dType>
T_dType abs(complex_wrapper<T_dType> inSample) {
    double real = (T_dType)inSample.real();
    double imag = (T_dType)inSample.real();
    double result = sqrt((real * real + imag * imag));
}

inline bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}
template <typename T>
void println(T obj) {
    std::cout << obj << std::endl;
}
template <typename T>
void print(T obj) {
    std::cout << obj;
}
template <typename T_type, int L>
int exportArrayToFile(complex_wrapper<T_type> dataArray[L], std::string fileName) {
    int flag = 0;
    std::ofstream outFile;
    const char* tempFileName = fileName.c_str();
    outFile.open(tempFileName, std::ios::out);
    if (outFile.is_open() == false) {
        std::cout << "Cannot Open :" << fileName << " For writing output data...." << std::endl;
        std::cout << "Exiting...." << std::endl;
        flag = 1;
    }
    outFile << L << std::endl;
    for (int i = 0; i < L; i++) {
        outFile << dataArray[i].real() << std::endl;
        outFile << dataArray[i].imag() << std::endl;
    }
    return flag;
}

template <typename T_type, int rows, int cols>
int export2DArrayToFile(complex_wrapper<T_type> dataArray[rows][cols], std::string fileName) {
    int flag = 0;
    std::ofstream outFile;
    const char* tempFileName = fileName.c_str();
    outFile.open(tempFileName, std::ios::out);
    std::cout << "\n----------------File Export Message --------------\n";
    if (outFile.is_open() == false) {
        std::cout << "Cannot Open :" << fileName << " For writing output data...." << std::endl;
        std::cout << "Exiting...." << std::endl;
        flag = 1;
    }
    outFile << cols << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFile << dataArray[i][j].real() << std::endl;
            outFile << dataArray[i][j].imag() << std::endl;
        }
    }
    if (flag == 0) std::cout << "\n--------------File Exported Successfully----------\n";
    std::cout << "The export path is::" << fileName << "\n";
    return flag;
}

template <typename T_type, int rows, int cols>
int exportArrayToFile(complex_wrapper<T_type> dataArray[rows][cols], std::string fileName) {
    std::ofstream outFile;
    const char* tempFileName = fileName.c_str();
    outFile.open(tempFileName, std::ios::out);
    if (outFile.is_open() == false) {
        std::cout << "Cannot Open :" << fileName << " For writing output data...." << std::endl;
        std::cout << "Exiting...." << std::endl;
        exit(1);
    }
    outFile << rows * cols << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFile << dataArray[i][j].real() << std::endl;
            outFile << dataArray[i][j].imag() << std::endl;
        }
    }
}

std::string get_error_header() {
    return "\n\ns!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
}
std::string get_error_footer() {
    return "e!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n";
}
void print_error_header() {
    std::cout << get_error_header();
}
void print_error_footer() {
    std::cout << get_error_footer();
}
std::string get_phase_footer() {
    return "e-p========================================================================================================"
           "==e-p\n\n";
}
std::string get_phase_header() {
    return "\n\ns-p===================================================================================================="
           "======s-p\n";
}
void print_phase_header() {
    std::cout << get_phase_header();
}
void print_phase_footer() {
    std::cout << get_phase_footer();
}
std::string get_function_footer() {
    return "e==============================================================================e\n\n";
}
std::string get_function_header() {
    return "\n\ns==============================================================================s\n";
}
void print_function_header() {
    std::cout << get_function_header();
}
void print_function_footer() {
    std::cout << get_function_footer();
}

std::string get_section_header() {
    return "----";
}
void print_section_header() {
    std::cout << get_section_header();
}

// This function will read a file and compare it with expected output for functional verification
// It will print max error, min error, and mean square error for any given comparison
// It will also print log information in a file passed through ofstream
template <typename T_dType>
VerificationResults verifyArrayOutput_with_snr2(std::ofstream& logStream,
                                                std::string strArrayName,
                                                std::string fiofLog,
                                                complex_wrapper<T_dType> inData[],
                                                T_dType maxAllowedABSError,
                                                int percentageAllowedError,
                                                int dataLength) {
    VerificationResults verif_result;
    logStream << currentDateTime() << std::endl;
    std::cout << currentDateTime() << std::endl;
    logStream << "=====================================================================================\n Starting "
                 "verification against matlab golden reference model."
              << std::endl;
    std::cout << "=====================================================================================\n Starting "
                 "verification against matlab golden reference model."
              << std::endl;
    std::fstream fiofFile;
    logStream << "Trying to Open Gold test vectors file: " << fiofLog << std::endl;
    std::cout << "Trying to Open Gold test vectors file: " << fiofLog << std::endl;
    const char* fiofLogPtr = fiofLog.c_str();
    fiofFile.open(fiofLogPtr, std::ios_base::in);
    int veriFlag = 0;
    int verif_totalNumberOfErrors = 0;
    int sampleCompared = 0;
    int verif_status = 0;
    // Variables to collect SNR ///////////////////// /////////////////////////////////////////START
    double totalSignalPower = 0;
    double totalNoisePower = 0;
    double calSNR;
    if (fiofFile.is_open()) /// Try to open the file for data verification
    {
        // Verification data file opened successfully start verification process

        // Verification Start
        logStream << "Opened verification data file :  " << fiofLog << " Successfully" << std::endl;
        std::cout << "Opened verification data file :  " << fiofLog << " Successfully" << std::endl;

        // Variables to collect SNR ///////////////////////////////////////////////////////////////END
        // Variables to collect absolute error statistics /////////////////////////////////////////START
        double currentABSError;
        double maxABSError;
        double accumABSError = 0;
        // Variables to collect absolute error statistics /////////////////////////////////////////END

        // Variables to collect  error statistics of REAL PART ////////////////////////////////////START
        double currentREALError = 0;
        double maxREALError;
        double minREALError;
        double accumREALError = 0;
        // Variables to collect  error statistics of REAL PART //////////////////////////////////////END

        // Variables to collect  error statistics of IMAG PART ////////////////////////////////////START
        double currentIMAGError;
        double maxIMAGError;
        double minIMAGError;
        double accumIMAGError = 0;
        // Variables to collect  error statistics of IMAG PART //////////////////////////////////////END
        // complexNum<T_dType> goldenSample;
        int dataLengthFromFile;
        if (fiofFile.eof()) // Reaing Array Length for Verification: The verification data in the file is not
                            // appropriate, less number of samples
        {
            logStream << "The data length in the verification file is less then the required no. of samples..."
                      << std::endl
                      << "Exiting..." << std::endl;
            std::cout << "The data length in the verification file is less then the required no. of samples..."
                      << std::endl
                      << "Exiting..." << std::endl;
            veriFlag = 1;
        } else // Array Length for Verfication is available for reading... /d
        {
            fiofFile >> dataLengthFromFile;
            if (dataLengthFromFile != dataLength) {
                logStream
                    << "The data length in the file does not matches the data array length passed for verification..."
                    << std::endl
                    << "Exiting..." << std::endl;
                std::cout
                    << "The data length in the file does not matches the data array length passed for verification..."
                    << std::endl
                    << "Exiting..." << std::endl;
                veriFlag = 1;
            } else // d
            {
                for (int i = 0; i < dataLength; i++) {
                    if (fiofFile.eof()) // The verification data in the file is not appropriate, less number of samples
                    {
                        logStream
                            << "The data length in the verification file is less then the required no. of samples..."
                            << std::endl
                            << "Exiting..." << std::endl;
                        std::cout
                            << "The data length in the verification file is less then the required no. of samples..."
                            << std::endl
                            << "Exiting..." << std::endl;
                        veriFlag = 1;
                        break;
                    } else // File opened and verification data available
                    {
                        double goldenSample_real;
                        double goldenSample_imag;
                        fiofFile >> goldenSample_real; // goldenSample.real;
                        fiofFile >> goldenSample_imag; // goldenSample.imag;
                        // goldenSample = complex_wrapper<T_dType>(temp_real,temp_imag);
                        // complexNum<T_dType> temp  = inData[i] - goldenSample;
                        double inData_real, inData_imag;

                        inData_real = inData[i].real();
                        inData_imag = inData[i].imag();
                        currentREALError = inData_real - goldenSample_real;
                        currentIMAGError = inData_imag - goldenSample_imag;
                        totalNoisePower += currentREALError * currentREALError + currentIMAGError * currentIMAGError;
                        totalSignalPower += inData_real * inData_real + inData_imag * inData_imag;

                        currentABSError =
                            sqrt(currentIMAGError * currentIMAGError + currentREALError * currentREALError);
                        /// kepp track of max and min errors
                        if (i == 0) // first sample read for verification so init maximums and minimums
                        {
                            maxABSError = currentABSError;
                            maxREALError = currentREALError;
                            maxIMAGError = currentIMAGError;
                            minREALError = currentREALError;
                            minIMAGError = currentIMAGError;
                        } else {
                            if (currentABSError < 0) currentABSError *= -1;
                            // Check for maximum errors
                            if (currentABSError > maxABSError) maxABSError = currentABSError;
                            if (currentREALError > maxREALError) maxREALError = currentREALError;
                            if (currentIMAGError > maxIMAGError) maxIMAGError = currentIMAGError;
                            // Check for minimum errors
                            if (currentREALError < minREALError) minREALError = currentREALError;
                            if (currentIMAGError < minIMAGError) minIMAGError = currentIMAGError;
                            double golden_sample_abs =
                                sqrt(goldenSample_imag * goldenSample_imag + goldenSample_real * goldenSample_real);
                            double percent_error = (currentABSError / golden_sample_abs) * 100;
                            if (percent_error > maxAllowedABSError) {
                                logStream << "The data sample different significantly, comparing sample no:" << i
                                          << " The error is : " << percent_error << "%" << std::endl;
                                std::cout << "The data sample different significantly, comparing sample no:" << i
                                          << " The error is : " << percent_error << "%"
                                          << std::endl; //<< "The data sample different significantly, comparing sample
                                                        // no:"<<i<< std::endl;
                                std::cout << "Absolute Error : " << currentABSError << std::endl;
                                // veriFlag=1;
                                verif_totalNumberOfErrors++;
                            }
                            accumABSError += currentABSError;
                            accumREALError += currentREALError;
                            accumIMAGError += currentIMAGError;
                        }
                    }
                    sampleCompared++;
                }
                logStream << "----------------------------Verification Summary--------------------------------"
                          << std::endl;
                std::cout << "----------------------------Verification Summary--------------------------------"
                          << std::endl;
                int temp_percent_mismatch = int(double(verif_totalNumberOfErrors * 100) / double(dataLength));
                // verification successfull log statistics !!
                if (veriFlag == 0 && sampleCompared == dataLength && temp_percent_mismatch < percentageAllowedError) {
                    logStream << "Verification for data array" << strArrayName << "is successful." << std::endl;
                    std::cout << "Verification for data array" << strArrayName << "is successful." << std::endl;
                    verif_status = 0;
                } else {
                    logStream << "Verification for data array :" << strArrayName << "has FAILED." << std::endl;
                    std::cout << "Verification for data array : " << strArrayName << "has FAILED." << std::endl;
                    verif_status = 1;
                }
                //////////////////////////////////////////////////////////////////////////////////////
                logStream << "Error Rate is:" << temp_percent_mismatch << " % " << std::endl;
                std::cout << "Error Rate is:" << temp_percent_mismatch << " % " << std::endl;

                logStream << "Total Number of samples mismatched :" << verif_totalNumberOfErrors << " % " << std::endl;
                std::cout << "Total Number of samples mismatched :" << verif_totalNumberOfErrors << " % " << std::endl;

                logStream << "The \"MAXIMUM ABSOLUTE ERROR\" : " << maxABSError << std::endl;
                std::cout << "The \"MAXIMUM ABSOLUTE ERROR\" : " << maxABSError << std::endl;

                logStream << "The \"MAXIMUM REAL ERROR\" : " << maxREALError << std::endl;
                std::cout << "The \"MAXIMUM REAL ERROR\" : " << maxREALError << std::endl;

                logStream << "The \"MAXIMUM IMAG ERROR\" : " << maxIMAGError << std::endl;
                std::cout << "The \"MAXIMUM IMAG ERROR\" : " << maxIMAGError << std::endl;

                //////////////////////////////////////////////////////////////////////////////////////

                logStream << "The \"AVERAGE ABSOLUTE ERROR\" : " << (accumABSError / dataLength) << std::endl;
                std::cout << "The \"AVERAGE ABSOLUTE ERROR\" : " << (accumABSError / dataLength) << std::endl;

                logStream << "The \"AVERAGE REAL ERROR\" : " << (accumREALError / dataLength) << std::endl;
                std::cout << "The \"AVERAGE REAL ERROR\" : " << (accumREALError / dataLength) << std::endl;

                logStream << "The \"AVERAGE IMAG ERROR\" : " << (accumIMAGError / dataLength) << std::endl;
                std::cout << "The \"AVERAGE IMAG ERROR\" : " << (accumIMAGError / dataLength) << std::endl;

                //////////////////////////////////////////////////////////////////////////////////////

                logStream << "The \"MINIMUM REAL  ERROR\" : " << minREALError << std::endl;
                std::cout << "The \"MINIMUM REAL  ERROR\" : " << minREALError << std::endl;

                logStream << "The \"MINIMUM IMAG  ERROR\" : " << minIMAGError << std::endl;
                std::cout << "The \"MINIMUM IMAG  ERROR\" : " << minIMAGError << std::endl;
                calSNR = 10 * log10(totalSignalPower / totalNoisePower);
                std::cout << "-----------------------------------------------------------------------------------------"
                             "-----------------------\n";
                std::cout << "--------The signal to noise ratio : SNR ( verification version, compared against Octave "
                             "output) :: "
                          << calSNR << " dbs";
                std::cout << "-----------------------------------------------------------------------------------------"
                             "-----------------------\n";
                std::cout << "=====================================================================================\n";

            } //%%%
        }     ///%%%
    } else    // Cannot Open verification data File SUCCCESSFULLY Error out !!
    {
        logStream << "Cannot Open input data file for verfication..." << std::endl << "Exiting..." << std::endl;
        std::cout << "Cannot Open input data file for verfication..." << std::endl << "Exiting..." << std::endl;
        verif_status = 1;
    }
    verif_result.m_statusFlag = verif_status;
    verif_result.m_snr = calSNR;

    // return verif_status;
    return verif_result;
}
// This function will open golden test vector file and collect statistics
template <typename T_dType>
DataMinMaxStats calculateDataStats(std::string p_inputDataFileName, int p_dataLength) {
    const char* inputDataFilePtr = p_inputDataFileName.c_str();
    std::fstream inputDataFileStream;
    inputDataFileStream.open(inputDataFilePtr, std::ios_base::in);
    int dataLength = 0;
    DataMinMaxStats dataStats;
    double realPart;
    double imagPart;
    if (inputDataFileStream.is_open()) {
        inputDataFileStream >> dataLength;
        assert(dataLength == p_dataLength); // The data length in file should be same
        for (int i = 0; i < dataLength; ++i) {
            inputDataFileStream >> realPart;
            inputDataFileStream >> imagPart;
            double abs = sqrt(realPart * realPart + imagPart * imagPart);
            if (i == 0) {
                dataStats.m_averageReal = realPart;
                dataStats.m_averageImag = imagPart;
                dataStats.m_averageAbs = abs;
                dataStats.m_maxAbs = abs;
                dataStats.m_minAbs = abs;
                dataStats.m_maxImag = imagPart;
                dataStats.m_minImag = imagPart;
                dataStats.m_maxReal = realPart;
                dataStats.m_minReal = realPart;
            } else {
                if (abs > dataStats.m_maxAbs) dataStats.m_maxAbs = abs;

                if (abs < dataStats.m_minAbs) dataStats.m_minAbs = abs;

                if (realPart > dataStats.m_maxReal) dataStats.m_maxReal = realPart;

                if (imagPart > dataStats.m_maxImag) dataStats.m_maxImag = imagPart;

                if (realPart < dataStats.m_minReal) dataStats.m_minReal = realPart;

                if (imagPart < dataStats.m_minImag) dataStats.m_minImag = imagPart;

                dataStats.m_averageReal += realPart;
                dataStats.m_averageImag += imagPart;
            }
        } // end data for loop
        dataStats.m_averageAbs /= dataLength;
        dataStats.m_averageReal /= dataLength;
        dataStats.m_averageImag /= dataLength;
    } else {
        std::cout << "Cannot open golden test vector file for stats" << std::endl;
        exit(1);
    }
    return dataStats;
}
// This function will read a file and compare it with expected output for functional verification
// It will print max error, min error, and mean square error for any given comparison
// It will also print log information in a file passed through ofstream
template <typename T_dType>
VerificationResults verifyArrayOutput_with_snr(std::ofstream& logStream,
                                               std::string strArrayName,
                                               std::string fiofLog,
                                               complex_wrapper<T_dType> inData[],
                                               T_dType maxAllowedABSError,
                                               int percentageAllowedError,
                                               int dataLength) {
    VerificationResults verif_result;
    logStream << currentDateTime() << std::endl;
    std::cout << currentDateTime() << std::endl;
    logStream << get_function_header() << " Starting verification against matlab golden reference model." << std::endl;
    print_function_header();
    std::cout << "Starting verification against matlab golden reference model." << std::endl;
    std::fstream fiofFile;
    logStream << "Trying to Open Gold test vectors file: " << fiofLog << std::endl;
    std::cout << "Trying to Open Gold test vectors file: " << fiofLog << std::endl;
    const char* fiofLogPtr = fiofLog.c_str();
    fiofFile.open(fiofLogPtr, std::ios_base::in);
    int veriFlag = 0;
    int verif_totalNumberOfErrors = 0;
    int sampleCompared = 0;
    int verif_status = 0;
    // Variables to collect SNR ///////////////////// /////////////////////////////////////////START
    double totalSignalPower = 0;
    double totalNoisePower = 0;
    double calSNR;
    DataMinMaxStats dataStats = calculateDataStats<T_dType>(fiofLog, dataLength);
    std::cout << dataStats;
    if (fiofFile.is_open()) /// Try to open the file for data verification
    {
        // Verification data file opened successfully start verification process

        // Verification Start
        logStream << "Opened verification data file :  " << fiofLog << " Successfully" << std::endl;
        std::cout << "Opened verification data file :  " << fiofLog << " Successfully" << std::endl;

        // Variables to collect SNR ///////////////////////////////////////////////////////////////END
        // Variables to collect absolute error statistics /////////////////////////////////////////START
        double currentABSError;
        double maxABSError;
        double accumABSError = 0;
        // Variables to collect absolute error statistics /////////////////////////////////////////END

        // Variables to collect  error statistics of REAL PART ////////////////////////////////////START
        double currentREALError = 0;
        double maxREALError;
        double minREALError;
        double accumREALError = 0;
        // Variables to collect  error statistics of REAL PART //////////////////////////////////////END

        // Variables to collect  error statistics of IMAG PART ////////////////////////////////////START
        double currentIMAGError;
        double maxIMAGError;
        double minIMAGError;
        double accumIMAGError = 0;
        // Variables to collect  error statistics of IMAG PART //////////////////////////////////////END
        // complexNum<T_dType> goldenSample;
        int dataLengthFromFile;
        if (fiofFile.eof()) // Reaing Array Length for Verification: The verification data in the file is not
                            // appropriate, less number of samples
        {
            logStream << "The data length in the verification file is less then the required no. of samples..."
                      << std::endl
                      << "Exiting..." << std::endl;
            std::cout << "The data length in the verification file is less then the required no. of samples..."
                      << std::endl
                      << "Exiting..." << std::endl;
            veriFlag = 1;
        } else // Array Length for Verfication is available for reading... /d
        {
            fiofFile >> dataLengthFromFile;
            if (dataLengthFromFile != dataLength) {
                logStream
                    << "The data length in the file does not matches the data array length passed for verification..."
                    << std::endl
                    << "Exiting..." << std::endl;
                std::cout
                    << "The data length in the file does not matches the data array length passed for verification..."
                    << std::endl
                    << "Exiting..." << std::endl;
                veriFlag = 1;
            } else // d
            {
                for (int i = 0; i < dataLength; i++) {
                    if (fiofFile.eof()) // The verification data in the file is not appropriate, less number of samples
                    {
                        logStream
                            << "The data length in the verification file is less then the required no. of samples..."
                            << std::endl
                            << "Exiting..." << std::endl;
                        std::cout
                            << "The data length in the verification file is less then the required no. of samples..."
                            << std::endl
                            << "Exiting..." << std::endl;
                        veriFlag = 1;
                        break;
                    } else // File opened and verification data available
                    {
                        double goldenSample_real;
                        double goldenSample_imag;
                        fiofFile >> goldenSample_real; // goldenSample.real;
                        fiofFile >> goldenSample_imag; // goldenSample.imag;
                        // goldenSample = complex_wrapper<T_dType>(temp_real,temp_imag);
                        // complexNum<T_dType> temp  = inData[i] - goldenSample;
                        double inData_real, inData_imag;

                        inData_real = inData[i].real();
                        inData_imag = inData[i].imag();
                        currentREALError = inData_real - goldenSample_real;
                        currentIMAGError = inData_imag - goldenSample_imag;
                        totalNoisePower += currentREALError * currentREALError + currentIMAGError * currentIMAGError;
                        totalSignalPower += inData_real * inData_real + inData_imag * inData_imag;

                        currentABSError =
                            sqrt(currentIMAGError * currentIMAGError + currentREALError * currentREALError);
                        /// kepp track of max and min errors
                        if (i == 0) // first sample read for verification so init maximums and minimums
                        {
                            maxABSError = currentABSError;
                            maxREALError = currentREALError;
                            maxIMAGError = currentIMAGError;
                            minREALError = currentREALError;
                            minIMAGError = currentIMAGError;
                        } else {
                            if (currentABSError < 0) currentABSError *= -1;
                            // Check for maximum errors
                            if (currentABSError > maxABSError) maxABSError = currentABSError;
                            if (currentREALError > maxREALError) maxREALError = currentREALError;
                            if (currentIMAGError > maxIMAGError) maxIMAGError = currentIMAGError;
                            // Check for minimum errors
                            if (currentREALError < minREALError) minREALError = currentREALError;
                            if (currentIMAGError < minIMAGError) minIMAGError = currentIMAGError;
                            double golden_sample_abs =
                                sqrt(goldenSample_imag * goldenSample_imag + goldenSample_real * goldenSample_real);
                            // double percent_error = (currentABSError/golden_sample_abs)*100;
                            double percent_error = (currentABSError / dataStats.m_maxAbs) * 100;
                            if (percent_error > maxAllowedABSError) {
                                logStream << "The data sample different significantly, comparing sample no:" << i
                                          << " The error is : " << percent_error << "%" << std::endl;
                                std::cout << "Golden Output: ( " << goldenSample_real << ", " << goldenSample_imag
                                          << " Output Sample: ( " << inData_real << ", " << inData_imag << ")"
                                          << std::endl;
                                std::cout << "The data sample different significantly, comparing sample no:" << i
                                          << " The error is : " << percent_error << "%"
                                          << std::endl; //<< "The data sample different significantly, comparing sample
                                                        // no:"<<i<< std::endl;
                                                        // veriFlag=1;
                                std::cout << "Absolute Error : " << currentABSError << std::endl;
                                verif_totalNumberOfErrors++;
                            }
                            accumABSError += currentABSError;
                            accumREALError += currentREALError;
                            accumIMAGError += currentIMAGError;
                        }
                    }
                    sampleCompared++;
                }
                logStream << get_section_header() << "Verification Summary:" << std::endl;
                std::cout << get_section_header() << "Verification Summary:" << std::endl;
                int temp_percent_mismatch = int(double(verif_totalNumberOfErrors * 100) / double(dataLength));
                // verification successfull log statistics !!
                if (veriFlag == 0 && sampleCompared == dataLength && temp_percent_mismatch < percentageAllowedError) {
                    logStream << "Verification for data array" << strArrayName << " is successful." << std::endl;
                    std::cout << "Verification for data array" << strArrayName << " is successful." << std::endl;
                    verif_status = 0;
                } else {
                    logStream << "Verification for data array :" << strArrayName << "has FAILED." << std::endl;
                    std::cout << "Verification for data array : " << strArrayName << "has FAILED." << std::endl;
                    verif_status = 1;
                }
                //////////////////////////////////////////////////////////////////////////////////////
                logStream << get_section_header() << "Error Rate Summary";
                logStream << "Error Rate is:" << temp_percent_mismatch << " % " << std::endl;
                std::cout << "Error Rate is:" << temp_percent_mismatch << " % " << std::endl;

                logStream << "Total Number of samples mismatched :" << verif_totalNumberOfErrors << " % " << std::endl;
                std::cout << "Total Number of samples mismatched :" << verif_totalNumberOfErrors << " % " << std::endl;

                logStream << "The \"MAXIMUM ABSOLUTE ERROR\" : " << maxABSError << std::endl;
                std::cout << "The \"MAXIMUM ABSOLUTE ERROR\" : " << maxABSError << std::endl;

                logStream << "The \"MAXIMUM REAL ERROR\" : " << maxREALError << std::endl;
                std::cout << "The \"MAXIMUM REAL ERROR\" : " << maxREALError << std::endl;

                logStream << "The \"MAXIMUM IMAG ERROR\" : " << maxIMAGError << std::endl;
                std::cout << "The \"MAXIMUM IMAG ERROR\" : " << maxIMAGError << std::endl;

                //////////////////////////////////////////////////////////////////////////////////////

                logStream << "The \"AVERAGE ABSOLUTE ERROR\" : " << (accumABSError / dataLength) << std::endl;
                std::cout << "The \"AVERAGE ABSOLUTE ERROR\" : " << (accumABSError / dataLength) << std::endl;

                logStream << "The \"AVERAGE REAL ERROR\" : " << (accumREALError / dataLength) << std::endl;
                std::cout << "The \"AVERAGE REAL ERROR\" : " << (accumREALError / dataLength) << std::endl;

                logStream << "The \"AVERAGE IMAG ERROR\" : " << (accumIMAGError / dataLength) << std::endl;
                std::cout << "The \"AVERAGE IMAG ERROR\" : " << (accumIMAGError / dataLength) << std::endl;

                //////////////////////////////////////////////////////////////////////////////////////

                logStream << "The \"MINIMUM REAL  ERROR\" : " << minREALError << std::endl;
                std::cout << "The \"MINIMUM REAL  ERROR\" : " << minREALError << std::endl;

                logStream << "The \"MINIMUM IMAG  ERROR\" : " << minIMAGError << std::endl;
                std::cout << "The \"MINIMUM IMAG  ERROR\" : " << minIMAGError << std::endl;
                calSNR = 10 * log10(totalSignalPower / totalNoisePower);
                std::cout
                    << get_section_header()
                    << "SNR: The signal to noise ratio: ( verification version, compared against Matlab output) :: "
                    << calSNR << " dbs\n";
                print_function_footer();

            } //%%%
        }     ///%%%
    } else    // Cannot Open verification data File SUCCCESSFULLY Error out !!
    {
        logStream << "Cannot Open input data file for verfication..." << std::endl << "Exiting..." << std::endl;
        std::cout << "Cannot Open input data file for verfication..." << std::endl << "Exiting..." << std::endl;
        verif_status = 1;
    }
    verif_result.m_statusFlag = verif_status;
    verif_result.m_snr = calSNR;

    // return verif_status;
    // return verif_result.status_flag;
    return verif_result;
}

// This function will read a file and compare it with expected output for functional verification
// It will print max error, min error, and mean square error for any given comparison
// It will also print log information in a file passed through ofstream
template <typename T_dType>
int verifyArrayOutput_old_print_formating(std::ofstream& logStream,
                                          std::string strArrayName,
                                          std::string fiofLog,
                                          complex_wrapper<T_dType> inData[],
                                          T_dType maxAllowedABSError,
                                          int percentageAllowedError,
                                          int dataLength) {
    logStream << currentDateTime() << std::endl;
    std::cout << currentDateTime() << std::endl;
    logStream << "####################################---" << strArrayName
              << "---#######################################" << std::endl;
    std::cout << "####################################---" << strArrayName
              << "---#######################################" << std::endl;
    logStream << "fiofVerificationStart:" << strArrayName << std::endl
              << "Verifying Array:" << strArrayName << std::endl;
    std::cout << "fiofVerificationStart:" << strArrayName << std::endl
              << "Verifying Array:" << strArrayName << std::endl;
    logStream << "Starting Verification..." << std::endl;
    std::cout << "Starting Verification..." << std::endl;
    std::fstream fiofFile;
    logStream << "Trying to Open :" << fiofLog << std::endl;
    std::cout << "Trying to Open :" << fiofLog << std::endl;
    const char* fiofLogPtr = fiofLog.c_str();
    fiofFile.open(fiofLogPtr, std::ios_base::in);
    int veriFlag = 0;
    int verif_totalNumberOfErrors = 0;
    int sampleCompared = 0;
    int verif_status = 0;
    if (fiofFile.is_open()) /// Try to open the file for data verification
    {
        // Verification data file opened successfully start verification process

        // Verification Start
        logStream << "Opened verfication data file :  " << fiofLog << " <----Successfully----->" << std::endl;
        std::cout << "Opened verfication data file :  " << fiofLog << " <----Successfully----->..." << std::endl;

        // Variables to collect SNR ///////////////////// /////////////////////////////////////////START
        double totalSignalPower = 0;
        double totalNoisePower = 0;
        double calSNR;
        // Variables to collect SNR ///////////////////////////////////////////////////////////////END
        // Variables to collect absolute error statistics /////////////////////////////////////////START
        double currentABSError;
        double maxABSError;
        double accumABSError = 0;
        // Variables to collect absolute error statistics /////////////////////////////////////////END

        // Variables to collect  error statistics of REAL PART ////////////////////////////////////START
        double currentREALError = 0;
        double maxREALError;
        double minREALError;
        double accumREALError = 0;
        // Variables to collect  error statistics of REAL PART //////////////////////////////////////END

        // Variables to collect  error statistics of IMAG PART ////////////////////////////////////START
        double currentIMAGError;
        double maxIMAGError;
        double minIMAGError;
        double accumIMAGError = 0;
        // Variables to collect  error statistics of IMAG PART //////////////////////////////////////END
        // complexNum<T_dType> goldenSample;
        int dataLengthFromFile;
        if (fiofFile.eof()) // Reaing Array Length for Verification: The verification data in the file is not
                            // appropriate, less number of samples
        {
            logStream << "The data length in the verification file is less then the required no. of samples..."
                      << std::endl
                      << "Exiting..." << std::endl;
            std::cout << "The data length in the verification file is less then the required no. of samples..."
                      << std::endl
                      << "Exiting..." << std::endl;
            veriFlag = 1;
        } else // Array Length for Verfication is available for reading... /d
        {
            fiofFile >> dataLengthFromFile;
            if (dataLengthFromFile != dataLength) {
                logStream
                    << "The data length in the file does not matches the data array length passed for verification..."
                    << std::endl
                    << "Exiting..." << std::endl;
                std::cout
                    << "The data length in the file does not matches the data array length passed for verification..."
                    << std::endl
                    << "Exiting..." << std::endl;
                veriFlag = 1;
            } else // d
            {
                for (int i = 0; i < dataLength; i++) {
                    if (fiofFile.eof()) // The verification data in the file is not appropriate, less number of samples
                    {
                        logStream
                            << "The data length in the verification file is less then the required no. of samples..."
                            << std::endl
                            << "Exiting..." << std::endl;
                        std::cout
                            << "The data length in the verification file is less then the required no. of samples..."
                            << std::endl
                            << "Exiting..." << std::endl;
                        veriFlag = 1;
                        break;
                    } else // File opened and verification data available
                    {
                        double goldenSample_real;
                        double goldenSample_imag;
                        fiofFile >> goldenSample_real; // goldenSample.real;
                        fiofFile >> goldenSample_imag; // goldenSample.imag;
                        // goldenSample = complex_wrapper<T_dType>(temp_real,temp_imag);
                        // complexNum<T_dType> temp  = inData[i] - goldenSample;
                        double inData_real, inData_imag;

                        inData_real = inData[i].real();
                        inData_imag = inData[i].imag();
                        currentREALError = inData_real - goldenSample_real;
                        currentIMAGError = inData_imag - goldenSample_imag;
                        totalNoisePower += currentREALError * currentREALError + currentIMAGError * currentIMAGError;
                        totalSignalPower +=
                            goldenSample_real * goldenSample_real + goldenSample_imag * goldenSample_imag;

                        currentABSError =
                            sqrt(currentIMAGError * currentIMAGError + currentREALError * currentREALError);
                        /// kepp track of max and min errors
                        if (i == 0) // first sample read for verification so init maximums and minimums
                        {
                            maxABSError = currentABSError;
                            maxREALError = currentREALError;
                            maxIMAGError = currentIMAGError;
                            minREALError = currentREALError;
                            minIMAGError = currentIMAGError;
                        } else {
                            if (currentABSError < 0) currentABSError *= -1;
                            // Check for maximum errors
                            if (currentABSError > maxABSError) maxABSError = currentABSError;
                            if (currentREALError > maxREALError) maxREALError = currentREALError;
                            if (currentIMAGError > maxIMAGError) maxIMAGError = currentIMAGError;
                            // Check for minimum errors
                            if (currentREALError < minREALError) minREALError = currentREALError;
                            if (currentIMAGError < minIMAGError) minIMAGError = currentIMAGError;
                            double golden_sample_abs =
                                sqrt(goldenSample_imag * goldenSample_imag + goldenSample_real * goldenSample_real);
                            double percent_error = (currentABSError / golden_sample_abs) * 100;
                            if (percent_error > maxAllowedABSError) {
                                logStream << "The data sample different significantly, comparing sample no:" << i
                                          << " The error is : " << percent_error << "%" << std::endl;
                                std::cout << "The data sample different significantly, comparing sample no:" << i
                                          << " The error is : " << percent_error << "%"
                                          << std::endl; //<< "The data sample different significantly, comparing sample
                                                        // no:"<<i<< std::endl;
                                // veriFlag=1;
                                verif_totalNumberOfErrors++;
                            }
                            accumABSError += currentABSError;
                            accumREALError += currentREALError;
                            accumIMAGError += currentIMAGError;
                        }
                    }
                    sampleCompared++;
                }
                int temp_percent_mismatch = int(double(verif_totalNumberOfErrors * 100) / double(dataLength));
                // verification successfull log statistics !!
                if (veriFlag == 0 && sampleCompared == dataLength && temp_percent_mismatch < percentageAllowedError) {
                    logStream << "Verification for data array" << strArrayName << "is SUCCCESSFUL !!" << std::endl;
                    std::cout << "Verification for data array" << strArrayName << "is SUCCCESSFUL !!" << std::endl;
                    verif_status = 0;
                } else {
                    logStream << "Verification for data array :" << strArrayName << "is FAILED !!" << std::endl;
                    std::cout << "Verification for data array : " << strArrayName << "is FAILED !!" << std::endl;
                    verif_status = 1;
                }
                //////////////////////////////////////////////////////////////////////////////////////
                logStream << "************************Error Rate is :" << temp_percent_mismatch
                          << " % ******************" << std::endl;
                std::cout << "************************Error Rate is :" << temp_percent_mismatch
                          << " % ******************" << std::endl;

                logStream << "************************Sample Mismatches :" << verif_totalNumberOfErrors
                          << " % ******************" << std::endl;
                std::cout << "************************Sample Mismatches :" << verif_totalNumberOfErrors
                          << " % ******************" << std::endl;

                logStream << "************************Maximum Error Numbers******************" << std::endl;
                std::cout << "************************Maximum Error Numbers******************" << std::endl;
                logStream << "The \"MAXIMUM ABSOLUTE ERROR\" : " << maxABSError << std::endl;
                std::cout << "The \"MAXIMUM ABSOLUTE ERROR\" : " << maxABSError << std::endl;

                logStream << "The \"MAXIMUM REAL ERROR\" : " << maxREALError << std::endl;
                std::cout << "The \"MAXIMUM REAL ERROR\" : " << maxREALError << std::endl;

                logStream << "The \"MAXIMUM IMAG ERROR\" : " << maxIMAGError << std::endl;
                std::cout << "The \"MAXIMUM IMAG ERROR\" : " << maxIMAGError << std::endl;

                //////////////////////////////////////////////////////////////////////////////////////
                logStream << "************************Average Error Numbers******************" << std::endl;
                std::cout << "************************Average Error Numbers******************" << std::endl;

                logStream << "The \"AVERAGE ABSOLUTE ERROR\" : " << (accumABSError / dataLength) << std::endl;
                std::cout << "The \"AVERAGE ABSOLUTE ERROR\" : " << (accumABSError / dataLength) << std::endl;

                logStream << "The \"AVERAGE REAL ERROR\" : " << (accumREALError / dataLength) << std::endl;
                std::cout << "The \"AVERAGE REAL ERROR\" : " << (accumREALError / dataLength) << std::endl;

                logStream << "The \"AVERAGE IMAG ERROR\" : " << (accumIMAGError / dataLength) << std::endl;
                std::cout << "The \"AVERAGE IMAG ERROR\" : " << (accumIMAGError / dataLength) << std::endl;

                //////////////////////////////////////////////////////////////////////////////////////
                logStream << "************************Minimum Error Numbers******************" << std::endl;
                std::cout << "************************Minimum Error Numbers******************" << std::endl;

                logStream << "The \"MINIMUM REAL  ERROR\" : " << minREALError << std::endl;
                std::cout << "The \"MINIMUM REAL  ERROR\" : " << minREALError << std::endl;

                logStream << "The \"MINIMUM IMAG  ERROR\" : " << minIMAGError << std::endl;
                std::cout << "The \"MINIMUM IMAG  ERROR\" : " << minIMAGError << std::endl;
                calSNR = 10 * log10(totalSignalPower / totalNoisePower);
                std::cout << "-----------------------------------------------------------------------------------------"
                             "-----------------------\n";
                std::cout << "\n\n\n--------The signal to noise ratio : SNR ( verification version, compared against "
                             "Octave output) :: "
                          << calSNR << " dbs \n\n\n";
                std::cout << "-----------------------------------------------------------------------------------------"
                             "-----------------------\n";

            } //%%%
        }     ///%%%
    } else    // Cannot Open verification data File SUCCCESSFULLY Error out !!
    {
        logStream << "Cannot Open input data file for verfication..." << std::endl << "Exiting..." << std::endl;
        std::cout << "Cannot Open input data file for verfication..." << std::endl << "Exiting..." << std::endl;
        verif_status = 1;
    }

    return verif_status;
}
void print_status_bar(int len, int unit) {
    for (int i = 0; i < len / unit; i++) {
        std::cout << "||";
    }
    std::cout << std::endl << std::flush;
}

void print_percent_complete(int i, double val) {
    std::cout << std::flush;
    std::cout << i << ": Loading Completed: " << val << " %";
    std::cout << "\r";
}
template <typename T_dType>
int readComplexArrayFromFile(std::ofstream& logStream,
                             std::string strArrayName,
                             std::string fiofInputFileName,
                             complex_wrapper<T_dType> outData[],
                             int dataLength) {
    print_function_header();
    logStream << currentDateTime() << std::endl;
    std::cout << currentDateTime() << std::endl;
    double max_real, max_imag;
    double min_real, min_imag;
    double sum_real, sum_imag;
    double avg_real, avg_imag;
    // logStream << "####################################---" << strArrayName <<
    // "---#######################################" << std::endl;  std::cout <<
    // "####################################---"
    // << strArrayName << "---#######################################" << std::endl;  logStream <<
    // "fiofVerificationReadInputData:" << strArrayName << std::endl << "Reading Array:" << strArrayName << std::endl;
    // std::cout << "fiofVerificationReadInputData:" << strArrayName << std::endl << "Reading Array:" << strArrayName <<
    // std::endl;  std::cout << "fiofVerificationReadInputData:" << strArrayName << std::endl << "Reading Array:" <<
    // strArrayName << std::endl;
    std::fstream fiofFile;
    logStream << "Trying to open data file: " << fiofInputFileName << std::endl;
    std::cout << "Trying to open data file: " << fiofInputFileName << std::endl;
    const char* fiofInputFileNamePtr = fiofInputFileName.c_str();
    fiofFile.open(fiofInputFileNamePtr, std::ios_base::in);
    int veriFlag = 0;
    if (fiofFile.is_open()) /// Try to open the file for data verification
    {
        logStream << "Opened data file :  " << fiofInputFileName << " Successfully." << std::endl;
        std::cout << "Opened data file :  " << fiofInputFileName << " Successfully." << std::endl;
        if (fiofFile.eof()) // Reaing Array Length for Verification: The verification data in the file is not
                            // appropriate, less number of samples
        {
            logStream << "The data length in the verification file:" << fiofInputFileName
                      << " is less then the required no. of samples..." << std::endl
                      << "Exiting..." << std::endl;
            std::cout << "The data length in the verification file:" << fiofInputFileName
                      << " is less then the required no. of samples..." << std::endl
                      << "Exiting..." << std::endl;
            veriFlag = 1;
        } else // Array Length for Verfication is available for reading... /d
        {
            int dataLengthFromFile;
            fiofFile >> dataLengthFromFile;
            if (dataLengthFromFile != dataLength) {
                logStream << "The data length in the file" << fiofInputFileName
                          << "does not matches the data array length passed for verification..." << std::endl
                          << "Exiting..." << std::endl;
                std::cout << "The data length in the file" << fiofInputFileName
                          << "does not matches the data array length passed for verification..." << std::endl
                          << "Exiting..." << std::endl;
                veriFlag = 1;
            } else // Array Length found appropriate now READ ALL the dat from FILE
            {
                for (int i = 0; i < dataLength; i++) {
                    // File opened data legnth is appropriate !!!
                    if (fiofFile.eof()) // The  data in the file is not appropriate, LESS number of samples
                    {
                        logStream << "The data length in the input data file" << fiofInputFileName
                                  << "is less then the required no. of samples..." << std::endl
                                  << "Exiting..." << std::endl;
                        std::cout << "The data length in the input data file" << fiofInputFileName
                                  << "is less then the required no. of samples..." << std::endl
                                  << "Exiting..." << std::endl;
                        veriFlag = 1;
                        break;
                    } else // File opened and verification data available
                    {
                        double temp_real, temp_imag;
                        fiofFile >> temp_real; // outData[i].real;
                        fiofFile >> temp_imag; // outData[i].imag;
                        double abs_real = temp_real >= 0 ? temp_real : (-1 * temp_real);
                        double abs_imag = temp_imag >= 0 ? temp_imag : (-1 * temp_imag);
                        if (i == 0) {
                            max_real = temp_real;
                            max_imag = temp_imag;

                            min_real = temp_real;
                            min_imag = temp_imag;

                            sum_real = temp_real;
                            sum_imag = temp_imag;

                        } else {
                            if (max_real < abs_real) max_real = abs_real;
                            if (max_imag < abs_imag) max_imag = abs_imag;
                            if (min_real > abs_real) min_real = abs_real;
                            if (min_imag > abs_imag) min_imag = abs_imag;
                            sum_real += abs_real;
                            sum_imag += abs_imag;
                        }
                        outData[i] = complex_wrapper<T_dType>(temp_real, temp_imag);
                        if ((i % 100) == 0) {
                            double pcl = 100 * double(i) / double(dataLength);
                            print_percent_complete(i, pcl);
                        }
                    }
                }
            }
        }

    } else // Cannot Open verification data File SUCCCESSFULLY Error out !!
    {
        logStream << "Cannot Open input data file for stimulus..." << fiofInputFileName << std::endl
                  << "Exiting..." << std::endl;
        std::cout << "Cannot Open input data file for stimulus..." << fiofInputFileName << std::endl
                  << "Exiting..." << std::endl;
        veriFlag = 1;
    }
    fiofFile.close();
    std::cout << get_section_header() << "Data Statistics:\n";
    std::cout << "Max real: " << max_real << std::endl;
    std::cout << "Max imag: " << max_imag << std::endl;
    std::cout << "Min real: " << min_real << std::endl;
    std::cout << "Min imag: " << min_imag << std::endl;
    std::cout << "Avg real: " << sum_real / dataLength << std::endl;
    std::cout << "Avg real: " << sum_imag / dataLength << std::endl;
    if (max_real > 1 || max_imag > 1) {
        std::cout << "The Data in not normalized\n";
    }
    print_function_footer();
    return veriFlag;
}

template <typename T_dType>
int readComplexArrayFromFile(std::string fiofInputFileName, complex_wrapper<T_dType> outData[], int dataLength) {
    print_function_header();
    std::cout << currentDateTime() << std::endl;
    double max_real, max_imag;
    double min_real, min_imag;
    double sum_real, sum_imag;
    double avg_real, avg_imag;
    std::fstream fiofFile;
    std::cout << "Trying to open data file: " << fiofInputFileName << std::endl;
    const char* fiofInputFileNamePtr = fiofInputFileName.c_str();
    fiofFile.open(fiofInputFileNamePtr, std::ios_base::in);
    int veriFlag = 0;
    if (fiofFile.is_open()) /// Try to open the file for data verification
    {
        std::cout << "Opened data file :  " << fiofInputFileName << " Successfully." << std::endl;
        if (fiofFile.eof()) // Reaing Array Length for Verification: The verification data in the file is not
                            // appropriate, less number of samples
        {
            std::cout << "The data length in the verification file:" << fiofInputFileName
                      << " is less then the required no. of samples..." << std::endl
                      << "Exiting..." << std::endl;
            veriFlag = 1;
        } else // Array Length for Verfication is available for reading... /d
        {
            int dataLengthFromFile;
            fiofFile >> dataLengthFromFile;
            if (dataLengthFromFile != dataLength) {
                std::cout << "The data length in the file" << fiofInputFileName
                          << "does not matches the data array length passed for verification..." << std::endl
                          << "Exiting..." << std::endl;
                veriFlag = 1;
            } else // Array Length found appropriate now READ ALL the dat from FILE
            {
                for (int i = 0; i < dataLength; i++) {
                    // File opened data legnth is appropriate !!!
                    if (fiofFile.eof()) // The  data in the file is not appropriate, LESS number of samples
                    {
                        std::cout << "The data length in the input data file" << fiofInputFileName
                                  << "is less then the required no. of samples..." << std::endl
                                  << "Exiting..." << std::endl;
                        veriFlag = 1;
                        break;
                    } else // File opened and verification data available
                    {
                        double temp_real, temp_imag;
                        fiofFile >> temp_real; // outData[i].real;
                        fiofFile >> temp_imag; // outData[i].imag;
                        double abs_real = temp_real >= 0 ? temp_real : (-1 * temp_real);
                        double abs_imag = temp_imag >= 0 ? temp_imag : (-1 * temp_imag);
                        if (i == 0) {
                            max_real = temp_real;
                            max_imag = temp_imag;

                            min_real = temp_real;
                            min_imag = temp_imag;

                            sum_real = temp_real;
                            sum_imag = temp_imag;

                        } else {
                            if (max_real < abs_real) max_real = abs_real;
                            if (max_imag < abs_imag) max_imag = abs_imag;
                            if (min_real > abs_real) min_real = abs_real;
                            if (min_imag > abs_imag) min_imag = abs_imag;
                            sum_real += abs_real;
                            sum_imag += abs_imag;
                        }
                        outData[i] = complex_wrapper<T_dType>(temp_real, temp_imag);
                        if ((i % 100) == 0) {
                            double pcl = 100 * double(i) / double(dataLength);
                            print_percent_complete(i, pcl);
                        }
                    }
                }
            }
        }

    } else // Cannot Open verification data File SUCCCESSFULLY Error out !!
    {
        std::cout << "Cannot Open input data file for stimulus..." << fiofInputFileName << std::endl
                  << "Exiting..." << std::endl;
        veriFlag = 1;
    }
    fiofFile.close();
    std::cout << get_section_header() << "Data Statistics:\n";
    std::cout << "Max real: " << max_real << std::endl;
    std::cout << "Max imag: " << max_imag << std::endl;
    std::cout << "Min real: " << min_real << std::endl;
    std::cout << "Min imag: " << min_imag << std::endl;
    std::cout << "Avg real: " << sum_real / dataLength << std::endl;
    std::cout << "Avg real: " << sum_imag / dataLength << std::endl;
    if (max_real > 1 || max_imag > 1) {
        std::cout << "The Data in not normalized\n";
    }
    print_function_footer();
    return veriFlag;
}

template <int size_dim1, int size_dim2, int offset, int dim, typename T_dtype>
bool verifyContineous2DArray(T_dtype data[size_dim1][size_dim2]) {
    if (dim == 1) {
        T_dtype num = offset;
        for (int d2 = 0; d2 < size_dim2; d2++) {
            for (int d1 = 0; d1 < size_dim1; d1++) {
                if (num != data[d1][d2])
                    return false;
                else
                    num++;
            }
        }
    } else if (dim == 2) {
        T_dtype num = offset;
        for (int d1 = 0; d1 < size_dim1; d1++) {
            for (int d2 = 0; d2 < size_dim2; d2++) {
                if (num != data[d1][d2])
                    return false;
                else
                    num++;
            }
        }
    } else
        return false;

    return true;
}

#endif // !MVERIFICATIONUTLITYFUNCTIONS_H_
